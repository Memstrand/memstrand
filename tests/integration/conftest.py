"""Fixtures shared by the real-process integration suite: a real `memstrand
serve` daemon subprocess, a CLI-subprocess helper, and the OpenAI-compatible
stub server (`stub_openai.StubOpenAI`).

Hermeticity: every subprocess environment is built explicitly (PATH, a
sandboxed HOME, an explicit MEMSTRAND_HOME, plus whatever MEMSTRAND_* this
test wants) -- never `os.environ` copied wholesale -- so nothing ambient
(a real `~/.memstrand`, a stray MEMSTRAND_TOKEN in the CI runner's own
environment, ...) can leak into a child process. `tests/conftest.py`'s
autouse `_hermetic_env` fixture already does the equivalent for the
in-process unit suite; this mirrors it for subprocesses, which don't
inherit fixture-level `monkeypatch` changes.

Collection gate: real subprocess/socket tests are slow and not meant to
run as part of the fast unit suite's bare `pytest -q`. `pytest_collection_
modifyitems` below deselects every test under `tests/integration` unless
`tests/integration` is explicitly named on the command line -- CI's
dedicated `integration` job (`pytest -q tests/integration`) still runs
them; the main `test` job's bare `pytest -q` (run across the OS/Python
matrix) does not. This lives here, in an owned file, rather than touching
the shared root `tests/conftest.py` or adding pytest config that would
affect every invocation uniformly.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest

from .stub_openai import StubOpenAI, free_port

REPO_ROOT = Path(__file__).resolve().parents[2]

__all__ = [
    "REPO_ROOT",
    "DaemonHandle",
    "base_env",
    "free_port",
    "memstrand_argv",
    "run_cli",
]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    args = [str(a) for a in config.invocation_params.args]
    explicit = any("tests/integration" in a or a.endswith("tests/integration") for a in args)
    if explicit:
        return
    skip = pytest.mark.skip(
        reason="integration tests only run when tests/integration is explicitly "
        "targeted on the command line (see tests/integration/conftest.py)"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


# ======================================================================================
# subprocess invocation helpers
# ======================================================================================


def _memstrand_console_script() -> str | None:
    """The `memstrand` console script `pip install -e` drops next to this
    interpreter, if there is one. `cli.py` has no `if __name__ ==
    "__main__":` guard, so `python -m memstrand.cli` is a silent no-op --
    the console script (or the `-c` shim below) is the only way to invoke
    `cli.main()` as a subprocess."""
    for candidate in (Path(sys.executable).parent / "memstrand", Path(sys.executable).parent / "memstrand.exe"):
        if candidate.exists():
            return str(candidate)
    return None


def memstrand_argv(*args: str) -> list[str]:
    script = _memstrand_console_script()
    if script:
        return [script, *args]
    return [sys.executable, "-c", "import sys; from memstrand.cli import main; sys.exit(main())", *args]


def base_env(memstrand_home: Path, **overrides: str) -> dict[str, str]:
    """A minimal, explicit subprocess environment: PATH, a sandboxed HOME
    (sibling of `memstrand_home`, never the real one), MEMSTRAND_HOME, plus
    whatever this test wants to set. Never `os.environ` wholesale."""
    fake_home = memstrand_home.parent / "fake-home"
    env: dict[str, str] = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(fake_home),
        "MEMSTRAND_HOME": str(memstrand_home),
    }
    # Needed on some platforms/CI runners for the interpreter / sockets /
    # encoding to behave the same as the parent process.
    for key in ("SYSTEMROOT", "PYTHONPATH", "PYTHONIOENCODING", "LANG", "LC_ALL"):
        if key in os.environ:
            env[key] = os.environ[key]
    env.update(overrides)
    return env


def run_cli(
    args: list[str],
    env: dict[str, str],
    *,
    timeout: float = 30.0,
    input: str | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        memstrand_argv(*args),
        env=env,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        input=input,
        check=False,
    )


# ======================================================================================
# daemon fixture: a real `memstrand serve` subprocess
# ======================================================================================


def _drain(stream, sink: list[str]) -> threading.Thread:
    def _read():
        sink.extend(iter(stream.readline, ""))

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    return t


@dataclass
class DaemonHandle:
    proc: subprocess.Popen
    url: str
    port: int
    home: Path
    env: dict[str, str]
    token: str | None = None
    stderr_lines: list[str] = field(default_factory=list)
    stdout_lines: list[str] = field(default_factory=list)

    def wait_health(self, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else None
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"daemon exited early (code {self.proc.returncode}) before becoming healthy:\n{self.tail()}"
                )
            try:
                r = httpx.get(f"{self.url}/api/health", timeout=1.0, headers=headers)
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        raise TimeoutError(f"daemon at {self.url} never became healthy:\n{self.tail()}")

    def tail(self, n: int = 40) -> str:
        return "".join(self.stderr_lines[-n:]) or "(no stderr captured)"

    def stop(self, timeout: float = 5.0) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=timeout)

    def kill9(self, timeout: float = 5.0) -> None:
        """SIGKILL -- for crash-safety tests, no chance for the process to
        clean up (close stores, release locks) on its own."""
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=timeout)


@pytest.fixture
def start_daemon(tmp_path):
    """Factory fixture: `start_daemon(**opts) -> DaemonHandle`. Call it
    more than once per test for scenarios that need two daemons (e.g. a
    second fresh one to import an export into)."""
    handles: list[DaemonHandle] = []
    counter = {"n": 0}

    def _start(
        *,
        home: Path | None = None,
        port: int | None = None,
        token: str | None = None,
        stub: StubOpenAI | None = None,
        llm_base_url: str | None = None,
        embed_base_url: str | None = None,
        embed_backend: str | None = "openai",
        embed_model: str | None = "stub-embed",
        llm_model: str | None = "stub-llm",
        extra_env: dict[str, str] | None = None,
        scheduler: bool = False,
        wait: bool = True,
        health_timeout: float = 20.0,
    ) -> DaemonHandle:
        n = counter["n"]
        counter["n"] += 1
        home = home or (tmp_path / f"daemon-home-{n}")
        port = port or free_port()

        env = base_env(home, MEMSTRAND_HOST="127.0.0.1", MEMSTRAND_PORT=str(port))
        if embed_backend:
            env["MEMSTRAND_EMBED_BACKEND"] = embed_backend
        if embed_model:
            env["MEMSTRAND_EMBED_MODEL"] = embed_model
        if llm_model:
            env["MEMSTRAND_LLM_MODEL"] = llm_model
        resolved_llm_url = llm_base_url or (stub.url if stub is not None else None)
        resolved_embed_url = embed_base_url or (stub.url if stub is not None else None)
        if resolved_llm_url:
            env["MEMSTRAND_LLM_BASE_URL"] = resolved_llm_url
        if resolved_embed_url:
            env["MEMSTRAND_EMBED_BASE_URL"] = resolved_embed_url
        if token:
            env["MEMSTRAND_TOKEN"] = token
        if extra_env:
            env.update(extra_env)

        args = ["serve"]
        if not scheduler:
            args.append("--no-scheduler")

        proc = subprocess.Popen(
            memstrand_argv(*args),
            env=env,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        handle = DaemonHandle(
            proc=proc, url=f"http://127.0.0.1:{port}", port=port, home=home, env=env, token=token
        )
        _drain(proc.stdout, handle.stdout_lines)
        _drain(proc.stderr, handle.stderr_lines)
        handles.append(handle)
        if wait:
            handle.wait_health(timeout=health_timeout)
        return handle

    yield _start

    for h in handles:
        h.stop()


def daemon_client_env(daemon: DaemonHandle, *, port: int | None = None, token=True, **overrides) -> dict[str, str]:
    """Env for a CLI subprocess acting as a client of `daemon`. Uses the
    SAME MEMSTRAND_HOME as the daemon by default (the realistic case: an
    operator's CLI shares a home with the daemon they're talking to) --
    this matters for the "wrong port" test, which relies on a CLI that
    can't reach the daemon falling back to a *local* Memory() on that same,
    already-locked home and getting a clean wrapped error instead of a
    silent, wrong success.

    `token`: `True` (default) uses the daemon's own token (if any); a
    falsy value (`None`/`False`) omits it entirely (to exercise the
    "wrong/missing token" rejection path); a string uses that exact value.
    """
    resolved_token = daemon.token if token is True else (token or None)
    env = base_env(
        daemon.home, MEMSTRAND_HOST="127.0.0.1", MEMSTRAND_PORT=str(port if port is not None else daemon.port)
    )
    if resolved_token:
        env["MEMSTRAND_TOKEN"] = resolved_token
    env.update(overrides)
    return env


# ======================================================================================
# the stub OpenAI-compatible server
# ======================================================================================


@pytest.fixture
def stub():
    srv = StubOpenAI()
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


@pytest.fixture
def stub_factory():
    """For tests that need to pre-reserve a port before the stub is
    actually listening there (e.g. "daemon starts with the embedding
    backend down, then it comes up")."""
    created: list[StubOpenAI] = []

    def _make(port: int | None = None) -> StubOpenAI:
        srv = StubOpenAI(port=port)
        created.append(srv)
        return srv

    yield _make

    for srv in created:
        srv.stop()
