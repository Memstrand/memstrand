"""Scenario #1: the daemon must start fine with its embedding backend
down, answer `/api/health` regardless, surface a clear "embedding backend
unreachable" error on the first `remember` while it's down, and then
succeed -- without a restart -- once the backend comes up. `Memory`
constructs its vector store eagerly, at `Memory.__init__` time (so a
locked on-disk Qdrant store, `StoreLocked`, fails fast at daemon startup
instead of on first use) -- but that construction never contacts the
embedding backend itself; embedding only happens per-call
(`remember`/`recall`/...), which is why the daemon can still start, and
answer `/api/health`, with the embedding backend down. See `api.py`'s
`Memory.__init__`/`embeddings.py`'s `OpenAIEmbedder`.

Scenario #9 (`test_second_daemon_on_same_home_fails_fast_with_store_locked_error`,
below): a second `memstrand serve` pointed at a home a live daemon
already holds must fail the same way a CLI client does -- fast, with the
`StoreLocked` message, exit 1, no ready banner, no traceback -- instead of
hanging, racing the first daemon's stores, or crashing with a raw Qdrant
lock string."""

from __future__ import annotations

import subprocess

import httpx
import pytest

from .conftest import REPO_ROOT, base_env, daemon_client_env, memstrand_argv, run_cli
from .stub_openai import free_port

pytestmark = pytest.mark.integration


def test_daemon_survives_embedding_backend_down_then_recovers(start_daemon, stub_factory):
    # Reserve a port but do NOT start the stub there yet -- the daemon
    # must come up anyway and only fail the embedding-dependent call.
    embed_port = free_port()
    stub = stub_factory(port=embed_port)

    daemon = start_daemon(stub=None, embed_base_url=f"http://127.0.0.1:{embed_port}/v1")

    # /api/health never touches the embedder -- must be fine immediately.
    r = httpx.get(f"{daemon.url}/api/health", timeout=5.0)
    assert r.status_code == 200
    assert r.json()["ok"] is True

    env = daemon_client_env(daemon)
    result = run_cli(["remember", "the office wifi is hunter2"], env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "error: " in result.stderr
    assert "unreachable" in result.stderr
    assert "Traceback" not in result.stderr

    # Now bring the embedding backend up, on the exact port the daemon was
    # already configured to use -- no daemon restart.
    stub.start()
    try:
        result = run_cli(["remember", "the office wifi is hunter2"], env)
        assert result.returncode == 0, result.stdout + result.stderr
        atom_id = result.stdout.strip()
        assert atom_id.startswith("mem:")

        result = run_cli(["recall", "office wifi"], env)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "hunter2" in result.stdout
    finally:
        stub.stop()


def test_second_daemon_on_same_home_fails_fast_with_store_locked_error(start_daemon, stub):
    """A second `memstrand serve` pointed at a home a live daemon already
    holds must fail fast with the `StoreLocked` message (exit 1, one clean
    stderr line, no ready banner, no traceback) instead of hanging, racing
    the first daemon's embedded Qdrant/SQLite stores, or crashing with a
    raw unwrapped Qdrant lock string. `server.serve()` catches `StoreLocked`
    (raised eagerly by `Memory.__init__` when the on-disk Qdrant lock is
    already held) around `Memory()`+`create_app()` construction, prints
    `error: ...`, and exits 1 before `uvicorn.run()` (and therefore before
    printing the "memstrand daemon listening on ..." banner) is ever
    reached."""
    daemon = start_daemon(stub=stub)

    # A second `serve` on the SAME MEMSTRAND_HOME, a different port (the
    # port isn't what's locked -- the home's on-disk Qdrant store is).
    second_port = free_port()
    env = base_env(
        daemon.home,
        MEMSTRAND_HOST="127.0.0.1",
        MEMSTRAND_PORT=str(second_port),
        MEMSTRAND_EMBED_BACKEND="openai",
        MEMSTRAND_EMBED_BASE_URL=stub.url,
        MEMSTRAND_EMBED_MODEL="stub-embed",
        MEMSTRAND_LLM_BASE_URL=stub.url,
        MEMSTRAND_LLM_MODEL="stub-llm",
    )
    result = subprocess.run(
        memstrand_argv("serve", "--no-scheduler"),
        env=env,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=20.0,
        check=False,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "error: " in result.stderr
    assert "locked by another process" in result.stderr
    assert "already accessed" not in result.stderr
    assert "Traceback" not in result.stderr
    # no ready banner -- it never got that far
    assert "memstrand daemon listening" not in result.stdout

    # the first daemon is unaffected by the failed second attempt
    r = httpx.get(f"{daemon.url}/api/health", timeout=5.0)
    assert r.status_code == 200
