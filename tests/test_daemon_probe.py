"""Tri-state daemon detection (`client.probe_daemon`) against *real* local
HTTP servers -- not monkeypatched `httpx.get` -- plus every call site that
must branch on `"up"` / `"rejected"` / `"down"` instead of the old boolean.

The bug this guards against: `daemon_running()` used to collapse any
non-200 response (including 401 for a missing/wrong `MEMSTRAND_TOKEN`) into
`False` ("no daemon"), so a client with the wrong token would fall through
to opening a *local* `Memory()` against stores a real daemon already holds
-- crashing on Qdrant's on-disk "already accessed by another instance"
lock instead of reporting the actual problem (bad token). `"rejected"` must
always be handled like `"up"` (something owns the stores), never like
`"down"`.

Real sockets are used throughout (stdlib `http.server`, bound to
`127.0.0.1:0`) rather than mocking `httpx`, so these tests exercise the
real transport-error paths (connection refused) that a mocked transport
can't reproduce.
"""

from __future__ import annotations

import http.server
import threading
from contextlib import contextmanager

import pytest

from memstrand import cli, mcp_server
from memstrand.client import DaemonRejected, RemoteMemory, get_memory, probe_daemon
from memstrand.config import get_settings

# ======================================================================================
# a configurable-status-code /api/health stub, bound to an OS-assigned free port
# ======================================================================================


class _HealthStubHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        status = self.server.status_code
        body = b'{"ok": true}' if status == 200 else b'{"error": "nope"}'
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # keep test output clean


@contextmanager
def _health_server(status_code: int):
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _HealthStubHandler)
    httpd.status_code = status_code
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        thread.join(timeout=2)
        httpd.server_close()


def _closed_port() -> int:
    """A port nothing is listening on: bind to let the OS pick a free one,
    then close it immediately. Tiny reuse-race window, standard practice
    for "nothing here" test fixtures."""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ======================================================================================
# probe_daemon: 200 / 401 / 500 / closed port
# ======================================================================================


def test_probe_daemon_up_on_200():
    with _health_server(200) as httpd:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        assert probe_daemon(url) == "up"


def test_probe_daemon_rejected_on_401():
    with _health_server(401) as httpd:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        assert probe_daemon(url) == "rejected"


def test_probe_daemon_rejected_on_500():
    # A 5xx is still "something answered and didn't give us a clean 200" --
    # tri-state collapses every non-200 into "rejected", never "down".
    with _health_server(500) as httpd:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        assert probe_daemon(url) == "rejected"


def test_probe_daemon_down_on_closed_port():
    url = f"http://127.0.0.1:{_closed_port()}"
    assert probe_daemon(url) == "down"


# ======================================================================================
# get_memory(): DaemonRejected vs RemoteMemory vs local fallback
# ======================================================================================


def test_get_memory_raises_daemon_rejected_against_real_server(monkeypatch):
    with _health_server(401) as httpd:
        monkeypatch.setenv("MEMSTRAND_HOST", "127.0.0.1")
        monkeypatch.setenv("MEMSTRAND_PORT", str(httpd.server_address[1]))
        monkeypatch.setattr(
            "memstrand.api.Memory",
            lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not construct a local Memory")),
        )
        with pytest.raises(DaemonRejected) as exc_info:
            get_memory()
        message = str(exc_info.value)
        assert "rejected this client" in message
        assert "MEMSTRAND_TOKEN" in message


def test_get_memory_returns_remote_memory_against_real_server(monkeypatch):
    with _health_server(200) as httpd:
        monkeypatch.setenv("MEMSTRAND_HOST", "127.0.0.1")
        monkeypatch.setenv("MEMSTRAND_PORT", str(httpd.server_address[1]))
        mem = get_memory()
        try:
            assert isinstance(mem, RemoteMemory)
        finally:
            mem.close()


def test_get_memory_opens_local_when_port_is_closed(monkeypatch):
    monkeypatch.setenv("MEMSTRAND_HOST", "127.0.0.1")
    monkeypatch.setenv("MEMSTRAND_PORT", str(_closed_port()))
    sentinel = object()
    monkeypatch.setattr("memstrand.api.Memory", lambda **kwargs: sentinel)
    assert get_memory() is sentinel


# ======================================================================================
# hooks: a rejected daemon must never construct a local Memory, and must log
# the specific, actionable hook.log line instead of losing the turn silently
# ======================================================================================


def test_hook_claude_code_on_rejected_daemon_logs_and_skips_without_opening_memory(monkeypatch, capsys):
    with _health_server(401) as httpd:
        monkeypatch.setenv("MEMSTRAND_HOST", "127.0.0.1")
        monkeypatch.setenv("MEMSTRAND_PORT", str(httpd.server_address[1]))
        monkeypatch.setattr(
            "memstrand.api.Memory",
            lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not construct a local Memory")),
        )

        rc = cli.main(["hook", "claude-code"])

        assert rc == 0
        out, err = capsys.readouterr()
        # Stop/SessionEnd/PreCompact hooks must print nothing, success or not.
        assert out == ""
        assert err == ""

        log_path = get_settings().home / "hook.log"
        assert log_path.exists()
        log_text = log_path.read_text()
        assert (
            "daemon requires MEMSTRAND_TOKEN which this hook does not have — "
            "turns NOT logged; they will be picked up on the next run once fixed"
        ) in log_text


def test_hook_claude_code_context_on_rejected_daemon_prints_nothing(monkeypatch, capsys):
    # UserPromptSubmit/SessionStart: only queries memory when probe_daemon()
    # == "up" -- "rejected" must fall through the same as "down" here (no
    # context line printed), not crash on an unexpectedly-opened local Memory.
    with _health_server(401) as httpd:
        monkeypatch.setenv("MEMSTRAND_HOST", "127.0.0.1")
        monkeypatch.setenv("MEMSTRAND_PORT", str(httpd.server_address[1]))
        monkeypatch.setattr(
            "memstrand.api.Memory",
            lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not construct a local Memory")),
        )

        rc = cli.main(["hook", "claude-code-context"])

        assert rc == 0
        out, _err = capsys.readouterr()
        assert out == ""


# ======================================================================================
# CLI: a rejected daemon must surface a clean, actionable error (exit 1,
# no traceback) instead of the old silent-fallback-then-crash behavior
# ======================================================================================


def test_cli_ingest_watch_on_rejected_daemon_exits_1_with_clean_error(monkeypatch, capsys):
    with _health_server(401) as httpd:
        monkeypatch.setenv("MEMSTRAND_HOST", "127.0.0.1")
        monkeypatch.setenv("MEMSTRAND_PORT", str(httpd.server_address[1]))

        rc = cli.main(["ingest", "/does/not/matter", "--watch"])

        assert rc == 1
        out, err = capsys.readouterr()
        assert out == ""
        assert err.startswith("error: ")
        assert "rejected this client" in err
        assert "MEMSTRAND_TOKEN" in err
        # a clean, expected error -- never a raw traceback to the user
        assert "Traceback" not in err


# ======================================================================================
# mcp_server.main(): must refuse to start on "rejected" exactly like "up"
# ======================================================================================


def test_mcp_main_exits_1_when_daemon_rejects(monkeypatch, capsys):
    monkeypatch.setattr(mcp_server, "probe_daemon", lambda *a, **k: "rejected")
    monkeypatch.setattr(
        mcp_server,
        "open_local_memory",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not open local stores")),
    )

    with pytest.raises(SystemExit) as exc_info:
        mcp_server.main()

    assert exc_info.value.code == 1
    _out, err = capsys.readouterr()
    assert "stdio" in err
    assert "MEMSTRAND_TOKEN" in err


def test_mcp_main_exits_1_when_daemon_up(monkeypatch, capsys):
    monkeypatch.setattr(mcp_server, "probe_daemon", lambda *a, **k: "up")
    monkeypatch.setattr(
        mcp_server,
        "open_local_memory",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not open local stores")),
    )

    with pytest.raises(SystemExit) as exc_info:
        mcp_server.main()

    assert exc_info.value.code == 1
    _out, err = capsys.readouterr()
    assert "stdio" in err
    # the plain "up" case has no token hint -- only "rejected" does
    assert "MEMSTRAND_TOKEN" not in err


def test_mcp_main_runs_when_daemon_down(monkeypatch):
    monkeypatch.setattr(mcp_server, "probe_daemon", lambda *a, **k: "down")
    sentinel = object()
    ran = {}

    class _FakeMCP:
        def run(self):
            ran["called"] = True

    monkeypatch.setattr(mcp_server, "build_mcp", lambda lazy_memory: _FakeMCP())
    monkeypatch.setattr(mcp_server, "_mem", None)
    monkeypatch.setattr(mcp_server, "open_local_memory", lambda *a, **k: sentinel)

    mcp_server.main()

    assert ran.get("called") is True
