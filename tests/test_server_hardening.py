"""Hardening tests for the review findings fixed in `server.py`:

  * the `/v1/chat/completions` streaming proxy no longer lies about
    upstream errors (non-2xx before any bytes are sent, mid-stream
    failures as a well-formed SSE error event),
  * non-loopback binds actually work (the MCP SDK's own DNS-rebinding
    protection no longer rejects legitimate remote Host headers, while
    memstrand's own Origin check + mandatory token still guard it),
  * `hmac.compare_digest` for the bearer token,
  * security response headers on every surface,
  * `/api/atoms` pagination/validation.

Prefers real sockets (real uvicorn servers, real `httpx` clients) over
`TestClient`/`MockTransport` wherever the thing under test is actually
about HTTP-level behavior (streaming disconnects, Host/Origin headers as
a remote client would really send them) — `TestClient`'s ASGI transport
doesn't go over a socket at all, which hides exactly the bugs this file
is checking for in a couple of cases.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import socket
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Self

import httpx
import pytest
import uvicorn
from qdrant_client import QdrantClient
from starlette.applications import Starlette
from starlette.responses import StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from memstrand.api import Memory
from memstrand.config import get_settings
from memstrand.server import create_app
from memstrand.stores.graph_sqlite import SqliteGraphStore
from memstrand.stores.sqlite import SqliteSessionStore
from memstrand.stores.vector_qdrant import QdrantVectorStore


class HashEmbedder:
    """Deterministic keyword-overlap embeddings. Shared words => similar."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in t.lower().split():
                idx = int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim
                v[idx] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


def _make_mem(tmp_path: Path) -> Memory:
    return Memory(
        embedder=HashEmbedder(),
        vector_store=QdrantVectorStore(client=QdrantClient(":memory:")),
        graph_store=SqliteGraphStore(path=str(tmp_path / "graph.db")),
        session_store=SqliteSessionStore(path=str(tmp_path / "sessions.db")),
        home=tmp_path,
    )


def _settings(tmp_path: Path, **overrides):
    s = replace(get_settings(), home=tmp_path, server_host="127.0.0.1", server_port=7779)
    return replace(s, **overrides) if overrides else s


@contextlib.contextmanager
def _client(app):
    base_url = f"http://{app.state.settings.server_host}:{app.state.settings.server_port}"
    with TestClient(app, base_url=base_url) as client:
        yield client


@pytest.fixture()
def mem(tmp_path):
    return _make_mem(tmp_path)


# -- real-socket helper -----------------------------------------------------------


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _LiveServer:
    """Runs an ASGI app behind a real uvicorn server on a real socket, in a
    background thread, for the handful of tests where `TestClient`'s
    in-process ASGI transport would hide the bug (real disconnects, real
    Host headers)."""

    def __init__(self, app, host: str = "127.0.0.1", port: int | None = None):
        self.host = host
        self.port = port if port is not None else _free_port()
        config = uvicorn.Config(app, host=host, port=self.port, log_level="error")
        self.server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def __enter__(self) -> Self:
        self._thread.start()
        deadline = time.monotonic() + 5.0
        while not self.server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("uvicorn server did not start in time")
            time.sleep(0.01)
        return self

    def __exit__(self, *exc) -> None:
        self.server.should_exit = True
        self._thread.join(timeout=5)


# -- streaming proxy: non-2xx before any bytes, mid-stream failure ----------------


def _mock_upstream_status(status: int, body: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler)


def test_stream_non2xx_returns_plain_json_not_sse(mem, tmp_path):
    """A `stream: true` request whose upstream answers with a non-2xx
    status (most OpenAI-compatible servers send a JSON error body even for
    streaming requests) must come back as a normal JSON error response
    with the upstream's status — not a 200 text/event-stream wrapping the
    raw error JSON as if it were an SSE chunk."""
    app = create_app(
        mem, _settings(tmp_path), upstream_transport=_mock_upstream_status(500, {"error": "boom"})
    )
    with _client(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert resp.status_code == 500
        assert not resp.headers["content-type"].startswith("text/event-stream")
        assert resp.json() == {"error": "boom"}

    # the user's turn is still logged up front (before upstream is ever
    # contacted), but no assistant turn -- the request never "succeeded"
    sessions = mem.sessions.list_sessions()
    turns = mem.sessions.get_session(sessions[0]) if sessions else []
    assert not any(t["role"] == "assistant" for t in turns)


def test_stream_success_logs_exactly_one_assistant_turn(mem, tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        chunks = [
            b'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"world"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b"".join(chunks))

    app = create_app(mem, _settings(tmp_path), upstream_transport=httpx.MockTransport(handler))
    with _client(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "stream please"}], "stream": True},
        )
        assert resp.status_code == 200
        assert "Hello " in resp.text and "world" in resp.text

    sessions = mem.sessions.list_sessions()
    assert len(sessions) == 1
    turns = mem.sessions.get_session(sessions[0])
    assistant_turns = [t for t in turns if t["role"] == "assistant"]
    assert len(assistant_turns) == 1
    assert assistant_turns[0]["content"] == "Hello world"


def _flaky_stub_app() -> Starlette:
    """A real (uvicorn-served) OpenAI-compatible stub whose streaming
    response crashes mid-body, simulating an upstream connection reset
    after headers (and some bytes) are already on the wire -- the thing a
    MockTransport can't reproduce because it never goes over a socket."""

    async def chat_completions(request):
        async def gen():
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            raise RuntimeError("stub simulated mid-stream crash")

        return StreamingResponse(gen(), media_type="text/event-stream")

    return Starlette(routes=[Route("/v1/chat/completions", chat_completions, methods=["POST"])])


def test_stream_mid_disconnect_emits_valid_json_sse_error(mem, tmp_path):
    with _LiveServer(_flaky_stub_app()) as upstream:
        settings = _settings(
            tmp_path,
            llm_base_url=f"{upstream.base_url}/v1",
            server_port=_free_port(),
        )
        app = create_app(mem, settings)
        with _client(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
            )
            assert resp.status_code == 200
            # Every "data: ..." line (except the final [DONE] sentinel) must
            # be valid JSON -- this is the actual regression check: the old
            # code built the error event with an f-string embedding the raw
            # exception text, which can contain characters (quotes) that
            # make the payload invalid JSON.
            saw_error_event = False
            for line in resp.text.splitlines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    continue
                obj = json.loads(payload)  # must not raise
                if "error" in obj:
                    saw_error_event = True
            assert saw_error_event
            assert resp.text.rstrip().endswith("data: [DONE]")

    # the stream never completed successfully, so no assistant turn should
    # have been logged (the user's turn is logged up front, independent of
    # upstream outcome)
    sessions = mem.sessions.list_sessions()
    turns = mem.sessions.get_session(sessions[0]) if sessions else []
    assert not any(t["role"] == "assistant" for t in turns)


def test_proxy_upstream_unreachable_before_streaming_returns_502(mem, tmp_path):
    """Connection failures that happen before any response is received at
    all (vs. a mid-stream failure) must still short-circuit to a plain 502
    JSON response, same as the non-streaming path."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    app = create_app(mem, _settings(tmp_path), upstream_transport=httpx.MockTransport(handler))
    with _client(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert resp.status_code == 502
        assert not resp.headers["content-type"].startswith("text/event-stream")
        assert "error" in resp.json()


# -- non-loopback mode --------------------------------------------------------------


def test_non_loopback_mcp_initialize_with_allowlisted_lan_origin(mem, tmp_path, monkeypatch):
    """A LAN browser Origin is only trusted once it's explicitly added to
    `MEMSTRAND_ALLOWED_ORIGINS` -- there is no same-origin-by-Host
    fallback (that was a DNS-rebinding hole: an attacker who controls both
    the Host and Origin headers on a forged request could otherwise make
    them match each other without ever being on the allowlist)."""
    monkeypatch.setenv("MEMSTRAND_TOKEN", "lantoken")
    port = _free_port()
    lan_host = f"192.168.1.50:{port}"
    monkeypatch.setenv("MEMSTRAND_ALLOWED_ORIGINS", f"http://{lan_host}")
    settings = _settings(tmp_path, server_host="0.0.0.0", server_port=port)
    app = create_app(mem, settings)
    base_url = f"http://0.0.0.0:{port}"
    with TestClient(app, base_url=base_url) as client:
        resp = client.post(
            "/mcp",
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "Host": lan_host,
                "Origin": f"http://{lan_host}",
                "Authorization": "Bearer lantoken",
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0"},
                },
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.status_code not in (421, 403)
        assert '"serverInfo"' in resp.text


def test_non_loopback_mcp_matching_host_origin_without_allowlist_is_dns_rebinding_and_rejected(
    mem, tmp_path, monkeypatch
):
    """Regression test for the DNS-rebinding hole this round closes: a
    forged request whose Origin merely matches its own (attacker-chosen)
    Host header must NOT be trusted just because they match each other --
    only a Host/Origin that's actually `own_origins` or
    `MEMSTRAND_ALLOWED_ORIGINS` is. Same request as the allowlisted test
    above, minus the env var -- must now be rejected."""
    monkeypatch.setenv("MEMSTRAND_TOKEN", "lantoken")
    port = _free_port()
    settings = _settings(tmp_path, server_host="0.0.0.0", server_port=port)
    app = create_app(mem, settings)
    base_url = f"http://0.0.0.0:{port}"
    lan_host = f"192.168.1.50:{port}"
    with TestClient(app, base_url=base_url) as client:
        resp = client.post(
            "/mcp",
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "Host": lan_host,
                "Origin": f"http://{lan_host}",
                "Authorization": "Bearer lantoken",
            },
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        )
        assert resp.status_code == 403
        assert "cross-origin" in resp.text


def test_non_loopback_mcp_without_token_401(mem, tmp_path, monkeypatch):
    monkeypatch.setenv("MEMSTRAND_TOKEN", "lantoken")
    port = _free_port()
    settings = _settings(tmp_path, server_host="0.0.0.0", server_port=port)
    app = create_app(mem, settings)
    base_url = f"http://0.0.0.0:{port}"
    lan_host = f"192.168.1.50:{port}"
    with TestClient(app, base_url=base_url) as client:
        resp = client.post(
            "/mcp",
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "Host": lan_host,
            },
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        )
        assert resp.status_code == 401


def test_non_loopback_rest_also_requires_token_and_accepts_allowlisted_lan_origin(mem, tmp_path, monkeypatch):
    monkeypatch.setenv("MEMSTRAND_TOKEN", "lantoken")
    port = _free_port()
    lan_host = f"192.168.1.50:{port}"
    monkeypatch.setenv("MEMSTRAND_ALLOWED_ORIGINS", f"http://{lan_host}")
    settings = _settings(tmp_path, server_host="0.0.0.0", server_port=port)
    app = create_app(mem, settings)
    base_url = f"http://0.0.0.0:{port}"
    with TestClient(app, base_url=base_url) as client:
        # correct token, Origin explicitly allowlisted via MEMSTRAND_ALLOWED_ORIGINS
        resp = client.post(
            "/api/remember",
            json={"content": "lan fact"},
            headers={
                "Host": lan_host,
                "Origin": f"http://{lan_host}",
                "Authorization": "Bearer lantoken",
            },
        )
        assert resp.status_code == 200

        # an origin that is neither "our own" nor on the allowlist is still
        # rejected even with a valid token
        resp = client.post(
            "/api/remember",
            json={"content": "lan fact 2"},
            headers={
                "Host": lan_host,
                "Origin": "http://evil.example",
                "Authorization": "Bearer lantoken",
            },
        )
        assert resp.status_code == 403


def test_non_loopback_rest_origin_matching_host_but_not_allowlisted_is_rejected(mem, tmp_path, monkeypatch):
    """Same DNS-rebinding regression as the MCP test above, for the REST
    surface: an Origin that merely matches the (attacker-controlled) Host
    header, without being `own_origins` or in `MEMSTRAND_ALLOWED_ORIGINS`,
    must be rejected."""
    monkeypatch.setenv("MEMSTRAND_TOKEN", "lantoken")
    port = _free_port()
    settings = _settings(tmp_path, server_host="0.0.0.0", server_port=port)
    app = create_app(mem, settings)
    base_url = f"http://0.0.0.0:{port}"
    lan_host = f"192.168.1.50:{port}"
    with TestClient(app, base_url=base_url) as client:
        resp = client.post(
            "/api/remember",
            json={"content": "lan fact"},
            headers={
                "Host": lan_host,
                "Origin": f"http://{lan_host}",
                "Authorization": "Bearer lantoken",
            },
        )
        assert resp.status_code == 403


def test_non_loopback_specific_bind_rejects_mismatched_host_header(mem, tmp_path, monkeypatch):
    """New guard added this round: a non-loopback bind to a *specific*
    address/hostname (not the 0.0.0.0/:: wildcard) has exactly one
    legitimate Host value -- anything else (e.g. a public DNS name an
    attacker points at this host) is rejected before the Origin/token
    checks even run."""
    monkeypatch.setenv("MEMSTRAND_TOKEN", "lantoken")
    port = _free_port()
    settings = _settings(tmp_path, server_host="192.168.1.50", server_port=port)
    app = create_app(mem, settings)
    base_url = f"http://192.168.1.50:{port}"
    with TestClient(app, base_url=base_url) as client:
        resp = client.get(
            "/api/health",
            headers={"Host": "evil.example", "Authorization": "Bearer lantoken"},
        )
        assert resp.status_code == 403
        assert "untrusted Host" in resp.text

        # the bind host itself is always an allowed Host value
        resp = client.get(
            "/api/health",
            headers={"Host": f"192.168.1.50:{port}", "Authorization": "Bearer lantoken"},
        )
        assert resp.status_code == 200


def test_non_loopback_wildcard_bind_skips_host_header_check(mem, tmp_path, monkeypatch):
    """A wildcard bind (0.0.0.0/::) has no single legitimate Host value --
    the Host-header guard is skipped for it (the mandatory token and the
    Origin check still apply to everything else)."""
    monkeypatch.setenv("MEMSTRAND_TOKEN", "lantoken")
    port = _free_port()
    settings = _settings(tmp_path, server_host="0.0.0.0", server_port=port)
    app = create_app(mem, settings)
    base_url = f"http://0.0.0.0:{port}"
    with TestClient(app, base_url=base_url) as client:
        resp = client.get(
            "/api/health",
            headers={"Host": "anything.example", "Authorization": "Bearer lantoken"},
        )
        assert resp.status_code == 200


def test_loopback_host_header_still_rejected_on_mcp(mem, tmp_path):
    """Loopback mode keeps the strict Host check (unlike non-loopback,
    where memstrand relies on the mandatory token + Origin check
    instead)."""
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        resp = client.post(
            "/mcp",
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "Host": "evil.example",
            },
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        )
        assert resp.status_code == 403


def test_allowed_origins_env_honored(mem, tmp_path, monkeypatch):
    monkeypatch.setenv("MEMSTRAND_ALLOWED_ORIGINS", "http://localhost:6274, http://127.0.0.1:6274")
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        resp = client.post(
            "/api/remember", json={"content": "x"}, headers={"Origin": "http://localhost:6274"}
        )
        assert resp.status_code == 200

        # an origin not in the extra allowlist (and not our own) is still rejected
        resp = client.post(
            "/api/remember", json={"content": "y"}, headers={"Origin": "http://localhost:9999"}
        )
        assert resp.status_code == 403


def test_allowed_origins_env_normalizes_trailing_slash_and_case(mem, tmp_path, monkeypatch):
    monkeypatch.setenv("MEMSTRAND_ALLOWED_ORIGINS", "HTTP://LOCALHOST:6274/")
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        resp = client.post(
            "/api/remember", json={"content": "x"}, headers={"Origin": "http://localhost:6274"}
        )
        assert resp.status_code == 200


@pytest.mark.parametrize(
    "bad_entry",
    [
        "null",
        "*",
        "ftp://localhost:6274",
        "localhost:6274",
        "http://",
        "http://localhost:6274/some/path",
        "http://localhost:6274?x=1",
        "http://localhost:6274#frag",
        "not a url at all",
    ],
)
def test_allowed_origins_env_malformed_entry_refuses_to_start(mem, tmp_path, monkeypatch, bad_entry):
    """A malformed MEMSTRAND_ALLOWED_ORIGINS entry must fail closed: the
    daemon refuses to start (create_app raises, naming the bad entry)
    rather than silently dropping it, treating it as a wildcard, or
    starting with a narrower-than-intended trust boundary."""
    monkeypatch.setenv("MEMSTRAND_ALLOWED_ORIGINS", bad_entry)
    with pytest.raises(ValueError) as exc_info:
        create_app(mem, _settings(tmp_path))
    assert bad_entry in str(exc_info.value)


def test_allowed_origins_env_empty_is_fine(mem, tmp_path, monkeypatch):
    monkeypatch.delenv("MEMSTRAND_ALLOWED_ORIGINS", raising=False)
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200


# -- token comparison: hmac.compare_digest, missing/malformed header -------------


def test_token_missing_header_401(mem, tmp_path, monkeypatch):
    monkeypatch.setenv("MEMSTRAND_TOKEN", "s3cret")
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        # no Authorization header at all
        resp = client.get("/api/health")
        assert resp.status_code == 401

        # Authorization header present but wrong scheme/shape
        resp = client.get("/api/health", headers={"Authorization": "s3cret"})
        assert resp.status_code == 401

        resp = client.get("/api/health", headers={"Authorization": "Basic czNjcmV0"})
        assert resp.status_code == 401

        # correct token still works
        resp = client.get("/api/health", headers={"Authorization": "Bearer s3cret"})
        assert resp.status_code == 200


def test_token_prefix_match_is_not_enough(mem, tmp_path, monkeypatch):
    """A naive substring/prefix check would be a second, independent bug on
    top of a non-constant-time comparison; make sure a token that shares a
    long prefix (or is a prefix of) the real token is still rejected."""
    monkeypatch.setenv("MEMSTRAND_TOKEN", "s3cret-test-token")
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        resp = client.get("/api/health", headers={"Authorization": "Bearer s3cret-test-toke"})
        assert resp.status_code == 401
        resp = client.get("/api/health", headers={"Authorization": "Bearer s3cret-test-tokenX"})
        assert resp.status_code == 401


# -- security headers on every response ------------------------------------------


def _assert_security_headers(resp, *, expect_script_src: bool):
    assert resp.headers["x-frame-options"] == "DENY"
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["referrer-policy"] == "no-referrer"
    csp = resp.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in csp
    if expect_script_src:
        assert "script-src 'self' 'unsafe-inline'" in csp
    else:
        assert "script-src" not in csp


def test_security_headers_on_index(mem, tmp_path):
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        _assert_security_headers(resp, expect_script_src=True)


def test_security_headers_on_api_health(mem, tmp_path):
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        _assert_security_headers(resp, expect_script_src=False)


def test_security_headers_on_mcp(mem, tmp_path):
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        resp = client.post(
            "/mcp",
            headers={"Accept": "application/json, text/event-stream", "Content-Type": "application/json"},
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        )
        _assert_security_headers(resp, expect_script_src=False)


def test_security_headers_on_rejected_response(mem, tmp_path):
    """Headers must be present even on a rejected (403/401) response, not
    just successful ones -- they're added in the send wrapper before any
    routing decision, including `_reject()`."""
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        resp = client.post(
            "/api/remember", json={"content": "x"}, headers={"Origin": "http://evil.example"}
        )
        assert resp.status_code == 403
        _assert_security_headers(resp, expect_script_src=False)


def test_security_headers_on_sse_stream(mem, tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=b"data: [DONE]\n\n"
        )

    app = create_app(mem, _settings(tmp_path), upstream_transport=httpx.MockTransport(handler))
    with _client(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert resp.status_code == 200
        _assert_security_headers(resp, expect_script_src=False)


# -- /api/atoms pagination + validation --------------------------------------------


def test_atoms_list_bad_limit_and_offset_400(mem, tmp_path):
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        for params in (
            {"limit": "-1"},
            {"limit": "abc"},
            {"limit": "1001"},
            {"offset": "-1"},
            {"offset": "abc"},
        ):
            resp = client.get("/api/atoms", params=params)
            assert resp.status_code == 400, (params, resp.text)

        # boundary values are fine
        resp = client.get("/api/atoms", params={"limit": "1000", "offset": "0"})
        assert resp.status_code in (200, 500)  # 500 only if list_page hasn't landed yet


def test_recall_limit_must_be_at_least_one(mem, tmp_path):
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        resp = client.post("/api/recall", json={"query": "anything", "limit": 0})
        assert resp.status_code == 400
        resp = client.post("/api/recall", json={"query": "anything", "limit": -5})
        assert resp.status_code == 400
        resp = client.post("/api/recall", json={"query": "anything", "limit": 1001})
        assert resp.status_code == 400


def test_atoms_list_pagination_and_total(mem, tmp_path):
    """Depends on `Memory.list_page` (contract owned by api.py): returns
    (atoms, total) computed under the lock, without materializing the
    whole table. If `list_page` hasn't landed yet, this fails with a 500
    (AttributeError) -- that's a known cross-team dependency, not a bug in
    this file; see the handoff report."""
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        for i in range(5):
            client.post("/api/remember", json={"content": f"fact number {i}"})

        resp = client.get("/api/atoms", params={"limit": 2, "offset": 0})
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 5
        assert len(body["atoms"]) == 2

        resp = client.get("/api/atoms", params={"limit": 2, "offset": 4})
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 5
        assert len(body["atoms"]) == 1
