"""Daemon tests: REST API, security guards, the OpenAI-compatible memory
proxy, the /mcp mount, and the background scheduler.

Same no-network discipline as tests/test_core_v2.py: a deterministic
HashEmbedder (no fastembed), in-memory Qdrant, tmp SQLite stores, and
every LLM/upstream call monkeypatched or routed through an
`httpx.MockTransport` — nothing here touches the network.
"""

from __future__ import annotations

import contextlib
import hashlib
import json as json_lib
import math
import time
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from qdrant_client import QdrantClient
from starlette.testclient import TestClient

from memstrand.api import Memory
from memstrand.config import get_settings
from memstrand.mcp_server import build_mcp
from memstrand.server import create_app, serve
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
    s = replace(get_settings(), home=tmp_path, server_host="127.0.0.1", server_port=7777)
    return replace(s, **overrides) if overrides else s


@contextlib.contextmanager
def _client(app):
    base_url = f"http://{app.state.settings.server_host}:{app.state.settings.server_port}"
    with TestClient(app, base_url=base_url) as client:
        yield client


@pytest.fixture()
def mem(tmp_path):
    return _make_mem(tmp_path)


# -- basic REST surface -----------------------------------------------------------


def test_health(mem, tmp_path):
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["atoms"] == 0
        assert "version" in body


def test_remember_then_recall(mem, tmp_path):
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        resp = client.post("/api/remember", json={"content": "the sky is blue over the office"})
        assert resp.status_code == 200
        atom = resp.json()
        assert atom["content"] == "the sky is blue over the office"
        assert atom["category"] == "fact"

        resp = client.post("/api/recall", json={"query": "sky blue office"})
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert any(r["id"] == atom["id"] for r in results)


def test_remember_accepts_agent_field(mem, tmp_path):
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        resp = client.post("/api/remember", json={"content": "logged by a subagent", "agent": "reviewer-bot"})
        assert resp.status_code == 200
        atom_id = resp.json()["id"]

    stored = mem.graph.get_atom(atom_id)
    assert stored.agent == "reviewer-bot"


def test_remember_invalid_category_400(mem, tmp_path):
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        resp = client.post("/api/remember", json={"content": "x", "category": "bogus"})
        assert resp.status_code == 400
        assert "bogus" in resp.json()["error"]


def test_remember_requires_content(mem, tmp_path):
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        resp = client.post("/api/remember", json={})
        assert resp.status_code == 400


def test_atoms_list_absent_vs_empty_task_id(mem, tmp_path):
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        client.post("/api/remember", json={"content": "a global fact"})
        client.post("/api/remember", json={"content": "a scoped fact", "task_id": "proj-x"})

        # absent task_id => no filter, sees both
        resp = client.get("/api/atoms")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2

        # present-but-empty task_id => global only
        resp = client.get("/api/atoms", params={"task_id": ""})
        body = resp.json()
        assert body["total"] == 1
        assert body["atoms"][0]["content"] == "a global fact"

        # specific task_id
        resp = client.get("/api/atoms", params={"task_id": "proj-x"})
        body = resp.json()
        assert body["total"] == 1
        assert body["atoms"][0]["content"] == "a scoped fact"


def test_atom_get_patch_delete_and_404s(mem, tmp_path):
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        resp = client.post("/api/remember", json={"content": "the wifi password is hunter1"})
        atom_id = resp.json()["id"]

        resp = client.get(f"/api/atoms/{atom_id}")
        assert resp.status_code == 200
        detail = resp.json()
        assert detail["content"] == "the wifi password is hunter1"
        assert "neighbors" in detail and "history" in detail

        resp = client.patch(f"/api/atoms/{atom_id}", json={"confidence": 0.42})
        assert resp.status_code == 200
        assert resp.json()["confidence"] == 0.42

        resp = client.delete(f"/api/atoms/{atom_id}")
        assert resp.status_code == 200
        assert resp.json() == {"forgot": atom_id}

        assert client.get(f"/api/atoms/{atom_id}").status_code == 404
        assert client.delete(f"/api/atoms/{atom_id}").status_code == 404
        assert client.patch(f"/api/atoms/{atom_id}", json={"confidence": 0.1}).status_code == 404


# -- turns + improve ----------------------------------------------------------------


class _FakeResp:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _fake_extraction_llm(url, headers=None, json=None, timeout=None):
    body = json
    system = body["messages"][0]["content"]
    user_msg = body["messages"][1]["content"]
    if "extract durable memories" in system:
        content = "fallback"
        for line in user_msg.splitlines():
            if line.startswith("USER:"):
                content = line[len("USER:") :].strip()
        payload = [
            {"category": "preference", "content": content, "provenance": "explicit_statement", "confidence": 0.9}
        ]
        return _FakeResp(json_lib.dumps(payload))
    if "resolve one candidate memory" in system:
        return _FakeResp(
            json_lib.dumps({"action": "add", "target_id": None, "relation": "related_to", "reason": "new"})
        )
    raise AssertionError(f"unexpected prompt: {system[:80]}")


def test_turns_single_and_batch_then_improve(mem, tmp_path, monkeypatch):
    monkeypatch.setattr(httpx, "post", _fake_extraction_llm)
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        resp = client.post(
            "/api/turns", json={"session_id": "s1", "role": "user", "content": "I prefer dark mode"}
        )
        assert resp.status_code == 200 and resp.json() == {"ok": True}

        resp = client.post(
            "/api/turns",
            json={
                "turns": [
                    {"session_id": "s1", "role": "assistant", "content": "noted"},
                    {"session_id": "s2", "role": "user", "content": "I also like vim keybindings"},
                ]
            },
        )
        assert resp.status_code == 200

        resp = client.post("/api/improve", json={"session_ids": ["s1", "s2"]})
        assert resp.status_code == 200
        report = resp.json()
        assert report["atoms"] >= 1


def test_turns_missing_fields_400(mem, tmp_path):
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        resp = client.post("/api/turns", json={"session_id": "s1", "role": "user"})
        assert resp.status_code == 400


# -- context / export / import / ingest ----------------------------------------------


def test_context(mem, tmp_path):
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        client.post("/api/remember", json={"content": "always respond concisely", "category": "instruction"})
        resp = client.post("/api/context", json={"query": "anything"})
        assert resp.status_code == 200
        assert "always respond concisely" in resp.json()["context"]


def test_export_import_roundtrip(mem, tmp_path):
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        client.post("/api/remember", json={"content": "an exportable fact", "task_id": "t1"})

        resp = client.get("/api/export", params={"format": "jsonl"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/x-ndjson")
        blob = resp.text

        mem2 = _make_mem(tmp_path / "mem2")
        app2 = create_app(mem2, _settings(tmp_path, server_port=7778))
        with _client(app2) as client2:
            resp = client2.post("/api/import", content=blob.encode())
            assert resp.status_code == 200
            report = resp.json()
            assert report["atoms"] == 1
            assert report["errors"] == []

        resp = client.get("/api/export", params={"format": "markdown"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        assert "an exportable fact" in resp.text

        resp = client.get("/api/export", params={"format": "bogus"})
        assert resp.status_code == 400


def test_ingest_tmp_markdown_dir(mem, tmp_path):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "one.md").write_text("# Title\n\nSome durable content about the project roadmap.")

    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        resp = client.post("/api/ingest", json={"path": str(notes_dir)})
        assert resp.status_code == 200
        report = resp.json()
        assert report["files_ingested"] == 1
        assert report["chunks"] >= 1

        resp = client.post("/api/ingest", json={"path": str(tmp_path / "does-not-exist")})
        assert resp.status_code == 400


# -- web UI -------------------------------------------------------------------------


def test_index_html_served(mem, tmp_path):
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert "memstrand" in resp.text


# -- security: Origin / Host / token -------------------------------------------------


def test_origin_check_rejects_foreign_rejects_only_mutating(mem, tmp_path):
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        resp = client.post(
            "/api/remember", json={"content": "x"}, headers={"Origin": "http://evil.example"}
        )
        assert resp.status_code == 403

        resp = client.post(
            "/api/remember", json={"content": "y"}, headers={"Origin": "http://127.0.0.1:7777"}
        )
        assert resp.status_code == 200

        resp = client.post("/api/remember", json={"content": "z"})
        assert resp.status_code == 200

        # GET is not mutating: a foreign Origin must not block it
        resp = client.get("/api/health", headers={"Origin": "http://evil.example"})
        assert resp.status_code == 200


def test_host_header_rejected_when_not_loopback(mem, tmp_path):
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        resp = client.get("/api/health", headers={"Host": "evil.example"})
        assert resp.status_code == 403


def test_token_enforcement(mem, tmp_path, monkeypatch):
    monkeypatch.setenv("MEMSTRAND_TOKEN", "s3cret")
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        resp = client.get("/api/health")
        assert resp.status_code == 401

        resp = client.get("/api/health", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401

        resp = client.get("/api/health", headers={"Authorization": "Bearer s3cret"})
        assert resp.status_code == 200

        # unprotected route stays open even with a token configured
        resp = client.get("/")
        assert resp.status_code == 200


def test_refuses_non_loopback_without_token(mem, tmp_path, monkeypatch):
    monkeypatch.delenv("MEMSTRAND_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        create_app(mem, _settings(tmp_path, server_host="0.0.0.0"))


# -- OpenAI-compatible proxy ----------------------------------------------------------


def _mock_upstream(stream_mode: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("/chat/completions"):
            return httpx.Response(404)
        body = json_lib.loads(request.content)
        if body.get("stream"):
            chunks = [
                b'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n',
                b'data: {"choices":[{"delta":{"content":"world"}}]}\n\n',
                b"data: [DONE]\n\n",
            ]
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=b"".join(chunks)
            )
        stream_mode["last_body"] = body
        sys_content = next((m["content"] for m in body["messages"] if m["role"] == "system"), "")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": f"reply saw: {sys_content[:10]}"}}]},
        )

    return httpx.MockTransport(handler)


def test_proxy_injects_memory_and_logs_turns(mem, tmp_path):
    mem.remember("always answer in haiku", category="instruction")
    captured: dict = {}
    app = create_app(mem, _settings(tmp_path), upstream_transport=_mock_upstream(captured))
    with _client(app) as client:
        resp = client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "tell me about haiku"}]}
        )
        assert resp.status_code == 200
        assert "haiku" in resp.json()["choices"][0]["message"]["content"].lower() or captured["last_body"]
        sys_msgs = [m for m in captured["last_body"]["messages"] if m["role"] == "system"]
        assert sys_msgs and "always answer in haiku" in sys_msgs[0]["content"]

    sessions = mem.sessions.list_sessions()
    assert len(sessions) == 1
    turns = mem.sessions.get_session(sessions[0])
    roles = [t["role"] for t in turns]
    assert roles == ["user", "assistant"]


def test_proxy_streaming_accumulates_and_logs(mem, tmp_path):
    captured: dict = {}
    app = create_app(mem, _settings(tmp_path), upstream_transport=_mock_upstream(captured))
    with _client(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "stream please"}], "stream": True},
        )
        assert resp.status_code == 200
        assert "Hello " in resp.text and "world" in resp.text

    sessions = mem.sessions.list_sessions()
    turns = mem.sessions.get_session(sessions[0])
    assistant_turns = [t for t in turns if t["role"] == "assistant"]
    assert assistant_turns and assistant_turns[-1]["content"] == "Hello world"


def test_proxy_upstream_down_returns_502(mem, tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    app = create_app(mem, _settings(tmp_path), upstream_transport=httpx.MockTransport(handler))
    with _client(app) as client:
        resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 502
        assert "error" in resp.json()


def test_proxy_disable_header_skips_injection_and_logging(mem, tmp_path):
    mem.remember("always answer in haiku", category="instruction")
    captured: dict = {}
    app = create_app(mem, _settings(tmp_path), upstream_transport=_mock_upstream(captured))
    with _client(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "tell me about haiku"}]},
            headers={"X-Memstrand-Disable": "1"},
        )
        assert resp.status_code == 200
        sys_msgs = [m for m in captured["last_body"]["messages"] if m["role"] == "system"]
        assert not sys_msgs

    assert mem.sessions.list_sessions() == []


def test_v1_models_fallback_when_upstream_unreachable(mem, tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    app = create_app(mem, _settings(tmp_path), upstream_transport=httpx.MockTransport(handler))
    with _client(app) as client:
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        assert resp.json()["data"][0]["id"] == mem.llm_model


# -- MCP ------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_build_mcp_tools_callable_in_process(mem):
    mcp = build_mcp(lambda: mem)
    result = await mcp.call_tool("remember", {"content": "a durable fact", "category": "fact"})
    # v2: call_tool returns a CallToolResult; dict returns serialize to a
    # single JSON text block, list returns land in structured_content.
    data = json_lib.loads(result.content[0].text)
    assert data["category"] == "fact"

    result = await mcp.call_tool("recall", {"query": "durable fact"})
    hits = result.structured_content["result"]
    assert any(h["content"] == "a durable fact" for h in hits)

    tools = {t.name for t in await mcp.list_tools()}
    assert {"remember", "recall", "context", "answer", "update", "forget", "log_turn", "improve", "ingest", "stats"} <= tools

    prompts = {p.name for p in await mcp.list_prompts()}
    assert "memory_context" in prompts

    resources = {str(r.uri) for r in await mcp.list_resources()}
    assert "memstrand://context" in resources


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_mcp_tool_rejects_invalid_category(mem):
    from mcp.server.mcpserver.exceptions import ToolError

    mcp = build_mcp(lambda: mem)

    async def _run():
        # v2 reports a generic ToolError message; the original validation
        # error (naming the bad value) is chained as __cause__.
        with pytest.raises(ToolError) as exc_info:
            await mcp.call_tool("remember", {"content": "x", "category": "bogus"})
        assert "bogus" in str(exc_info.value.__cause__)

    import anyio

    anyio.run(_run)


def test_mcp_mount_responds_to_initialize(mem, tmp_path):
    app = create_app(mem, _settings(tmp_path))
    with _client(app) as client:
        resp = client.post(
            "/mcp",
            headers={"Accept": "application/json, text/event-stream", "Content-Type": "application/json"},
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
        assert resp.status_code == 200
        assert '"serverInfo"' in resp.text


def test_mcp_bare_path_no_redirect(mem, tmp_path):
    """Regression: `Mount("/mcp", ...)` only matches "/mcp/...", so a bare
    POST to "/mcp" (no trailing slash — what `claude mcp add --transport
    http` is told to use, and what non-browser MCP clients send without
    following redirects) used to 307 to "/mcp/" via Starlette's
    redirect-slash fallback instead of responding directly."""
    app = create_app(mem, _settings(tmp_path))
    base_url = f"http://{app.state.settings.server_host}:{app.state.settings.server_port}"
    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    with TestClient(app, base_url=base_url, follow_redirects=False) as client:
        init_resp = client.post(
            "/mcp",
            headers=headers,
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
        assert init_resp.status_code == 200, init_resp.headers.get("location")
        session_id = init_resp.headers["mcp-session-id"]

        session_headers = {**headers, "mcp-session-id": session_id}
        client.post("/mcp", headers=session_headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"})

        tools_resp = client.post(
            "/mcp", headers=session_headers, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
        )
        assert tools_resp.status_code == 200
        assert "remember" in tools_resp.text

        # the trailing-slash form (what Mount matches directly) still works
        trailing_resp = client.post(
            "/mcp/",
            headers=headers,
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
        assert trailing_resp.status_code == 200


# -- scheduler --------------------------------------------------------------------------


def test_serve_signature_has_scheduler_kwarg():
    import inspect

    params = inspect.signature(serve).parameters
    assert list(params) == ["host", "port", "mem", "scheduler"]
    assert params["scheduler"].default is True


def test_scheduler_starts_and_stops(mem, tmp_path, monkeypatch):
    calls: list[float] = []
    monkeypatch.setattr(mem, "consolidate", lambda: calls.append(time.monotonic()) or {})

    settings = _settings(tmp_path, consolidate_hours=0.0)
    app = create_app(
        mem,
        settings,
        start_scheduler=True,
        scheduler_first_delay=0.01,
        scheduler_poll_interval=0.02,
        scheduler_improve_interval=999.0,
    )
    with _client(app):
        time.sleep(0.15)
        assert len(calls) >= 1

    count_after_shutdown = len(calls)
    time.sleep(0.15)
    assert len(calls) == count_after_shutdown  # thread really stopped
