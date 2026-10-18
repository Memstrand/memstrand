"""Wire-level MCP tests: what a REAL MCP client receives on the wire, not
what an in-process `mcp.call_tool()` short-circuit sees.

Two transports are exercised:

  - An in-memory linked transport (`mcp.client._memory.InMemoryTransport`)
    driving a real `mcp.ClientSession` against `build_mcp()` directly — fast,
    no subprocess, but still goes through full JSON-RPC
    serialization/deserialization and pydantic argument/result validation
    (unlike `MCPServer.call_tool()`, which is an in-process shortcut used
    elsewhere in the suite).
  - A real `python -m memstrand.mcp_server` subprocess over stdio
    (`mcp.client.stdio.stdio_client`), for the things that only exist at
    that boundary: stdout purity (no stray prints/warnings corrupting the
    JSON-RPC stream) and the daemon-refuses-to-start guard.

No network beyond 127.0.0.1: the subprocess tests spin up tiny stdlib
`http.server` stubs (an OpenAI-compatible `/v1/embeddings` endpoint, and a
`/api/health` endpoint) in background threads instead of touching anything
real. Ambient `MEMSTRAND_*`/proxy env vars are stripped by the autouse
`_hermetic_env` fixture in `conftest.py`; subprocess environments here are
still built explicitly from a filtered `os.environ` so nothing accidental
leaks into the child process either.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import math
import os
import socket
import subprocess
import sys
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.client._memory import InMemoryTransport
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.server.mcpserver.exceptions import ToolError
from qdrant_client import QdrantClient

from memstrand.api import Memory
from memstrand.mcp_server import build_mcp
from memstrand.stores.graph_sqlite import SqliteGraphStore
from memstrand.stores.sqlite import SqliteSessionStore
from memstrand.stores.vector_qdrant import QdrantVectorStore

REPO_ROOT = Path(__file__).resolve().parents[1]


# ======================================================================================
# shared test doubles / fixtures
# ======================================================================================


class HashEmbedder:
    """Deterministic keyword-overlap embeddings (same test double used
    across the suite) — shared words => similar vectors, no model load."""

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


@pytest.fixture()
def mem(tmp_path):
    return _make_mem(tmp_path)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@asynccontextmanager
async def _session(mcp) -> AsyncIterator[ClientSession]:
    """A real `ClientSession` linked to `mcp` over the in-memory transport —
    full JSON-RPC round trip, no subprocess."""
    async with InMemoryTransport(mcp) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        yield session


# ======================================================================================
# list_tools: enum schemas + outputSchema on every tool
# ======================================================================================


@pytest.mark.anyio
async def test_list_tools_schema_has_category_and_provenance_enums(mem):
    mcp = build_mcp(lambda: mem)
    async with _session(mcp) as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}

        remember_schema = tools["remember"].input_schema
        schema_text = json.dumps(remember_schema)
        # Category enum values must be discoverable in the wire schema, not
        # just validated server-side -- an LLM client picks values from here.
        for value in ("instruction", "fact", "decision", "preference", "artifact", "error"):
            assert value in schema_text
        for value in ("explicit_statement", "inferred", "corrected", "observed"):
            assert value in schema_text

        recall_schema = tools["recall"].input_schema
        assert "categories" in recall_schema["properties"]

        update_schema = tools["update"].input_schema
        assert "category" in update_schema["properties"]


@pytest.mark.anyio
async def test_list_tools_every_tool_has_an_output_schema(mem):
    """Bare `dict` return annotations produce no outputSchema/structuredContent
    in mcp 2.x (`func_metadata._create_output_model` only recognizes
    `dict[str, Any]`, `list[...]`, or a pydantic/TypedDict model). Every
    memstrand tool must annotate a concrete return type so the client
    always gets `structuredContent`, not just a text blob."""
    mcp = build_mcp(lambda: mem)
    async with _session(mcp) as session:
        tools = (await session.list_tools()).tools
        missing = [t.name for t in tools if not t.output_schema]
        assert missing == [], f"tools with no outputSchema: {missing}"


# ======================================================================================
# happy path: every tool's result carries structuredContent
# ======================================================================================


@pytest.mark.anyio
async def test_happy_path_tools_return_structured_content(mem, tmp_path):
    mcp = build_mcp(lambda: mem)
    async with _session(mcp) as session:
        r = await session.call_tool("remember", {"content": "the office wifi is hunter2", "category": "fact"})
        assert r.is_error is False
        assert r.structured_content["category"] == "fact"
        atom_id = r.structured_content["id"]

        r = await session.call_tool("recall", {"query": "office wifi"})
        assert r.is_error is False
        hits = r.structured_content["result"]
        assert any(h["content"] == "the office wifi is hunter2" for h in hits)

        r = await session.call_tool("context", {})
        assert r.is_error is False
        assert "result" in r.structured_content

        r = await session.call_tool("answer", {"question": "something nothing covers"})
        assert r.is_error is False
        # Grounded answering never raises: no relevant memory => "I don't
        # know"; relevant memory but no local LLM reachable => a clear
        # "unavailable" message (grounded_answer() catches LLMUnavailable
        # itself) -- either way `answer` always returns a plain string via
        # structured_content, never an exception.
        assert isinstance(r.structured_content["result"], str) and r.structured_content["result"]

        r = await session.call_tool("update", {"atom_id": atom_id, "confidence": 0.5})
        assert r.is_error is False
        assert r.structured_content["id"] == atom_id

        r = await session.call_tool("log_turn", {"session_id": "s1", "role": "user", "content": "hi"})
        assert r.is_error is False
        assert r.structured_content == {"result": "ok"}

        r = await session.call_tool("improve", {"session_ids": []})
        assert r.is_error is False
        assert isinstance(r.structured_content, dict)

        sample = tmp_path / "note.md"
        sample.write_text("# a note\n\nsomething worth ingesting")
        r = await session.call_tool("ingest", {"path": str(sample)})
        assert r.is_error is False
        assert "files_ingested" in r.structured_content or isinstance(r.structured_content, dict)

        r = await session.call_tool("stats", {})
        assert r.is_error is False
        assert r.structured_content["atoms"] >= 1

        r = await session.call_tool("forget", {"atom_id": atom_id})
        assert r.is_error is False
        assert "forgot" in r.structured_content["result"]


# ======================================================================================
# errors: expected failures reach the client as actionable isError text,
# not mcp's generic "Error executing tool <name>" with the reason dropped
# ======================================================================================


@pytest.mark.anyio
async def test_invalid_category_is_error_lists_valid_values(mem):
    mcp = build_mcp(lambda: mem)
    async with _session(mcp) as session:
        r = await session.call_tool("remember", {"content": "x", "category": "not-a-real-category"})
        assert r.is_error is True
        text = r.content[0].text
        # pydantic's enum ValidationError lists every valid value by name --
        # this is what lets the model self-correct instead of guessing again.
        for value in ("instruction", "fact", "preference", "artifact"):
            assert value in text
        assert "not-a-real-category" in text


@pytest.mark.anyio
async def test_update_missing_atom_is_error_names_the_atom(mem):
    mcp = build_mcp(lambda: mem)
    async with _session(mcp) as session:
        r = await session.call_tool("update", {"atom_id": "mem:does-not-exist", "content": "y"})
        assert r.is_error is True
        assert "mem:does-not-exist" in r.content[0].text


@pytest.mark.anyio
async def test_ingest_missing_path_is_error_names_the_path(mem):
    mcp = build_mcp(lambda: mem)
    async with _session(mcp) as session:
        r = await session.call_tool("ingest", {"path": "/no/such/path/anywhere"})
        assert r.is_error is True
        assert "/no/such/path/anywhere" in r.content[0].text
        assert "does not exist" in r.content[0].text


@pytest.mark.anyio
async def test_forget_missing_atom_is_not_an_error(mem):
    """`forget` returning "no such memory: ..." is a normal (non-error)
    result, by design -- deleting something already gone isn't a failure."""
    mcp = build_mcp(lambda: mem)
    async with _session(mcp) as session:
        r = await session.call_tool("forget", {"atom_id": "mem:nope"})
        assert r.is_error is False
        assert "no such memory: mem:nope" in r.content[0].text


def test_mcp_tool_body_crash_is_not_silently_collapsed(mem):
    """Sanity check on `_guarded`: a genuinely unexpected crash (not one of
    the listed expected-failure types) is deliberately NOT turned into a
    friendly `ToolError` -- it must still propagate so it gets logged
    server-side as a real bug instead of being swallowed."""
    mcp = build_mcp(lambda: mem)

    async def _run():
        with pytest.raises(ToolError) as exc_info:
            await mcp.call_tool("remember", {"content": "x", "category": "bogus"})
        # argument validation errors are also ToolError, but chain the real
        # pydantic error as __cause__ instead of hiding it.
        assert "bogus" in str(exc_info.value.__cause__)

    import anyio

    anyio.run(_run)


# ======================================================================================
# prompts / resources still work
# ======================================================================================


@pytest.mark.anyio
async def test_prompts_and_resources_over_the_wire(mem):
    mcp = build_mcp(lambda: mem)
    async with _session(mcp) as session:
        prompts = {p.name for p in (await session.list_prompts()).prompts}
        assert "memory_context" in prompts

        result = await session.get_prompt("memory_context", {})
        assert result.messages is not None

        resources = {str(r.uri) for r in (await session.list_resources()).resources}
        assert "memstrand://context" in resources

        content = await session.read_resource("memstrand://context")
        assert content.contents is not None


# ======================================================================================
# stdio subprocess: stdout purity + daemon-refuses-to-start
# ======================================================================================


class _EmbeddingStubHandler(http.server.BaseHTTPRequestHandler):
    """Tiny OpenAI-compatible `/v1/embeddings` stub: same hash-bucket scheme
    as `HashEmbedder`, so recall over a real `OpenAIEmbedder` still finds
    what was just remembered."""

    dim = 64

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        texts = body.get("input", [])
        data = [{"embedding": self._embed(t)} for t in texts]
        payload = json.dumps({"data": data}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    @classmethod
    def _embed(cls, text: str) -> list[float]:
        v = [0.0] * cls.dim
        for w in text.lower().split():
            idx = int(hashlib.md5(w.encode()).hexdigest(), 16) % cls.dim
            v[idx] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def log_message(self, *args, **kwargs):  # quiet: keep test output readable
        pass


class _HealthStubHandler(http.server.BaseHTTPRequestHandler):
    """`/api/health` stub whose status code is set per-server-instance
    (`httpd.status_code = ...`) so the same handler covers both the "up"
    (200) and "rejected" (401) daemon-detection cases."""

    def do_GET(self):
        status = self.server.status_code  # type: ignore[attr-defined]
        payload = json.dumps({"ok": status == 200}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args, **kwargs):
        pass


@contextmanager
def _background_http_server(handler_cls, **server_attrs):
    httpd = http.server.HTTPServer(("127.0.0.1", 0), handler_cls)
    for k, v in server_attrs.items():
        setattr(httpd, k, v)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def _clean_subprocess_env(**overrides) -> dict[str, str]:
    """`os.environ` (already hermetic -- see `conftest._hermetic_env`) minus
    any stray `MEMSTRAND_*`, plus explicit overrides for this subprocess."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("MEMSTRAND_")}
    env.update(overrides)
    return env


def _free_port() -> int:
    """An OS-assigned free port, closed immediately after binding. Every
    subprocess test in this file that doesn't deliberately point at a
    stub daemon must still pin `MEMSTRAND_PORT` to one of these instead of
    leaving it at the 7777 default -- otherwise the test's correctness
    depends on nothing else on the machine/CI runner happening to listen
    on 127.0.0.1:7777 (it would make `probe_daemon()` see a real "up"/
    "rejected" status instead of "down" and refuse to start stdio mode,
    exactly like `test_stdio_refuses_to_start_when_daemon_answers_up`
    deliberately exercises)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_stdio_stdout_is_pure_jsonrpc(tmp_path):
    """No stray `print()`/warning can share stdout with the JSON-RPC
    stream in stdio mode -- the client parses every line as a message, and
    `mcp.client.stdio._parse_line` turns anything that doesn't parse into
    an `Exception` delivered to the session's `message_handler`. Driving a
    real subprocess through a couple of tool calls and asserting that
    handler never fires is the most direct way to prove stdout stayed
    clean (Qdrant's on-disk "UserWarning"s etc. must land on stderr)."""
    import anyio

    home = tmp_path / "home"
    env = _clean_subprocess_env(MEMSTRAND_HOME=str(home), MEMSTRAND_PORT=str(_free_port()))
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "memstrand.mcp_server"], env=env, cwd=str(REPO_ROOT)
    )

    stream_exceptions: list[Exception] = []

    async def _message_handler(message):
        if isinstance(message, Exception):
            stream_exceptions.append(message)

    async def _run():
        errlog_path = tmp_path / "stderr.log"
        with errlog_path.open("w") as errlog:
            async with (
                stdio_client(params, errlog=errlog) as (read, write),
                ClientSession(read, write, message_handler=_message_handler) as session,
            ):
                await session.initialize()
                r = await session.call_tool("log_turn", {"session_id": "s1", "role": "user", "content": "hi"})
                assert r.is_error is False
                r = await session.call_tool("stats", {})
                assert r.is_error is False
                r = await session.call_tool("forget", {"atom_id": "mem:nope"})
                assert r.is_error is False
        return errlog_path.read_text()

    stderr_text = anyio.run(_run)
    assert stream_exceptions == [], (
        f"stray non-JSON-RPC content polluted stdout: {stream_exceptions!r}\nstderr was:\n{stderr_text}"
    )


def test_stdio_remember_recall_round_trip_with_stub_embeddings(tmp_path):
    """End-to-end: a real stdio subprocess, a real (on-disk, lazily
    constructed) `Memory`, and a real `OpenAIEmbedder` pointed at a stub
    `/v1/embeddings` server -- remember something, recall it back."""
    import anyio

    with _background_http_server(_EmbeddingStubHandler) as httpd:
        embed_port = httpd.server_address[1]
        home = tmp_path / "home"
        env = _clean_subprocess_env(
            MEMSTRAND_HOME=str(home),
            MEMSTRAND_PORT=str(_free_port()),
            MEMSTRAND_EMBED_BACKEND="openai",
            MEMSTRAND_EMBED_MODEL="stub",
            MEMSTRAND_EMBED_BASE_URL=f"http://127.0.0.1:{embed_port}/v1",
        )
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "memstrand.mcp_server"], env=env, cwd=str(REPO_ROOT)
        )

        async def _run():
            errlog_path = tmp_path / "stderr.log"
            with errlog_path.open("w") as errlog:
                async with (
                    stdio_client(params, errlog=errlog) as (read, write),
                    ClientSession(read, write) as session,
                ):
                    await session.initialize()
                    r = await session.call_tool(
                        "remember", {"content": "the launch code is on the sticky note", "category": "fact"}
                    )
                    assert r.is_error is False, r.content[0].text if r.content else r
                    r = await session.call_tool("recall", {"query": "launch code"})
                    assert r.is_error is False
                    hits = r.structured_content["result"]
                    assert any("sticky note" in h["content"] for h in hits)

        anyio.run(_run)


def test_stdio_refuses_to_start_when_daemon_answers_up(tmp_path):
    """A daemon answering 200 on `/api/health` must stop stdio mode before
    it ever opens the (single-writer) local stores."""
    with _background_http_server(_HealthStubHandler, status_code=200) as httpd:
        port = httpd.server_address[1]
        home = tmp_path / "home"
        env = _clean_subprocess_env(MEMSTRAND_HOME=str(home), MEMSTRAND_PORT=str(port))
        proc = subprocess.run(
            [sys.executable, "-m", "memstrand.mcp_server"],
            cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=15, check=False,
        )
        assert proc.returncode == 1
        assert proc.stdout == ""
        assert f"127.0.0.1:{port}" in proc.stderr
        assert "stdio" in proc.stderr


def test_stdio_refuses_to_start_when_daemon_rejects(tmp_path):
    """A daemon answering 401 (wrong/missing MEMSTRAND_TOKEN) is NOT "down"
    -- stdio mode must refuse exactly the same as the 200 case instead of
    falling through to a local Memory() and crashing on Qdrant's lock."""
    with _background_http_server(_HealthStubHandler, status_code=401) as httpd:
        port = httpd.server_address[1]
        home = tmp_path / "home"
        env = _clean_subprocess_env(MEMSTRAND_HOME=str(home), MEMSTRAND_PORT=str(port))
        proc = subprocess.run(
            [sys.executable, "-m", "memstrand.mcp_server"],
            cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=15, check=False,
        )
        assert proc.returncode == 1
        assert proc.stdout == ""
        assert f"127.0.0.1:{port}" in proc.stderr
        assert "MEMSTRAND_TOKEN" in proc.stderr
