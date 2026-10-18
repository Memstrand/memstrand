"""Scenario #5: MCP over real streamable-HTTP against a real daemon, using
the mcp 2.x client SDK end to end -- not the in-memory transport unit tests
elsewhere exercise.

Covers both the bare `/mcp` (no trailing slash, dispatched by server.py's
`_McpBareDispatch` since `Mount("/mcp", ...)` alone only matches a
trailing-slash path) and `/mcp/` mount points, the same error/structured-
content behavior already unit-tested in-process
(`tests/test_mcp_wire.py`), and token enforcement: a client with the right
`Authorization: Bearer <token>` header works, one with none gets a plain
HTTP 401 before MCP ever sees the request (`_SecurityMiddleware` rejects
it at the ASGI layer, independent of the MCP framing of the body).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import anyio
import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

pytestmark = pytest.mark.integration


@asynccontextmanager
async def _session(url: str, token: str | None = None):
    http_client = create_mcp_http_client(headers={"Authorization": f"Bearer {token}"}) if token else None
    async with (
        streamable_http_client(url, http_client=http_client) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


@pytest.mark.parametrize("suffix", ["/mcp", "/mcp/"])
def test_initialize_and_list_tools_over_http(start_daemon, stub, suffix):
    daemon = start_daemon(stub=stub)

    async def _run():
        async with _session(f"{daemon.url}{suffix}") as session:
            tools = {t.name for t in (await session.list_tools()).tools}
            assert "remember" in tools
            assert "recall" in tools

            remember_tool = next(t for t in (await session.list_tools()).tools if t.name == "remember")
            import json

            schema_text = json.dumps(remember_tool.input_schema)
            for value in ("instruction", "fact", "preference", "artifact"):
                assert value in schema_text

    anyio.run(_run)


def test_remember_recall_round_trip_structured_content(start_daemon, stub):
    daemon = start_daemon(stub=stub)

    async def _run():
        async with _session(f"{daemon.url}/mcp/") as session:
            r = await session.call_tool(
                "remember", {"content": "the office wifi is hunter2", "category": "fact"}
            )
            assert r.is_error is False
            assert r.structured_content["category"] == "fact"

            r = await session.call_tool("recall", {"query": "office wifi"})
            assert r.is_error is False
            hits = r.structured_content["result"]
            assert any(h["content"] == "the office wifi is hunter2" for h in hits)

    anyio.run(_run)


def test_invalid_category_is_error_lists_valid_values(start_daemon, stub):
    daemon = start_daemon(stub=stub)

    async def _run():
        async with _session(f"{daemon.url}/mcp/") as session:
            r = await session.call_tool("remember", {"content": "x", "category": "not-a-real-category"})
            assert r.is_error is True
            text = r.content[0].text
            for value in ("instruction", "fact", "preference", "artifact"):
                assert value in text
            assert "not-a-real-category" in text

    anyio.run(_run)


def test_ingest_missing_path_is_error_names_the_path(start_daemon, stub):
    daemon = start_daemon(stub=stub)

    async def _run():
        async with _session(f"{daemon.url}/mcp/") as session:
            r = await session.call_tool("ingest", {"path": "/no/such/path/anywhere"})
            assert r.is_error is True
            assert "/no/such/path/anywhere" in r.content[0].text
            assert "does not exist" in r.content[0].text

    anyio.run(_run)


def test_token_protected_mcp_requires_auth_header(start_daemon, stub):
    daemon = start_daemon(stub=stub, token="s3cret-token")

    # With the right token, over the real MCP client -- works end to end.
    async def _run():
        async with _session(f"{daemon.url}/mcp/", token="s3cret-token") as session:
            r = await session.call_tool("stats", {})
            assert r.is_error is False

    anyio.run(_run)

    # With no Authorization header at all: the security middleware rejects
    # the request before MCP framing is even parsed -- a plain HTTP 401,
    # not an MCP-level error.
    resp = httpx.post(
        f"{daemon.url}/mcp/",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        headers={"Accept": "application/json, text/event-stream", "Content-Type": "application/json"},
        timeout=5.0,
    )
    assert resp.status_code == 401
