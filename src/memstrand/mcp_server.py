"""MCP server: memstrand tools for any MCP-compatible agent.

    python -m memstrand.mcp_server

`build_mcp()` is the single source of truth for the tool surface. The
daemon (server.py) mounts the exact same MCPServer instance at /mcp over
streamable-HTTP so Claude Code, Cursor, etc. get identical tool
behavior whether they speak stdio or HTTP — only `get_mem()` differs
(a private lazily-constructed Memory for stdio, the daemon's shared
Memory for the mount).
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import httpx
from mcp.server.mcpserver import MCPServer

from .api import Memory
from .config import get_settings
from .schema import Category, Provenance

_CATEGORY_VALUES = [c.value for c in Category]
_PROVENANCE_VALUES = [p.value for p in Provenance]


def _category(value: str) -> Category:
    try:
        return Category(value)
    except ValueError:
        raise ValueError(f"invalid category {value!r}; must be one of {_CATEGORY_VALUES}") from None


def _provenance(value: str) -> Provenance:
    try:
        return Provenance(value)
    except ValueError:
        raise ValueError(f"invalid provenance {value!r}; must be one of {_PROVENANCE_VALUES}") from None


def build_mcp(get_mem: Callable[[], Memory]) -> MCPServer:
    """Register every memstrand tool against whatever `Memory` `get_mem()`
    returns, looked up fresh on every call so the daemon can share one
    long-lived Memory across requests while stdio mode builds its own.

    Transport configuration (e.g. `streamable_http_path="/"`, so
    server.py can `Mount("/mcp", ...)` the returned app's ASGI handler
    without doubling the path segment) is passed to
    `streamable_http_app()` / `run()` by the caller in the v2 SDK — it is
    not a constructor argument anymore, and is irrelevant to stdio
    transport.
    """
    mcp = MCPServer("memstrand")

    @mcp.tool()
    def remember(
        content: str,
        category: str = "fact",
        provenance: str = "explicit_statement",
        confidence: float = 0.8,
        task_id: str | None = None,
        host: str | None = None,
        ttl_days: float | None = None,
        session_id: str | None = None,
        source: str | None = None,
    ) -> dict:
        """Store one atomic, durable memory. Call this as soon as the user
        states a fact, preference, decision, instruction, or commitment —
        not for chit-chat or throwaway context. One atomic fact per call;
        split compound statements into separate calls."""
        atom = get_mem().remember(
            content,
            category=_category(category),
            provenance=_provenance(provenance),
            confidence=confidence,
            session_id=session_id,
            task_id=task_id,
            host=host,
            ttl_days=ttl_days,
            source=source,
        )
        return {"id": atom.id, "category": atom.category.value, "task_id": atom.task_id}

    @mcp.tool()
    def recall(
        query: str,
        task_id: str | None = None,
        host: str | None = None,
        categories: list[str] | None = None,
        limit: int = 10,
        include_superseded: bool = False,
    ) -> list[dict]:
        """Scoped hybrid (semantic + keyword) recall. Call this at the
        start of a task and whenever you need context the user may have
        given you before — their preferences, past decisions, standing
        instructions, or anything relevant to the current question.
        Returns a trimmed list of the most relevant memories, newest
        contradictions included so you never act on stale info."""
        cats = [_category(c) for c in categories] if categories else None
        hits = get_mem().recall(
            query,
            task_id=task_id,
            host=host,
            categories=cats,
            limit=limit,
            include_superseded=include_superseded,
        )
        return [
            {
                "id": h["id"],
                "score": h["score"],
                "category": h["category"],
                "content": h["content"],
                "provenance": h["provenance"],
                "confidence": h["confidence"],
                "task_id": h["task_id"],
                "source_ref": h["source_ref"],
                "contradictions": [c["content"] for c in h["contradictions"]],
            }
            for h in hits
        ]

    @mcp.tool()
    def context(query: str | None = None, task_id: str | None = None) -> str:
        """Markdown memory block (standing instructions/preferences plus
        query-relevant memories), ready to drop straight into a system
        prompt. Empty string if there is nothing worth saying."""
        return get_mem().context(query, task_id=task_id)

    @mcp.tool()
    def answer(question: str, task_id: str | None = None) -> str:
        """Answer a question using ONLY stored memories — grounded, never
        hallucinated. Returns an "I don't know" message if nothing in
        memory covers the question."""
        return get_mem().grounded_answer(question, task_id=task_id)

    @mcp.tool()
    def update(
        atom_id: str,
        content: str | None = None,
        category: str | None = None,
        confidence: float | None = None,
    ) -> dict:
        """Correct an existing memory. If `content` actually changes, a
        new atom is created superseding the old one (the old atom stays
        queryable, never silently erased); if only category/confidence
        change, the atom is updated in place."""
        atom = get_mem().update(
            atom_id,
            content=content,
            category=_category(category) if category is not None else None,
            confidence=confidence,
        )
        return {
            "id": atom.id,
            "category": atom.category.value,
            "content": atom.content,
            "task_id": atom.task_id,
        }

    @mcp.tool()
    def forget(atom_id: str) -> str:
        """Delete a memory, with provenance logging. Use when the user
        explicitly asks you to forget or delete something they told you."""
        found = get_mem().forget(atom_id)
        return f"forgot {atom_id}" if found else f"no such memory: {atom_id}"

    @mcp.tool()
    def log_turn(session_id: str, role: str, content: str, task_id: str | None = None) -> str:
        """Log one raw conversation turn (role: "user" or "assistant") for
        later distillation into typed memory via `improve`. Call this for
        every turn worth remembering, even before you know what parts of
        it are durable — distillation figures that out later."""
        get_mem().log_turn(session_id, role, content, task_id=task_id)
        return "ok"

    @mcp.tool()
    def improve(session_ids: list[str] | None = None) -> dict:
        """Distill logged session turns into typed permanent memory, in
        the background (returns immediately). Pass `session_ids=None` to
        distill every session that has new, undistilled turns."""
        return get_mem().improve(session_ids, background=True)

    @mcp.tool()
    def ingest(path: str, task_id: str | None = None) -> dict:
        """Ingest a file or directory (Markdown/Obsidian vault, plain
        text, or PDF) into memory. Safe to call repeatedly — only changed
        files are re-ingested."""
        return get_mem().ingest(path, task_id=task_id)

    @mcp.tool()
    def stats() -> dict:
        """Memory store stats: atom/edge/session counts and a breakdown
        by category and task — useful for sanity-checking what's stored."""
        return get_mem().stats()

    @mcp.prompt()
    def memory_context(query: str = "", task_id: str = "") -> str:
        """Fetch the memory context block for a query/task scope, for use
        as a system or user prompt addition."""
        return get_mem().context(query or None, task_id=task_id or None)

    @mcp.resource("memstrand://context")
    def standing_context() -> str:
        """Global standing instructions and preferences (no query) — the
        durable rules that apply no matter what task is active."""
        return get_mem().context(None)

    return mcp


_mem: Memory | None = None


def _lazy_memory() -> Memory:
    global _mem
    if _mem is None:
        _mem = Memory()
    return _mem


def _daemon_running(server_url: str, timeout: float = 0.5) -> bool:
    try:
        resp = httpx.get(f"{server_url}/api/health", timeout=timeout)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


def main() -> None:
    """stdio entry point. Opens the stores directly, so it must not run
    alongside the daemon (embedded Qdrant/SQLite are single-process) —
    if a daemon already answers on settings.server_url, tell the user to
    point their MCP client at its HTTP endpoint instead and exit."""
    settings = get_settings()
    if _daemon_running(settings.server_url):
        print(
            f"memstrand daemon is already running at {settings.server_url} "
            "and owns the stores. Configure your MCP client with the HTTP "
            f"endpoint {settings.server_url}/mcp instead of stdio.",
            file=sys.stderr,
        )
        sys.exit(1)
    build_mcp(_lazy_memory).run()


if __name__ == "__main__":
    main()
