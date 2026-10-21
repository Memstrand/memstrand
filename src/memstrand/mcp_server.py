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

import functools
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .client import RemoteError, open_local_memory, probe_daemon
from .config import get_settings
from .pipeline.extract import LLMUnavailable
from .schema import Category, Provenance
from .scopes import agent_name

if TYPE_CHECKING:
    # `.api` pulls in qdrant_client -> fastembed -> onnxruntime; only
    # imported for type annotations here. `_lazy_memory()`/`open_local_memory`
    # (client.py) are what actually construct one, on first real use --
    # merely importing this module (e.g. to check whether a daemon is
    # already up in `main()`) must stay cheap.
    from .api import Memory


def _guarded(fn):
    """Turn a tool body's expected failures into `ToolError` (message kept
    intact for the model) instead of letting mcp's `Tool.run()` collapse
    them into the generic "Error executing tool <name>" with the reason
    dropped — see `mcp.server.mcpserver.tools.base.Tool.run`: anything
    that escapes a tool body which isn't already `ToolError`/`ResourceError`/
    `MCPError` becomes an `UnexpectedToolError` whose own text never
    reaches the client. Argument-validation failures (bad enum value,
    wrong type, missing required field) never reach here at all — the SDK
    catches those earlier, directly off the typed `Category`/`Provenance`
    parameters, and reports them the same way.

    Anything not listed below is a genuine crash and is left to propagate
    (and be logged server-side) unwrapped."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except KeyError as exc:
            # str(KeyError("no such atom: x")) is "'no such atom: x'" (repr
            # of its one arg) -- unwrap it back to the plain message.
            (message,) = exc.args or (str(exc),)
            raise ToolError(str(message)) from exc
        except (ValueError, FileNotFoundError, RemoteError) as exc:
            raise ToolError(str(exc)) from exc
        except LLMUnavailable as exc:
            kind = "transient, retrying later may work" if exc.transient else "not transient"
            raise ToolError(f"local LLM unavailable ({kind}): {exc}") from exc

    return wrapper


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
    @_guarded
    def remember(
        content: str,
        category: Category = Category.fact,
        provenance: Provenance = Provenance.explicit_statement,
        confidence: float = 0.8,
        task_id: str | None = None,
        host: str | None = None,
        ttl_days: float | None = None,
        session_id: str | None = None,
        source: str | None = None,
        scopes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Store one atomic, durable memory. Call this as soon as the user
        states a fact, preference, decision, instruction, or commitment —
        not for chit-chat or throwaway context. One atomic fact per call;
        split compound statements into separate calls. `scopes` lists the
        repos/tasks it applies to (e.g. ["github.com/org/repo"]); omit both
        it and `task_id` for a global memory."""
        atom = get_mem().remember(
            content,
            category=category,
            provenance=provenance,
            confidence=confidence,
            session_id=session_id,
            task_id=task_id,
            host=host,
            ttl_days=ttl_days,
            source=source,
            scopes=scopes,
            agent=agent_name("mcp"),
        )
        return {
            "id": atom.id, "category": atom.category.value,
            "task_id": atom.task_id, "scopes": list(atom.scopes),
        }

    @mcp.tool()
    @_guarded
    def recall(
        query: str,
        task_id: str | None = None,
        host: str | None = None,
        categories: list[Category] | None = None,
        limit: int = 10,
        include_superseded: bool = False,
        scopes: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Scoped hybrid (semantic + keyword) recall. Call this at the
        start of a task and whenever you need context the user may have
        given you before — their preferences, past decisions, standing
        instructions, or anything relevant to the current question.
        Returns a trimmed list of the most relevant memories, newest
        contradictions included so you never act on stale info."""
        hits = get_mem().recall(
            query,
            task_id=task_id,
            host=host,
            categories=categories,
            limit=limit,
            include_superseded=include_superseded,
            scopes=scopes,
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
                "scopes": h["scopes"],
                "source_ref": h["source_ref"],
                "contradictions": [c["content"] for c in h["contradictions"]],
            }
            for h in hits
        ]

    @mcp.tool()
    @_guarded
    def context(
        query: str | None = None, task_id: str | None = None, scopes: list[str] | None = None
    ) -> str:
        """Markdown memory block (standing instructions/preferences plus
        query-relevant memories), ready to drop straight into a system
        prompt. Empty string if there is nothing worth saying."""
        return get_mem().context(query, task_id=task_id, scopes=scopes)

    @mcp.tool()
    @_guarded
    def answer(question: str, task_id: str | None = None, scopes: list[str] | None = None) -> str:
        """Answer a question using ONLY stored memories — grounded, never
        hallucinated. Returns an "I don't know" message if nothing in
        memory covers the question."""
        return get_mem().grounded_answer(question, task_id=task_id, scopes=scopes)

    @mcp.tool()
    @_guarded
    def scope(
        atom_id: str,
        set_scopes: list[str] | None = None,
        add: list[str] | None = None,
        remove: list[str] | None = None,
        keep: bool = False,
    ) -> dict[str, Any]:
        """Change which repos/tasks a memory applies to. `set_scopes`
        replaces the whole set; `add`/`remove` adjust it. "" means global.
        `keep=True` confirms the current scopes unchanged. Use after asking
        the user whether a memory applies everywhere or only to specific
        repos — this also resolves its pending scope review."""
        mem = get_mem()
        if keep:
            atom = mem.confirm_scopes(atom_id)
        else:
            atom = mem.set_scopes(atom_id, scopes=set_scopes, add=add, remove=remove)
        return {"id": atom.id, "scopes": list(atom.scopes)}

    @mcp.tool()
    @_guarded
    def update(
        atom_id: str,
        content: str | None = None,
        category: Category | None = None,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        """Correct an existing memory. If `content` actually changes, a
        new atom is created superseding the old one (the old atom stays
        queryable, never silently erased); if only category/confidence
        change, the atom is updated in place."""
        atom = get_mem().update(
            atom_id,
            content=content,
            category=category,
            confidence=confidence,
        )
        return {
            "id": atom.id,
            "category": atom.category.value,
            "content": atom.content,
            "task_id": atom.task_id,
        }

    @mcp.tool()
    @_guarded
    def forget(atom_id: str) -> str:
        """Delete a memory, with provenance logging. Use when the user
        explicitly asks you to forget or delete something they told you."""
        found = get_mem().forget(atom_id)
        return f"forgot {atom_id}" if found else f"no such memory: {atom_id}"

    @mcp.tool()
    @_guarded
    def log_turn(
        session_id: str,
        role: Literal["user", "assistant", "system", "tool"],
        content: str,
        task_id: str | None = None,
        scopes: list[str] | None = None,
    ) -> str:
        """Log one raw conversation turn for later distillation into typed
        memory via `improve`. Call this for every turn worth remembering,
        even before you know what parts of it are durable — distillation
        figures that out later. `scopes` lists other repos this turn
        touched, beyond the session's own `task_id`."""
        get_mem().log_turn(session_id, role, content, task_id=task_id, scopes=scopes)
        return "ok"

    @mcp.tool()
    @_guarded
    def improve(session_ids: list[str] | None = None) -> dict[str, Any]:
        """Distill logged session turns into typed permanent memory, in
        the background (returns immediately). Pass `session_ids=None` to
        distill every session that has new, undistilled turns."""
        return get_mem().improve(session_ids, background=True)

    @mcp.tool()
    @_guarded
    def ingest(path: str, task_id: str | None = None) -> dict[str, Any]:
        """Ingest a file or directory (Markdown/Obsidian vault, plain
        text, or PDF) into memory. Safe to call repeatedly — only changed
        files are re-ingested."""
        resolved = Path(path).expanduser()
        if not resolved.exists():
            raise ToolError(f"path does not exist: {path}")
        return get_mem().ingest(str(resolved), task_id=task_id)

    @mcp.tool()
    @_guarded
    def stats() -> dict[str, Any]:
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
        _mem = open_local_memory()
    return _mem


def main() -> None:
    """stdio entry point. Opens the stores directly, so it must not run
    alongside the daemon (embedded Qdrant/SQLite are single-process) —
    if a daemon already answers on settings.server_url (whether or not it
    accepts this client's token), tell the user to point their MCP client
    at its HTTP endpoint instead and exit. A "rejected" probe (daemon up,
    wrong/missing MEMSTRAND_TOKEN) is NOT "daemon down": opening the local
    stores anyway would crash on Qdrant's on-disk lock, so it gets the
    same refusal, plus a hint about the token."""
    settings = get_settings()
    token = os.environ.get("MEMSTRAND_TOKEN")
    status = probe_daemon(settings.server_url, token=token)
    if status in ("up", "rejected"):
        message = (
            f"memstrand daemon is already running at {settings.server_url} "
            "and owns the stores. Configure your MCP client with the HTTP "
            f"endpoint {settings.server_url}/mcp instead of stdio."
        )
        if status == "rejected":
            message += (
                " The daemon rejected this probe — if it requires a token, set "
                "MEMSTRAND_TOKEN (sent as an `Authorization: Bearer <token>` header) "
                "or configure your MCP client's HTTP connection with that header."
            )
        print(message, file=sys.stderr)
        sys.exit(1)
    build_mcp(_lazy_memory).run()


if __name__ == "__main__":
    main()
