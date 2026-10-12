"""MCP server: memstrand tools for any MCP-compatible agent.

    python -m memstrand.mcp_server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .api import Memory
from .schema import Category, Provenance

mcp = FastMCP("memstrand")
_mem: Memory | None = None


def _memory() -> Memory:
    global _mem
    if _mem is None:
        _mem = Memory()
    return _mem


@mcp.tool()
def remember(
    content: str,
    category: str = "fact",
    provenance: str = "explicit_statement",
    confidence: float = 0.8,
    session_id: str | None = None,
    task_id: str | None = None,
) -> str:
    """Store one atomic memory."""
    atom = _memory().remember(
        content,
        category=Category(category),
        provenance=Provenance(provenance),
        confidence=confidence,
        session_id=session_id,
        task_id=task_id,
    )
    return atom.id


@mcp.tool()
def recall(
    query: str,
    session_id: str | None = None,
    task_id: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Scoped semantic recall."""
    return _memory().recall(
        query, session_id=session_id, task_id=task_id, limit=limit
    )


@mcp.tool()
def forget(atom_id: str) -> str:
    """Delete a memory with provenance logging."""
    _memory().forget(atom_id)
    return f"forgot {atom_id}"


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
