"""Keyword (BM25) index over atom content — SQLite FTS5.

Hybrid recall fuses this with the vector store (reciprocal rank fusion)
so recall still works for things embeddings are bad at: ids, codes, exact
tokens like "ZX-4471". Shares the scope/TTL rules used everywhere else so
a keyword hit can never leak across task/host scope or surface an expired
atom.
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from datetime import UTC, datetime

from ..config import ensure_private_dir, secure_file
from ..ranking import keyword_strength
from ..schema import MemoryAtom, ScopedQuery, encode_scopes

_WORD_RE = re.compile(r"\w+", re.UNICODE)

_SCHEMA_COLS = "content, atom_id UNINDEXED, task_id UNINDEXED, host UNINDEXED, category UNINDEXED, expires_at UNINDEXED"


def _sanitize_query(text: str) -> str:
    """Extract word tokens, quote each (so FTS5 syntax characters in
    content never break the query), OR-join. Empty input -> empty query."""
    tokens = _WORD_RE.findall(text)
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)


def _scope_clause(query: ScopedQuery) -> tuple[str, list]:
    """Scoped query sees its own scopes PLUS globals — matching the vector
    store's rule exactly. The `task_id` column holds the encoded scope set
    (`schema.encode_scopes`: "||" = global member); legacy plain values
    ("" = global, "x" = one task) match too, until rewritten."""
    scope_terms = ["task_id = ''", "instr(task_id, '||') > 0"]
    params: list = []
    for scope in query.scopes:
        scope_terms.append("task_id = ?")
        scope_terms.append("instr(task_id, ?) > 0")
        params.extend([scope, f"|{scope}|"])
    clauses = [f"({' OR '.join(scope_terms)})", "(host = ? OR host = '')"]
    params.append(query.host or "")
    if query.categories:
        placeholders = ", ".join("?" for _ in query.categories)
        clauses.append(f"(category IN ({placeholders}))")
        params.extend(c.value for c in query.categories)
    return " AND ".join(clauses), params


class SqliteKeywordIndex:
    def __init__(self, path: str = "./memstrand.db"):
        is_new = path == ":memory:" or not os.path.exists(path)
        if path != ":memory:":
            parent = os.path.dirname(path)
            if parent:
                # deep-review fix #12: directories memstrand creates are 0o700.
                ensure_private_dir(parent)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        # Wait up to 5s for a lock instead of raising "database is locked"
        # immediately — several stores share this file across threads/
        # processes (graph + sessions + keyword index).
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()
        if is_new and path != ":memory:":
            # deep-review fix #12: a db file we created is 0o600 (shared
            # with the other sqlite-backed stores; harmless to chmod again
            # if another store got here first).
            secure_file(path)
            secure_file(f"{path}-wal")
            secure_file(f"{path}-shm")

    def _init_schema(self) -> None:
        with self._lock:
            try:
                self.conn.execute(
                    f"""CREATE VIRTUAL TABLE IF NOT EXISTS atoms_fts USING fts5(
                        {_SCHEMA_COLS},
                        tokenize = 'porter unicode61 remove_diacritics 2')"""
                )
            except sqlite3.OperationalError:
                # porter tokenizer unavailable in this SQLite build
                self.conn.execute(
                    f"""CREATE VIRTUAL TABLE IF NOT EXISTS atoms_fts USING fts5(
                        {_SCHEMA_COLS},
                        tokenize = 'unicode61 remove_diacritics 2')"""
                )
            self.conn.commit()

    def add(self, atom: MemoryAtom) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM atoms_fts WHERE atom_id = ?", (atom.id,))
            self.conn.execute(
                "INSERT INTO atoms_fts(content, atom_id, task_id, host, category, expires_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    atom.content,
                    atom.id,
                    encode_scopes(atom.scopes),
                    atom.host or "",
                    atom.category.value,
                    atom.expires_at.timestamp() if atom.expires_at else 0.0,
                ),
            )
            self.conn.commit()

    def delete(self, atom_id: str) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM atoms_fts WHERE atom_id = ?", (atom_id,))
            self.conn.commit()

    def search(
        self,
        text: str,
        query: ScopedQuery,
        limit: int = 30,
        now: datetime | None = None,
    ) -> list[tuple[str, float]]:
        match = _sanitize_query(text)
        if not match:
            return []
        now = now or datetime.now(UTC)
        scope_sql, scope_params = _scope_clause(query)
        sql = (
            "SELECT atom_id, bm25(atoms_fts) FROM atoms_fts "
            f"WHERE atoms_fts MATCH ? AND {scope_sql} "
            "AND (expires_at <= 0.0 OR expires_at > ?) "
            "ORDER BY bm25(atoms_fts) LIMIT ?"
        )
        params = [match, *scope_params, now.timestamp(), limit]
        with self._lock:
            try:
                rows = self.conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                # malformed FTS5 MATCH syntax: treat as no keyword hits
                return []
        if not rows:
            return []
        # SQLite's bm25() is "lower (more negative) is better". Convert to
        # an ABSOLUTE 0..1 strength via `keyword_strength` rather than
        # min-max normalizing within this result set — a single (or tied)
        # hit must not always read 1.0 regardless of how weak the actual
        # match is; recall's relevance filtering depends on this being a
        # real, comparable-across-queries signal.
        return [(row[0], keyword_strength(row[1])) for row in rows]

    def close(self) -> None:
        with self._lock:
            self.conn.close()
