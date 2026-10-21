"""SQLite-backed GraphStore. The default backend.

Kuzu upstream was archived after the Oct 2025 acquisition, and embedded
Kuzu only tolerates a single writer process — a poor fit for a daemon
serving multiple agents. SQLite (WAL mode) is boring, embeddable, and
already a dependency via the session store, so it's the default here;
Kuzu stays available behind the same GraphStore protocol for anyone who
wants a real graph query language.

Shares the `memstrand.db` file with SqliteSessionStore: table names
(`atoms`, `edges`) are namespaced to avoid collisions.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import UTC, datetime

from ..config import ensure_private_dir, secure_file
from ..schema import (
    Category,
    MemoryAtom,
    MemoryEdge,
    Provenance,
    RelationType,
    decode_scopes,
    encode_scopes,
)
from .base import sort_hop_results

# Column order used for every atoms INSERT/SELECT — keep in sync with MemoryAtom.
# `task_id` holds the ENCODED scope set (`schema.encode_scopes`, e.g. "||a|");
# the column name is kept for on-disk compatibility, and legacy plain values
# ("a" / "") still decode as a one-element set.
ATOM_COLUMNS = (
    "id", "category", "content", "provenance", "confidence",
    "session_id", "task_id", "host", "agent",
    "created_at", "updated_at", "expires_at", "access_count",
    "fsrs_difficulty", "fsrs_stability_days", "fsrs_next_review",
    "source", "source_ref", "last_accessed_at",
)

EDGE_COLUMNS = ("source_id", "target_id", "relation", "confidence", "reason", "weight", "created_at")


def _iso(dt: datetime | None) -> str:
    return dt.isoformat() if dt else ""


def _row_to_atom(row: tuple) -> MemoryAtom:
    (
        aid, cat, content, prov, conf, sess, task, host, agent,
        created, updated, expires, access_count,
        fsrs_diff, fsrs_stab, fsrs_next, source, source_ref, last_accessed,
    ) = row
    created_at = datetime.fromisoformat(created) if created else datetime.now(UTC)
    return MemoryAtom(
        id=aid,
        category=Category(cat),
        content=content,
        provenance=Provenance(prov),
        confidence=conf,
        session_id=sess or None,
        scopes=decode_scopes(task),
        host=host or None,
        agent=agent or None,
        created_at=created_at,
        # migrated/legacy rows may lack updated_at: fall back to created_at
        updated_at=datetime.fromisoformat(updated) if updated else created_at,
        expires_at=datetime.fromisoformat(expires) if expires else None,
        access_count=access_count or 0,
        fsrs_difficulty=fsrs_diff if fsrs_diff is not None else 0.5,
        fsrs_stability_days=fsrs_stab if fsrs_stab is not None else 30.0,
        fsrs_next_review=datetime.fromisoformat(fsrs_next) if fsrs_next else None,
        source=source or None,
        source_ref=source_ref or None,
        last_accessed_at=datetime.fromisoformat(last_accessed) if last_accessed else None,
    )


def _atom_to_params(atom: MemoryAtom) -> tuple:
    return (
        atom.id, atom.category.value, atom.content, atom.provenance.value, atom.confidence,
        atom.session_id or "", encode_scopes(atom.scopes), atom.host or "", atom.agent or "",
        _iso(atom.created_at), _iso(atom.updated_at), _iso(atom.expires_at), atom.access_count,
        atom.fsrs_difficulty, atom.fsrs_stability_days, _iso(atom.fsrs_next_review),
        atom.source or "", atom.source_ref or "", _iso(atom.last_accessed_at),
    )


def _atom_filter_clause(
    category: str | None, task_id: str | None, host: str | None, agent: str | None = None
) -> tuple[str, list]:
    """Shared WHERE-clause builder for list_atoms/count_atoms so the two
    can never drift out of sync on filter semantics (fix #10).

    `task_id` filters by scope MEMBERSHIP ("" = atoms in the global scope),
    matching both the encoded form and legacy plain values."""
    clauses: list[str] = []
    params: list = []
    for col, value in (("category", category), ("host", host), ("agent", agent)):
        if value is None:
            continue
        if value == "":
            clauses.append(f"({col} IS NULL OR {col} = '')")
        else:
            clauses.append(f"{col} = ?")
            params.append(value)
    if task_id is not None:
        if task_id == "":
            clauses.append("(task_id IS NULL OR task_id = '' OR instr(task_id, '||') > 0)")
        else:
            clauses.append("(task_id = ? OR instr(task_id, ?) > 0)")
            params.extend([task_id, f"|{task_id}|"])
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


def _row_to_edge(row: tuple) -> MemoryEdge:
    (src, dst, rel, conf, reason, weight, created) = row
    return MemoryEdge(
        source_id=src,
        target_id=dst,
        relation=RelationType(rel),
        confidence=conf,
        reason=reason or "",
        weight=weight,
        created_at=datetime.fromisoformat(created) if created else datetime.now(UTC),
    )


class SqliteGraphStore:
    """Typed relationship graph over SQLite. Thread-safe: a daemon may
    serve several agents concurrently from one connection."""

    def __init__(self, path: str = "./memstrand.db"):
        parent = os.path.dirname(path)
        if parent:
            # deep-review fix #12: directories memstrand creates are 0o700.
            ensure_private_dir(parent)
        self._lock = threading.RLock()
        is_new = not os.path.exists(path)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        # Wait up to 5s for a lock instead of raising "database is locked"
        # immediately — several stores share this file across threads/
        # processes (graph + sessions + keyword index).
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()
        if is_new:
            # deep-review fix #12: a db file we created is 0o600 (shared
            # with SqliteSessionStore/SqliteKeywordIndex on the same path;
            # harmless to chmod again if another store got here first).
            secure_file(path)
            secure_file(f"{path}-wal")
            secure_file(f"{path}-shm")

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.execute(
                f"""CREATE TABLE IF NOT EXISTS atoms(
                    {", ".join(ATOM_COLUMNS)},
                    PRIMARY KEY (id))"""
            )
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS edges(
                    edge_pk INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT, target_id TEXT, relation TEXT,
                    confidence REAL, reason TEXT, weight REAL, created_at TEXT,
                    UNIQUE(source_id, target_id, relation))"""
            )
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_atoms_source ON atoms(source)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_atoms_created ON atoms(created_at)")
            self.conn.commit()

    # -- atoms ----------------------------------------------------------------

    def add_atom(self, atom: MemoryAtom) -> None:
        cols = ", ".join(ATOM_COLUMNS)
        placeholders = ", ".join("?" for _ in ATOM_COLUMNS)
        updates = ", ".join(f"{c}=excluded.{c}" for c in ATOM_COLUMNS if c != "id")
        with self._lock:
            self.conn.execute(
                f"INSERT INTO atoms({cols}) VALUES ({placeholders}) "
                f"ON CONFLICT(id) DO UPDATE SET {updates}",
                _atom_to_params(atom),
            )
            self.conn.commit()

    def get_atom(self, atom_id: str) -> MemoryAtom | None:
        cols = ", ".join(ATOM_COLUMNS)
        with self._lock:
            row = self.conn.execute(
                f"SELECT {cols} FROM atoms WHERE id=?", (atom_id,)
            ).fetchone()
        return _row_to_atom(row) if row else None

    def get_atoms(self, atom_ids: list[str]) -> dict[str, MemoryAtom]:
        if not atom_ids:
            return {}
        cols = ", ".join(ATOM_COLUMNS)
        placeholders = ", ".join("?" for _ in atom_ids)
        with self._lock:
            rows = self.conn.execute(
                f"SELECT {cols} FROM atoms WHERE id IN ({placeholders})", atom_ids
            ).fetchall()
        return {row[0]: _row_to_atom(row) for row in rows}

    def list_atoms(
        self,
        category: str | None = None,
        task_id: str | None = None,
        host: str | None = None,
        limit: int = 100,
        offset: int = 0,
        agent: str | None = None,
    ) -> list[MemoryAtom]:
        cols = ", ".join(ATOM_COLUMNS)
        where, params = _atom_filter_clause(category, task_id, host, agent=agent)
        params = [*params, limit, offset]
        with self._lock:
            rows = self.conn.execute(
                f"SELECT {cols} FROM atoms {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                params,
            ).fetchall()
        return [_row_to_atom(row) for row in rows]

    def count_atoms(
        self,
        category: str | None = None,
        task_id: str | None = None,
        host: str | None = None,
        agent: str | None = None,
    ) -> int:
        where, params = _atom_filter_clause(category, task_id, host, agent=agent)
        with self._lock:
            row = self.conn.execute(f"SELECT COUNT(*) FROM atoms {where}", params).fetchone()
        return int(row[0])

    def ids_by_source(self, source: str) -> list[str]:
        with self._lock:
            rows = self.conn.execute("SELECT id FROM atoms WHERE source=?", (source,)).fetchall()
        return [r[0] for r in rows]

    def expired_ids(self, now: datetime) -> list[str]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT id FROM atoms WHERE expires_at IS NOT NULL AND expires_at != '' "
                "AND expires_at <= ?",
                (_iso(now),),
            ).fetchall()
        return [r[0] for r in rows]

    def record_access(self, atom_ids: list[str], when: datetime) -> None:
        if not atom_ids:
            return
        with self._lock:
            self.conn.executemany(
                "UPDATE atoms SET access_count = access_count + 1, last_accessed_at = ? WHERE id = ?",
                [(_iso(when), aid) for aid in atom_ids],
            )
            self.conn.commit()

    def update_fsrs(
        self, atom_id: str, difficulty: float, stability_days: float, next_review: datetime | None
    ) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE atoms SET fsrs_difficulty=?, fsrs_stability_days=?, fsrs_next_review=? WHERE id=?",
                (difficulty, stability_days, _iso(next_review), atom_id),
            )
            self.conn.commit()

    def delete_atom(self, atom_id: str) -> None:
        """Removes the atom and every edge touching it. Facts are never
        pruned by consolidation, but an explicit forget() removes."""
        with self._lock:
            self.conn.execute("DELETE FROM atoms WHERE id=?", (atom_id,))
            self.conn.execute(
                "DELETE FROM edges WHERE source_id=? OR target_id=?", (atom_id, atom_id)
            )
            self.conn.commit()

    # -- edges ------------------------------------------------------------------

    def add_edge(self, edge: MemoryEdge) -> None:
        """No-op if either endpoint is missing. Re-adding the same
        (source, target, relation) updates confidence/reason in place
        rather than duplicating — weight (Hebbian strength) is untouched."""
        with self._lock:
            if self.get_atom(edge.source_id) is None or self.get_atom(edge.target_id) is None:
                return
            self.conn.execute(
                """INSERT INTO edges(source_id, target_id, relation, confidence, reason, weight, created_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(source_id, target_id, relation)
                   DO UPDATE SET confidence=excluded.confidence, reason=excluded.reason""",
                (
                    edge.source_id, edge.target_id, edge.relation.value,
                    edge.confidence, edge.reason, edge.weight, _iso(edge.created_at),
                ),
            )
            self.conn.commit()

    def all_edges(self) -> list[MemoryEdge]:
        cols = ", ".join(EDGE_COLUMNS)
        with self._lock:
            rows = self.conn.execute(f"SELECT {cols} FROM edges").fetchall()
        return [_row_to_edge(row) for row in rows]

    def _one_hop(self, atom_id: str, relation: str | None) -> list[tuple[MemoryAtom, MemoryEdge]]:
        cols = ", ".join(EDGE_COLUMNS)
        query = (
            f"SELECT {cols} FROM edges WHERE (source_id=? OR target_id=?)"
        )
        params: list = [atom_id, atom_id]
        if relation:
            query += " AND relation=?"
            params.append(relation)
        # Deterministic order (fix: cross-backend neighbors(depth>1) parity)
        # -- see `sort_hop_results`'s docstring. SQLite gives no ordering
        # guarantee without this, and a plain Python sort below is what
        # actually makes the result identical to Kuzu's (this ORDER BY is
        # belt-and-suspenders, cheap and harmless, but the real guarantee
        # comes from `sort_hop_results`).
        query += " ORDER BY weight DESC, relation ASC, source_id ASC, target_id ASC"
        with self._lock:
            rows = self.conn.execute(query, params).fetchall()
            out = []
            for row in rows:
                edge = _row_to_edge(row)
                other_id = edge.target_id if edge.source_id == atom_id else edge.source_id
                other = self.get_atom(other_id)
                if other is not None:
                    out.append((other, edge))
        return sort_hop_results(out)

    def neighbors(
        self, atom_id: str, relation: str | None = None, depth: int = 1
    ) -> list[tuple[MemoryAtom, MemoryEdge]]:
        """Both directions; the returned edge keeps its true source/target.
        depth>1 is a BFS over the depth-1 primitive — never revisits a
        node and never returns the start atom.

        Parallel edges (more than one edge connecting the same pair — a
        second relation, or one edge in each direction) mean a node can be
        reachable via more than one edge at the same depth; the dedup-by-id
        `seen` set below keeps only the FIRST one encountered. `_one_hop`
        always returns its rows pre-sorted via `sort_hop_results`
        (highest weight, then relation name, then source/target id), so
        that "first" edge is deterministic and identical across every
        `GraphStore` backend, not an artifact of SQLite's/Kuzu's own
        iteration order."""
        with self._lock:
            out = self._one_hop(atom_id, relation)
            if depth <= 1:
                return out
            seen = {atom_id} | {a.id for a, _ in out}
            frontier = [a.id for a, _ in out]
            for _ in range(depth - 1):
                nxt = []
                for fid in frontier:
                    for na, ne in self._one_hop(fid, relation):
                        if na.id not in seen:
                            seen.add(na.id)
                            out.append((na, ne))
                            nxt.append(na.id)
                frontier = nxt
            return out

    def contradicting(self, atom_id: str) -> list[tuple[MemoryAtom, MemoryEdge]]:
        return self.neighbors(atom_id, relation=RelationType.contradicts.value)

    def superseded_ids(self, atom_ids: list[str]) -> set[str]:
        """Subset of atom_ids that are the TARGET of a 'supersedes' edge
        — i.e. an atom has been replaced and something newer exists."""
        if not atom_ids:
            return set()
        placeholders = ", ".join("?" for _ in atom_ids)
        with self._lock:
            rows = self.conn.execute(
                f"SELECT DISTINCT target_id FROM edges WHERE relation=? AND target_id IN ({placeholders})",
                [RelationType.supersedes.value, *atom_ids],
            ).fetchall()
        return {r[0] for r in rows}

    def strengthen_edge(self, a_id: str, b_id: str, amount: float = 0.1) -> None:
        """Hebbian: use strengthens the edge, capped. Matches edges in
        EITHER direction between a and b (there may be more than one
        relation type linking the same pair)."""
        with self._lock:
            self.conn.execute(
                """UPDATE edges SET weight = weight + ? * (1 - weight / 10)
                   WHERE (source_id=? AND target_id=?) OR (source_id=? AND target_id=?)""",
                (amount, a_id, b_id, b_id, a_id),
            )
            self.conn.commit()

    def decay_edges(self, factor: float = 0.98, prune_below: float = 0.1) -> int:
        """Hebbian decay. Prunes weak EDGES — never facts (atoms)."""
        with self._lock:
            self.conn.execute("UPDATE edges SET weight = weight * ?", (factor,))
            cur = self.conn.execute("DELETE FROM edges WHERE weight < ?", (prune_below,))
            self.conn.commit()
            return cur.rowcount if cur.rowcount is not None else 0

    def close(self) -> None:
        with self._lock:
            self.conn.close()
