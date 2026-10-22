"""Kuzu-backed GraphStore. Embedded, zero infrastructure.

Pinned to kuzu==0.11.3 (upstream archived Oct 2025). Kuzu is no longer
the default backend (SqliteGraphStore is — see graph_sqlite.py: embedded
Kuzu only tolerates a single writer process, a poor fit for a daemon)
but stays available behind the same GraphStore protocol. `kuzu` itself
is an optional extra now (`pip install memstrand[kuzu]`), so the import
is lazy: importing this module must not require kuzu to be installed.
"""

from __future__ import annotations

import os
import threading
from datetime import UTC, datetime

from ..config import ensure_private_dir
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


def _atom_filter_clause(
    category: str | None, task_id: str | None, host: str | None, alias: str = "a",
    agent: str | None = None, tag: str | None = None,
) -> tuple[str, dict]:
    """Shared WHERE-clause builder for list_atoms/count_atoms so the two
    can never drift out of sync on filter semantics (fix #10).

    `task_id` filters by scope MEMBERSHIP ("" = atoms in the global scope);
    the column holds the encoded scope set (see graph_sqlite.ATOM_COLUMNS)."""
    clauses: list[str] = []
    params: dict = {}
    for col, value in (("category", category), ("host", host), ("agent", agent)):
        if value is None:
            continue
        if value == "":
            clauses.append(f"({alias}.{col} IS NULL OR {alias}.{col} = '')")
        else:
            clauses.append(f"{alias}.{col} = ${col}")
            params[col] = value
    if tag is not None:
        clauses.append(f"contains({alias}.tags, $tag_member)")
        params["tag_member"] = f"|{tag}|"
    if task_id is not None:
        col = f"{alias}.task_id"
        if task_id == "":
            clauses.append(f"({col} IS NULL OR {col} = '' OR contains({col}, '||'))")
        else:
            clauses.append(f"({col} = $task_id OR contains({col}, $task_member))")
            params["task_id"] = task_id
            params["task_member"] = f"|{task_id}|"
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


# Columns added after the v0.1 schema. ALTER TABLE ... ADD IF NOT EXISTS is
# idempotent, so re-running this against a fresh (already-current) DB or an
# old v0.1 DB both just work — no separate migration-detection path needed.
_ATOM_MIGRATIONS = (
    ("agent", "STRING", "''"),
    ("updated_at", "STRING", "''"),
    ("access_count", "INT64", "0"),
    ("fsrs_difficulty", "DOUBLE", "0.5"),
    ("fsrs_stability_days", "DOUBLE", "30.0"),
    ("fsrs_next_review", "STRING", "''"),
    ("source", "STRING", "''"),
    ("source_ref", "STRING", "''"),
    ("last_accessed_at", "STRING", "''"),
    ("tags", "STRING", "''"),  # encoded like scopes: "|a|b|"
)
_LINK_MIGRATIONS = (("created_at", "STRING", "''"),)


def _row_to_atom(row: tuple) -> MemoryAtom:
    (
        aid, cat, content, prov, conf, sess, task, host, agent,
        created, expires, updated, access_count,
        fsrs_diff, fsrs_stab, fsrs_next, source, source_ref, last_accessed, tags,
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
        # migrated v0.1 rows lack updated_at: fall back to created_at
        updated_at=datetime.fromisoformat(updated) if updated else created_at,
        expires_at=datetime.fromisoformat(expires) if expires else None,
        access_count=access_count or 0,
        fsrs_difficulty=fsrs_diff if fsrs_diff is not None else 0.5,
        fsrs_stability_days=fsrs_stab if fsrs_stab is not None else 30.0,
        fsrs_next_review=datetime.fromisoformat(fsrs_next) if fsrs_next else None,
        source=source or None,
        source_ref=source_ref or None,
        last_accessed_at=datetime.fromisoformat(last_accessed) if last_accessed else None,
        tags=[t for t in (tags or "").split("|") if t],
    )


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


class KuzuGraphStore:
    def __init__(self, path: str = "./kuzu_data"):
        try:
            import kuzu
        except ImportError as e:
            raise ImportError(
                "KuzuGraphStore requires the optional 'kuzu' extra. "
                "Install it with: pip install 'memstrand[kuzu]'"
            ) from e
        self._lock = threading.RLock()
        # deep-review fix #12: `path` itself is a Kuzu-managed database
        # file/bundle (Kuzu errors out if it already exists as a plain
        # directory), so only the PARENT directory is ours to harden here
        # — created 0o700 under our own umask before Kuzu creates `path`.
        parent = os.path.dirname(path)
        if parent:
            ensure_private_dir(parent)
        self.db = kuzu.Database(path)
        self.conn = kuzu.Connection(self.db)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.execute(
                """CREATE NODE TABLE IF NOT EXISTS Atom(
                    id STRING, category STRING, content STRING,
                    provenance STRING, confidence DOUBLE,
                    session_id STRING, task_id STRING, host STRING, agent STRING,
                    created_at STRING, expires_at STRING, updated_at STRING,
                    access_count INT64,
                    fsrs_difficulty DOUBLE, fsrs_stability_days DOUBLE, fsrs_next_review STRING,
                    source STRING, source_ref STRING, last_accessed_at STRING, tags STRING,
                    PRIMARY KEY (id))"""
            )
            self.conn.execute(
                """CREATE REL TABLE IF NOT EXISTS LINK(
                    FROM Atom TO Atom,
                    relation STRING, confidence DOUBLE,
                    reason STRING, weight DOUBLE, created_at STRING)"""
            )
            # migrate any v0.1 database that predates the columns above
            for col, typ, default in _ATOM_MIGRATIONS:
                self.conn.execute(f"ALTER TABLE Atom ADD IF NOT EXISTS {col} {typ} DEFAULT {default}")
            for col, typ, default in _LINK_MIGRATIONS:
                self.conn.execute(f"ALTER TABLE LINK ADD IF NOT EXISTS {col} {typ} DEFAULT {default}")

    def add_atom(self, atom: MemoryAtom) -> None:
        with self._lock:
            self.conn.execute(
                """MERGE (a:Atom {id: $id})
                   SET a.category=$cat, a.content=$content,
                       a.provenance=$prov, a.confidence=$conf,
                       a.session_id=$sess, a.task_id=$task, a.host=$host, a.agent=$agent,
                       a.created_at=$created, a.expires_at=$expires, a.updated_at=$updated,
                       a.access_count=$access_count,
                       a.fsrs_difficulty=$fsrs_diff, a.fsrs_stability_days=$fsrs_stab,
                       a.fsrs_next_review=$fsrs_next,
                       a.source=$source, a.source_ref=$source_ref,
                       a.last_accessed_at=$last_accessed, a.tags=$tags""",
                {
                    "id": atom.id, "cat": atom.category.value,
                    "content": atom.content, "prov": atom.provenance.value,
                    "conf": atom.confidence,
                    "sess": atom.session_id or "", "task": encode_scopes(atom.scopes),
                    "host": atom.host or "", "agent": atom.agent or "",
                    "created": atom.created_at.isoformat(),
                    "expires": atom.expires_at.isoformat() if atom.expires_at else "",
                    "updated": atom.updated_at.isoformat(),
                    "access_count": atom.access_count,
                    "fsrs_diff": atom.fsrs_difficulty,
                    "fsrs_stab": atom.fsrs_stability_days,
                    "fsrs_next": atom.fsrs_next_review.isoformat() if atom.fsrs_next_review else "",
                    "source": atom.source or "", "source_ref": atom.source_ref or "",
                    "last_accessed": atom.last_accessed_at.isoformat() if atom.last_accessed_at else "",
                    "tags": f"|{'|'.join(atom.tags)}|" if atom.tags else "",
                },
            )

    def add_edge(self, edge: MemoryEdge) -> None:
        """No-op if either endpoint is missing. Re-adding the same
        (source, target, relation) updates confidence/reason in place
        instead of creating a duplicate edge."""
        with self._lock:
            if self.get_atom(edge.source_id) is None or self.get_atom(edge.target_id) is None:
                return
            existing = self.conn.execute(
                """MATCH (a:Atom {id: $src})-[l:LINK {relation: $rel}]->(b:Atom {id: $dst})
                   RETURN count(*)""",
                {"src": edge.source_id, "dst": edge.target_id, "rel": edge.relation.value},
            ).get_next()
            if existing and existing[0] > 0:
                self.conn.execute(
                    """MATCH (a:Atom {id: $src})-[l:LINK {relation: $rel}]->(b:Atom {id: $dst})
                       SET l.confidence=$conf, l.reason=$reason""",
                    {
                        "src": edge.source_id, "dst": edge.target_id, "rel": edge.relation.value,
                        "conf": edge.confidence, "reason": edge.reason,
                    },
                )
                return
            self.conn.execute(
                """MATCH (a:Atom {id: $src}), (b:Atom {id: $dst})
                   CREATE (a)-[l:LINK {relation: $rel, confidence: $conf,
                       reason: $reason, weight: $w, created_at: $created}]->(b)""",
                {
                    "src": edge.source_id, "dst": edge.target_id,
                    "rel": edge.relation.value, "conf": edge.confidence,
                    "reason": edge.reason, "w": edge.weight,
                    "created": edge.created_at.isoformat(),
                },
            )

    def get_atom(self, atom_id: str) -> MemoryAtom | None:
        """Row -> MemoryAtom hydration. Vector store is still the
        primary read path; this is the fallback for graph-only reads."""
        with self._lock:
            rows = self.conn.execute(
                """MATCH (a:Atom {id: $id})
                   RETURN a.id, a.category, a.content, a.provenance,
                          a.confidence, a.session_id, a.task_id, a.host, a.agent,
                          a.created_at, a.expires_at, a.updated_at, a.access_count,
                          a.fsrs_difficulty, a.fsrs_stability_days, a.fsrs_next_review,
                          a.source, a.source_ref, a.last_accessed_at, a.tags""",
                {"id": atom_id},
            ).get_all()
        return _row_to_atom(rows[0]) if rows else None

    def get_atoms(self, atom_ids: list[str]) -> dict[str, MemoryAtom]:
        if not atom_ids:
            return {}
        with self._lock:
            rows = self.conn.execute(
                """MATCH (a:Atom)
                   WHERE a.id IN $ids
                   RETURN a.id, a.category, a.content, a.provenance,
                          a.confidence, a.session_id, a.task_id, a.host, a.agent,
                          a.created_at, a.expires_at, a.updated_at, a.access_count,
                          a.fsrs_difficulty, a.fsrs_stability_days, a.fsrs_next_review,
                          a.source, a.source_ref, a.last_accessed_at, a.tags""",
                {"ids": atom_ids},
            ).get_all()
        return {row[0]: _row_to_atom(row) for row in rows}

    def list_atoms(
        self,
        category: str | None = None,
        task_id: str | None = None,
        host: str | None = None,
        limit: int = 100,
        offset: int = 0,
        agent: str | None = None,
        tag: str | None = None,
    ) -> list[MemoryAtom]:
        where, params = _atom_filter_clause(category, task_id, host, agent=agent, tag=tag)
        with self._lock:
            rows = self.conn.execute(
                f"""MATCH (a:Atom) {where}
                    RETURN a.id, a.category, a.content, a.provenance,
                           a.confidence, a.session_id, a.task_id, a.host, a.agent,
                           a.created_at, a.expires_at, a.updated_at, a.access_count,
                           a.fsrs_difficulty, a.fsrs_stability_days, a.fsrs_next_review,
                           a.source, a.source_ref, a.last_accessed_at, a.tags
                    ORDER BY a.created_at DESC
                    SKIP {int(offset)} LIMIT {int(limit)}""",
                params,
            ).get_all()
        return [_row_to_atom(row) for row in rows]

    def count_atoms(
        self,
        category: str | None = None,
        task_id: str | None = None,
        host: str | None = None,
        agent: str | None = None,
        tag: str | None = None,
    ) -> int:
        where, params = _atom_filter_clause(category, task_id, host, agent=agent, tag=tag)
        with self._lock:
            row = self.conn.execute(f"MATCH (a:Atom) {where} RETURN count(*)", params).get_next()
        return int(row[0]) if row else 0

    def ids_by_source(self, source: str) -> list[str]:
        with self._lock:
            rows = self.conn.execute(
                "MATCH (a:Atom {source: $source}) RETURN a.id", {"source": source}
            ).get_all()
        return [r[0] for r in rows]

    def expired_ids(self, now: datetime) -> list[str]:
        with self._lock:
            rows = self.conn.execute(
                """MATCH (a:Atom)
                   WHERE a.expires_at IS NOT NULL AND a.expires_at <> '' AND a.expires_at <= $now
                   RETURN a.id""",
                {"now": now.isoformat()},
            ).get_all()
        return [r[0] for r in rows]

    def record_access(self, atom_ids: list[str], when: datetime) -> None:
        if not atom_ids:
            return
        with self._lock:
            # NB: the bind parameter must not be named $when — it collides
            # with the Cypher WHEN keyword and fails to parse.
            self.conn.execute(
                """MATCH (a:Atom)
                   WHERE a.id IN $ids
                   SET a.access_count = a.access_count + 1, a.last_accessed_at = $accessed_at""",
                {"ids": atom_ids, "accessed_at": when.isoformat()},
            )

    def update_fsrs(
        self, atom_id: str, difficulty: float, stability_days: float, next_review: datetime | None
    ) -> None:
        with self._lock:
            self.conn.execute(
                """MATCH (a:Atom {id: $id})
                   SET a.fsrs_difficulty=$diff, a.fsrs_stability_days=$stab,
                       a.fsrs_next_review=$next""",
                {
                    "id": atom_id, "diff": difficulty, "stab": stability_days,
                    "next": next_review.isoformat() if next_review else "",
                },
            )

    def all_edges(self) -> list[MemoryEdge]:
        with self._lock:
            rows = self.conn.execute(
                """MATCH (a:Atom)-[l:LINK]->(b:Atom)
                   RETURN a.id, b.id, l.relation, l.confidence, l.reason, l.weight, l.created_at"""
            ).get_all()
        return [_row_to_edge(r) for r in rows]

    _ATOM_COLS = (
        "b.id, b.category, b.content, b.provenance, b.confidence, b.session_id, "
        "b.task_id, b.host, b.agent, b.created_at, b.expires_at, b.updated_at, "
        "b.access_count, b.fsrs_difficulty, b.fsrs_stability_days, b.fsrs_next_review, "
        "b.source, b.source_ref, b.last_accessed_at, b.tags"
    )

    def _one_hop(
        self, atom_id: str, relation: str | None
    ) -> list[tuple[MemoryAtom, MemoryEdge]]:
        """Both directions, true edge direction preserved. Kuzu's
        undirected pattern `(a)-[l]-(b)` always binds `a` to the queried
        node regardless of which way the edge actually points, so
        `a.id, b.id` is NOT reliably (source, target) — it was silently
        swapping direction on incoming edges. Two directed queries avoid
        that: one for atom_id as source, one for atom_id as target."""
        rel_filter = "AND l.relation = $rel" if relation else ""
        params: dict = {"id": atom_id}
        if relation:
            params["rel"] = relation
        out = []
        with self._lock:
            outgoing = self.conn.execute(
                f"""MATCH (a:Atom {{id: $id}})-[l:LINK]->(b:Atom)
                    WHERE b.id <> $id {rel_filter}
                    RETURN {self._ATOM_COLS},
                           a.id AS src_id, b.id AS dst_id,
                           l.relation, l.confidence, l.reason, l.weight, l.created_at""",
                params,
            ).get_all()
            incoming = self.conn.execute(
                f"""MATCH (b:Atom)-[l:LINK]->(a:Atom {{id: $id}})
                    WHERE b.id <> $id {rel_filter}
                    RETURN {self._ATOM_COLS},
                           b.id AS src_id, a.id AS dst_id,
                           l.relation, l.confidence, l.reason, l.weight, l.created_at""",
                params,
            ).get_all()
        for r in outgoing + incoming:
            out.append((_row_to_atom(r[:20]), _row_to_edge(r[20:])))
        # Deterministic order (fix: cross-backend neighbors(depth>1) parity)
        # -- Cypher gives no ordering guarantee either, and the outgoing/
        # incoming results are two separately-ordered result sets
        # concatenated together besides. `sort_hop_results` is the one
        # shared rule (see its docstring in stores/base.py) every
        # `GraphStore` backend's `_one_hop` sorts through, so a node
        # reachable via more than one parallel edge resolves to the same
        # "first" edge here as it does in SqliteGraphStore.
        return sort_hop_results(out)

    def neighbors(
        self, atom_id: str, relation: str | None = None, depth: int = 1
    ) -> list[tuple[MemoryAtom, MemoryEdge]]:
        """Depth-1 neighbor hydration, both directions. relation filters
        to one edge type (e.g. 'contradicts'). Deeper traversal is a BFS
        over this primitive — depth>1 expands one hop at a time, never
        revisiting a node and never returning the start atom.

        Parallel edges (more than one edge connecting the same pair) mean a
        node can be reachable via more than one edge at the same depth; the
        dedup-by-id `seen` set below keeps only the FIRST one encountered.
        `_one_hop` always returns its rows pre-sorted via
        `sort_hop_results`, so that "first" edge is deterministic and
        identical to `SqliteGraphStore`'s choice for the same data."""
        out = self._one_hop(atom_id, relation)
        if depth > 1:
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
        if not atom_ids:
            return set()
        with self._lock:
            rows = self.conn.execute(
                """MATCH (a:Atom)-[l:LINK {relation: $rel}]->(b:Atom)
                   WHERE b.id IN $ids
                   RETURN DISTINCT b.id""",
                {"rel": RelationType.supersedes.value, "ids": atom_ids},
            ).get_all()
        return {r[0] for r in rows}

    def tag_counts(self, task_id: str | None = None, limit: int = 100) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        for atom in self.list_atoms(task_id=task_id, limit=1_000_000):
            for tag in atom.tags:
                counts[tag] = counts.get(tag, 0) + 1
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]

    def related_by_tags(self, atom_id: str, min_shared: int = 2, limit: int = 20) -> list[tuple[str, int]]:
        me = self.get_atom(atom_id)
        if me is None or not me.tags:
            return []
        mine = set(me.tags)
        scored = [
            (a.id, len(mine & set(a.tags)))
            for a in self.list_atoms(limit=1_000_000)
            if a.id != atom_id
        ]
        scored = [(aid, n) for aid, n in scored if n >= min_shared]
        return sorted(scored, key=lambda t: (-t[1], t[0]))[:limit]

    def delete_atom(self, atom_id: str) -> None:
        """DETACH DELETE: removes the node and all its edges. Facts are
        never pruned by consolidation, but an explicit forget() removes."""
        with self._lock:
            self.conn.execute(
                "MATCH (a:Atom {id: $id}) DETACH DELETE a", {"id": atom_id}
            )

    def strengthen_edge(self, a_id: str, b_id: str, amount: float = 0.1) -> None:
        """Hebbian: use strengthens the edge, capped. Matches edges in
        EITHER direction between a and b (previously a->b only — a bug).
        Kuzu 0.11.3 crashes internally (KU_UNREACHABLE) on SET through an
        undirected relationship pattern, so each direction is a separate
        directed query instead of one `-[l:LINK]-` pattern."""
        with self._lock:
            for src, dst in ((a_id, b_id), (b_id, a_id)):
                self.conn.execute(
                    """MATCH (a:Atom {id: $src})-[l:LINK]->(b:Atom {id: $dst})
                       SET l.weight = l.weight + $amt * (1 - l.weight / 10)""",
                    {"src": src, "dst": dst, "amt": amount},
                )

    def decay_edges(self, factor: float = 0.98, prune_below: float = 0.1) -> int:
        """Hebbian decay. Prunes weak EDGES — never facts."""
        with self._lock:
            self.conn.execute(
                "MATCH ()-[l:LINK]->() SET l.weight = l.weight * $f",
                {"f": factor},
            )
            result = self.conn.execute(
                "MATCH ()-[l:LINK]->() WHERE l.weight < $floor "
                "DELETE l RETURN count(*)",
                {"floor": prune_below},
            )
            row = result.get_next()
        return int(row[0]) if row else 0

    def close(self) -> None:
        with self._lock:
            self.conn.close()
