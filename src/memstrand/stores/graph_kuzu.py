"""Kuzu-backed GraphStore. Embedded, zero infrastructure.

Pinned to kuzu==0.11.3 (upstream archived Oct 2025). All access goes
through the GraphStore protocol so a future fork can be swapped in.
"""

from __future__ import annotations

from datetime import UTC, datetime

import kuzu

from ..schema import Category, MemoryAtom, MemoryEdge, Provenance, RelationType


def _row_to_atom(row: tuple) -> MemoryAtom:
    (aid, cat, content, prov, conf, sess, task, host, created, expires) = row
    return MemoryAtom(
        id=aid,
        category=Category(cat),
        content=content,
        provenance=Provenance(prov),
        confidence=conf,
        session_id=sess or None,
        task_id=task or None,
        host=host or None,
        created_at=datetime.fromisoformat(created) if created else datetime.now(UTC),
        expires_at=datetime.fromisoformat(expires) if expires else None,
    )


def _row_to_edge(row: tuple) -> MemoryEdge:
    (src, dst, rel, conf, reason, weight) = row
    return MemoryEdge(
        source_id=src,
        target_id=dst,
        relation=RelationType(rel),
        confidence=conf,
        reason=reason or "",
        weight=weight,
    )


class KuzuGraphStore:
    def __init__(self, path: str = "./kuzu_data"):
        self.db = kuzu.Database(path)
        self.conn = kuzu.Connection(self.db)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """CREATE NODE TABLE IF NOT EXISTS Atom(
                id STRING, category STRING, content STRING,
                provenance STRING, confidence DOUBLE,
                session_id STRING, task_id STRING, host STRING,
                created_at STRING, expires_at STRING,
                PRIMARY KEY (id))"""
        )
        self.conn.execute(
            """CREATE REL TABLE IF NOT EXISTS LINK(
                FROM Atom TO Atom,
                relation STRING, confidence DOUBLE,
                reason STRING, weight DOUBLE)"""
        )

    def add_atom(self, atom: MemoryAtom) -> None:
        self.conn.execute(
            """MERGE (a:Atom {id: $id})
               SET a.category=$cat, a.content=$content,
                   a.provenance=$prov, a.confidence=$conf,
                   a.session_id=$sess, a.task_id=$task, a.host=$host,
                   a.created_at=$created,
                   a.expires_at=$expires""",
            {
                "id": atom.id, "cat": atom.category.value,
                "content": atom.content, "prov": atom.provenance.value,
                "conf": atom.confidence,
                "sess": atom.session_id or "", "task": atom.task_id or "",
                "host": atom.host or "",
                "created": atom.created_at.isoformat(),
                "expires": atom.expires_at.isoformat() if atom.expires_at else "",
            },
        )

    def add_edge(self, edge: MemoryEdge) -> None:
        self.conn.execute(
            """MATCH (a:Atom {id: $src}), (b:Atom {id: $dst})
               CREATE (a)-[l:LINK {relation: $rel, confidence: $conf,
                   reason: $reason, weight: $w}]->(b)""",
            {
                "src": edge.source_id, "dst": edge.target_id,
                "rel": edge.relation.value, "conf": edge.confidence,
                "reason": edge.reason, "w": edge.weight,
            },
        )

    def get_atom(self, atom_id: str) -> MemoryAtom | None:
        """Row -> MemoryAtom hydration. Vector store is still the
        primary read path; this is the fallback for graph-only reads."""
        rows = self.conn.execute(
            """MATCH (a:Atom {id: $id})
               RETURN a.id, a.category, a.content, a.provenance,
                      a.confidence, a.session_id, a.task_id, a.host,
                      a.created_at, a.expires_at""",
            {"id": atom_id},
        ).get_all()
        return _row_to_atom(rows[0]) if rows else None

    def neighbors(
        self, atom_id: str, relation: str | None = None, depth: int = 1
    ) -> list[tuple[MemoryAtom, MemoryEdge]]:
        """Depth-1 neighbor hydration, both directions. relation filters
        to one edge type (e.g. 'contradicts'). Deeper traversal is a BFS
        over this primitive — depth>1 expands one hop at a time."""
        rel_filter = "AND l.relation = $rel" if relation else ""
        params: dict = {"id": atom_id}
        if relation:
            params["rel"] = relation
        rows = self.conn.execute(
            f"""MATCH (a:Atom {{id: $id}})-[l:LINK]-(b:Atom)
                WHERE b.id <> $id {rel_filter}
                RETURN b.id AS bid, b.category, b.content, b.provenance,
                       b.confidence, b.session_id, b.task_id, b.host,
                       b.created_at, b.expires_at,
                       a.id AS src, b.id AS dst, l.relation, l.confidence, l.reason, l.weight""",
            params,
        ).get_all()
        out = []
        for r in rows:
            out.append((_row_to_atom(r[:10]), _row_to_edge(r[10:])))
        if depth > 1:
            seen = {atom_id} | {a.id for a, _ in out}
            frontier = [a.id for a, _ in out]
            for _ in range(depth - 1):
                nxt = []
                for fid in frontier:
                    for na, ne in self.neighbors(fid, relation=relation, depth=1):
                        if na.id not in seen:
                            seen.add(na.id)
                            out.append((na, ne))
                            nxt.append(na.id)
                frontier = nxt
        return out

    def delete_atom(self, atom_id: str) -> None:
        """DETACH DELETE: removes the node and all its edges. Facts are
        never pruned by consolidation, but an explicit forget() removes."""
        self.conn.execute(
            "MATCH (a:Atom {id: $id}) DETACH DELETE a", {"id": atom_id}
        )

    def contradicting(self, atom_id: str) -> list[tuple[MemoryAtom, MemoryEdge]]:
        return self.neighbors(atom_id, relation="contradicts")

    def strengthen_edge(self, source_id: str, target_id: str, amount: float = 0.1) -> None:
        """Hebbian: use strengthens the edge, capped."""
        self.conn.execute(
            """MATCH (a:Atom {id: $src})-[l:LINK]->(b:Atom {id: $dst})
               SET l.weight = l.weight + $amt * (1 - l.weight / 10)""",
            {"src": source_id, "dst": target_id, "amt": amount},
        )

    def decay_edges(self, factor: float = 0.98, prune_below: float = 0.1) -> int:
        """Hebbian decay. Prunes weak EDGES — never facts."""
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
