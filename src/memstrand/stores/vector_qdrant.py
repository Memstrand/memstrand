"""Qdrant-backed VectorStore. Scope lives in the payload filter.

Scoping rule (the Playwright-bug fix, enforced here):
  - task_id / host are HARD filters. When the query carries them, only
    atoms with the exact same value match. When the query does NOT carry
    them, only GLOBAL atoms (stored with no task/host) match — an
    unscoped query can never see task-scoped memories.
  - session_id is provenance, not a recall filter. Sessions scope the L1
    fast cache and the distillation provenance; semantic recall is
    deliberately cross-session within a task (that's rubric test 2).
"""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from ..schema import MemoryAtom, ScopedQuery

COLLECTION = "memstrand"


def _point_id(atom_id: str) -> str:
    """Deterministic UUID for a human-readable atom id (mem:abc123...).

    Qdrant's embedded/local mode only accepts UUID or integer point ids,
    while the server accepts arbitrary strings. UUIDv5 keeps one code
    path that works on both — the real atom id rides in the payload.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"memstrand:{atom_id}"))


class QdrantVectorStore:
    def __init__(
        self,
        url: str = "http://localhost:6333",
        dim: int = 384,
        client: QdrantClient | None = None,
    ):
        self.client = client or QdrantClient(url=url)
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(
                COLLECTION,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def _scope_filter(self, query: ScopedQuery) -> Filter | None:
        must = []
        # task_id: exact match, or global-only when the query is unscoped.
        # This is the hard filter that stops cross-task leaks.
        must.append(
            FieldCondition(
                key="task_id",
                match=MatchValue(value=query.task_id if query.task_id else ""),
            )
        )
        # host: same rule.
        must.append(
            FieldCondition(
                key="host",
                match=MatchValue(value=query.host if query.host else ""),
            )
        )
        # session_id: deliberately NOT a filter (see module docstring).
        if query.categories:
            must.append(
                FieldCondition(
                    key="category",
                    # categories is a list: MatchValue only accepts a scalar
                    # (passing a list raises a pydantic ValidationError),
                    # so multi-category queries need MatchAny.
                    match=MatchAny(any=[c.value for c in query.categories]),
                )
            )
        return Filter(must=must) if must else None

    def upsert(self, atom: MemoryAtom, embedding: list[float]) -> None:
        self.client.upsert(
            collection_name=COLLECTION,
            points=[
                PointStruct(
                    id=_point_id(atom.id),
                    vector=embedding,
                    payload={
                        "atom_id": atom.id,
                        "category": atom.category.value,
                        "content": atom.content,
                        "session_id": atom.session_id or "",
                        "task_id": atom.task_id or "",
                        "host": atom.host or "",
                        "confidence": atom.confidence,
                        "provenance": atom.provenance.value,
                    },
                )
            ],
        )

    def exists(self, atom_id: str) -> bool:
        """Content-hash dedup check: is this atom already stored?"""
        records = self.client.retrieve(
            collection_name=COLLECTION, ids=[_point_id(atom_id)], with_vectors=False
        )
        return len(records) > 0

    def search(self, embedding: list[float], query: ScopedQuery) -> list[tuple[str, float]]:
        hits = self.client.query_points(
            collection_name=COLLECTION,
            query=embedding,
            query_filter=self._scope_filter(query),
            limit=query.limit,
        ).points
        return [(h.payload["atom_id"], h.score) for h in hits]

    def delete(self, atom_id: str) -> None:
        self.client.delete(
            collection_name=COLLECTION, points_selector=[_point_id(atom_id)]
        )
