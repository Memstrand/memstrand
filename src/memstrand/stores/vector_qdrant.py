"""Qdrant-backed VectorStore. Scope lives in the payload filter.

Scoping rule (the Playwright-bug fix, generalized for globals):
  - task_id / host are HARD filters. A scoped query sees its own scope
    PLUS global atoms (task_id == "" / host == ""): task_id in
    {query.task_id, ""}. An UNSCOPED query only sees globals — it can
    never see task-scoped memories. Same rule for host.
  - session_id is provenance, not a recall filter. Sessions scope the L1
    fast cache and distillation provenance; semantic recall is
    deliberately cross-session within a scope.
  - TTL: `expires_at` is stored as an epoch float (0.0 = never expires)
    and filtered with a should-clause so expired atoms never come back.

No Docker required: pass `path=` for an embedded on-disk Qdrant (the
default when `MEMSTRAND_QDRANT_URL` is unset), `url=` for a real server,
or `client=` to inject one directly (tests use `QdrantClient(":memory:")`).
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PayloadSchemaType,
    PointStruct,
    Range,
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
        url: str | None = None,
        path: str | None = None,
        dim: int = 384,
        client: QdrantClient | None = None,
    ):
        if client is not None:
            self.client = client
        elif url:
            self.client = QdrantClient(url=url)
        elif path:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(":memory:")
        # Payload indexes only matter (and only work) against a real Qdrant
        # server. Embedded/local mode (`path=` or `:memory:`, including any
        # injected client that isn't known to be remote) scans every point
        # regardless, and attempting to create one there is a guaranteed
        # no-op that also prints `UserWarning: Payload indexes have no
        # effect in the local Qdrant` on every single startup.
        self._is_remote = bool(url)
        self.dim = dim
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(
                COLLECTION,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            if self._is_remote:
                self._ensure_payload_indexes()
        else:
            existing_dim = self.client.get_collection(COLLECTION).config.params.vectors.size
            if existing_dim != dim:
                raise ValueError(
                    f"embedding dim mismatch: collection '{COLLECTION}' was created with "
                    f"dim={existing_dim}, but the configured embedder produces dim={dim} "
                    "— embedding model changed; run `memstrand export` then re-import "
                    "into a fresh MEMSTRAND_HOME."
                )

    def _ensure_payload_indexes(self) -> None:
        """Server-side filtering is faster with indexes. Only called for a
        real Qdrant server (see `_is_remote` above) — best-effort regardless,
        since index creation isn't essential to correctness."""
        for key, schema in (
            ("task_id", PayloadSchemaType.KEYWORD),
            ("host", PayloadSchemaType.KEYWORD),
            ("category", PayloadSchemaType.KEYWORD),
            ("expires_at", PayloadSchemaType.FLOAT),
        ):
            try:
                self.client.create_payload_index(COLLECTION, field_name=key, field_schema=schema)
            except Exception:  # noqa: BLE001, S110 - best-effort only
                pass

    def _scope_filter(self, query: ScopedQuery, now: float | None = None) -> Filter:
        must = [
            FieldCondition(key="task_id", match=MatchAny(any=[query.task_id or "", ""])),
            FieldCondition(key="host", match=MatchAny(any=[query.host or "", ""])),
        ]
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
        now = now if now is not None else time.time()
        must.append(
            Filter(
                should=[
                    FieldCondition(key="expires_at", range=Range(lte=0.0)),
                    FieldCondition(key="expires_at", range=Range(gt=now)),
                ]
            )
        )
        return Filter(must=must)

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
                        "source": atom.source or "",
                        "created_at": atom.created_at.timestamp(),
                        "expires_at": atom.expires_at.timestamp() if atom.expires_at else 0.0,
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

    def search(
        self,
        embedding: list[float],
        query: ScopedQuery,
        now: datetime | None = None,
        limit: int | None = None,
    ) -> list[tuple[str, float]]:
        now_ts = (now or datetime.now(UTC)).timestamp()
        hits = self.client.query_points(
            collection_name=COLLECTION,
            query=embedding,
            query_filter=self._scope_filter(query, now=now_ts),
            limit=limit if limit is not None else query.limit,
        ).points
        return [(h.payload["atom_id"], h.score) for h in hits]

    def similar_to(self, atom_id: str, query: ScopedQuery, limit: int = 5) -> list[tuple[str, float]]:
        """Points most similar to an already-stored point, scope-filtered.
        Used by consolidation to find near-duplicates without re-embedding."""
        hits = self.client.query_points(
            collection_name=COLLECTION,
            query=_point_id(atom_id),
            query_filter=self._scope_filter(query, now=time.time()),
            limit=limit + 1,  # the atom itself is always the top hit
        ).points
        return [(h.payload["atom_id"], h.score) for h in hits if h.payload["atom_id"] != atom_id][:limit]

    def delete(self, atom_id: str) -> None:
        self.client.delete(collection_name=COLLECTION, points_selector=[_point_id(atom_id)])

    def close(self) -> None:
        self.client.close()
