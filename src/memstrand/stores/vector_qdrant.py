"""Qdrant-backed VectorStore. Scope lives in the payload filter.

Scoping rule (the Playwright-bug fix, generalized for globals):
  - scopes / host are HARD filters. Each point carries its atom's scope
    SET as a list payload (the global scope is the member ""); a query
    sees a point if any of its scopes is in {*query.scopes, ""}. An
    UNSCOPED query only sees globals — it can never see task-scoped
    memories. Pre-v2 points carry a single `task_id` string instead and
    are matched the same way until migration rewrites them. Same
    global-or-mine rule for host.
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

import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

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

from ..config import ensure_private_dir
from ..schema import MemoryAtom, ScopedQuery

COLLECTION = "memstrand"


class EmbeddingDimensionMismatch(ValueError):
    """The embedder's output dimension doesn't match this vector store's
    configured dimension. Almost always means `MEMSTRAND_EMBED_MODEL` (or
    the embedding backend) changed since the collection was created — a
    collision an opaque Qdrant "vector dimension" / numpy shape error
    wouldn't explain on its own, so every raise site here names the
    expected vs. actual dims and the configured embed backend/model, plus
    the fix."""


def _dim_mismatch_message(
    expected: int, actual: int, embed_backend: str | None, embed_model: str | None
) -> str:
    backend_bit = (
        f" (configured embed backend={embed_backend!r}, embed_model={embed_model!r})"
        if embed_backend or embed_model
        else ""
    )
    return (
        f"embedding dim mismatch: collection '{COLLECTION}' expects dim={expected}, "
        f"but got dim={actual}{backend_bit} — embedding model changed. Fix: restore "
        "the previous MEMSTRAND_EMBED_MODEL, or run `memstrand export` then import "
        "into a fresh MEMSTRAND_HOME."
    )


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
        dim: int | None = None,
        client: QdrantClient | None = None,
        embed_backend: str | None = None,
        embed_model: str | None = None,
    ):
        # Only used to enrich EmbeddingDimensionMismatch messages; this
        # class never uses them to pick a backend itself.
        self.embed_backend = embed_backend
        self.embed_model = embed_model
        # fix #1 (regression): the CLIENT is opened eagerly right here --
        # for an embedded on-disk store (`path=`), this is what takes
        # Qdrant's single-writer lock, and it needs no embedding at all.
        # Collection creation/dim validation (`_ensure_collection` below)
        # is deliberately deferred to the first call that actually knows a
        # real embedding dimension (an `upsert`, or an explicit `dim=` an
        # already-resolved caller passes in here) -- NOT bundled back into
        # `__init__` -- so `Memory()`/`memstrand serve` can still start up
        # with the embedding backend down, while the lock itself is taken
        # immediately instead of on first use (see Memory.__init__, which
        # wraps the raw "already accessed by another instance" RuntimeError
        # this raises into the friendlier `StoreLocked`).
        self._local_path: Path | None = None
        if client is not None:
            self.client = client
        elif url:
            self.client = QdrantClient(url=url)
        elif path:
            # fix #6: the directory Qdrant is about to create its `.lock`
            # file and (later) its collection tree in must already be
            # private BEFORE it creates anything there, not chmod'd after
            # the fact.
            self._local_path = Path(path)
            ensure_private_dir(self._local_path)
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
        self._collection_ready = False
        self._harden_local_storage()
        if dim is not None:
            self._ensure_collection(dim)

    def _harden_local_storage(self) -> None:
        """fix #6: tighten permissions on everything Qdrant has created
        under `self._local_path` so far -- dirs 0700, files 0600. Called
        right after the client opens (covers the `.lock` file) and again
        after collection creation (covers `collection/<name>/storage.sqlite`
        and friends). No-op for a remote (`url=`) or in-memory client, or
        before anything exists on disk yet."""
        if self._local_path is None or not self._local_path.exists():
            return
        for root, dirs, files in os.walk(self._local_path):
            for d in dirs:
                try:
                    os.chmod(os.path.join(root, d), 0o700)
                except OSError:
                    pass
            for f in files:
                try:
                    os.chmod(os.path.join(root, f), 0o600)
                except OSError:
                    pass
        try:
            self._local_path.chmod(0o700)
        except OSError:
            pass

    def _collection_known(self) -> bool:
        """True if the collection already exists -- either because this
        instance created it (`_ensure_collection`), or because an earlier
        instance/process already populated this same store -- WITHOUT
        creating one. Lets the read-only methods (`exists`/`search`/
        `delete`/`count`/`similar_to`) behave sanely (empty, never an
        error) before anything has ever been embedded into a fresh store
        (fix #1: `__init__` opens the client eagerly but no longer creates
        the collection, so a brand new `Memory()` must not crash the first
        `recall()`/`improve()` with `AttributeError`/`NoneType` the way
        the old fully-lazy `self.vectors = None` regression did)."""
        if self._collection_ready:
            return True
        if self.client.collection_exists(COLLECTION):
            self.dim = self.client.get_collection(COLLECTION).config.params.vectors.size
            self._collection_ready = True
            return True
        return False

    def _ensure_collection(self, dim: int) -> None:
        """Idempotent collection bootstrap: creates the collection (sized
        to `dim`) the first time a real embedding dimension is known, or
        validates `dim` against an already-created/pre-existing collection
        on every later call. Deliberately NOT called from `__init__`
        unless the caller already resolved a `dim=` itself -- see the
        class docstring note in `__init__`."""
        if self._collection_known():
            if dim != self.dim:
                raise EmbeddingDimensionMismatch(
                    _dim_mismatch_message(self.dim, dim, self.embed_backend, self.embed_model)
                )
            return
        self.client.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        if self._is_remote:
            self._ensure_payload_indexes()
        self.dim = dim
        self._collection_ready = True
        self._harden_local_storage()

    def _ensure_payload_indexes(self) -> None:
        """Server-side filtering is faster with indexes. Only called for a
        real Qdrant server (see `_is_remote` above) — best-effort regardless,
        since index creation isn't essential to correctness."""
        for key, schema in (
            ("scopes", PayloadSchemaType.KEYWORD),
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
        visible = [*query.scopes, ""]
        must = [
            Filter(
                should=[
                    FieldCondition(key="scopes", match=MatchAny(any=visible)),
                    # pre-v2 points (single `task_id`), until migrated
                    FieldCondition(key="task_id", match=MatchAny(any=visible)),
                ]
            ),
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

    def _check_dim(self, embedding: list[float]) -> None:
        """Proactively catch a dim mismatch before handing it to Qdrant:
        Qdrant's own "Vector dimension error" (or a numpy shape error, if
        it ever got that far) names neither the expected/actual dims nor
        the embed backend in a way a human can act on. Checking here means
        we only ever raise the enriched `EmbeddingDimensionMismatch`."""
        if len(embedding) != self.dim:
            raise EmbeddingDimensionMismatch(
                _dim_mismatch_message(self.dim, len(embedding), self.embed_backend, self.embed_model)
            )

    def upsert(self, atom: MemoryAtom, embedding: list[float]) -> None:
        # The ONE place the collection gets created: a real embedding is
        # the first moment a dim is actually known (see `__init__`/
        # `_ensure_collection`). Idempotent and cheap once the collection
        # already exists.
        self._ensure_collection(len(embedding))
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
                        "scopes": list(atom.scopes),
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
        """Content-hash dedup check: is this atom already stored? `False`
        (never an error) if the collection doesn't exist yet -- a fresh
        store that's never had anything upserted into it (fix #1)."""
        if not self._collection_known():
            return False
        records = self.client.retrieve(
            collection_name=COLLECTION, ids=[_point_id(atom_id)], with_vectors=False
        )
        return len(records) > 0

    def get_vector(self, atom_id: str) -> list[float] | None:
        """The stored embedding for `atom_id`, or None. Lets callers re-key
        or re-scope a point (migration, `Memory.set_scopes`) without
        re-embedding its content."""
        if not self._collection_known():
            return None
        records = self.client.retrieve(
            collection_name=COLLECTION, ids=[_point_id(atom_id)], with_vectors=True
        )
        if not records or records[0].vector is None:
            return None
        return list(records[0].vector)

    def search(
        self,
        embedding: list[float],
        query: ScopedQuery,
        now: datetime | None = None,
        limit: int | None = None,
    ) -> list[tuple[str, float]]:
        """`[]` (never an error) if the collection doesn't exist yet --
        nothing has ever been embedded into this store (fix #1)."""
        if not self._collection_known():
            return []
        self._check_dim(embedding)
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
        Used by consolidation to find near-duplicates without re-embedding.
        `[]` (never an error) if the collection doesn't exist yet."""
        if not self._collection_known():
            return []
        hits = self.client.query_points(
            collection_name=COLLECTION,
            query=_point_id(atom_id),
            query_filter=self._scope_filter(query, now=time.time()),
            limit=limit + 1,  # the atom itself is always the top hit
        ).points
        return [(h.payload["atom_id"], h.score) for h in hits if h.payload["atom_id"] != atom_id][:limit]

    def delete(self, atom_id: str) -> None:
        """No-op if the collection doesn't exist yet -- nothing to delete."""
        if not self._collection_known():
            return
        self.client.delete(collection_name=COLLECTION, points_selector=[_point_id(atom_id)])

    def count(self) -> int:
        """Point count in the collection. `0` (never an error) if the
        collection doesn't exist yet."""
        if not self._collection_known():
            return 0
        return self.client.count(collection_name=COLLECTION, exact=True).count

    def close(self) -> None:
        self.client.close()
