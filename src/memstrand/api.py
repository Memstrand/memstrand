"""memstrand public API: remember / recall / improve / forget / context.

Thread-safe by construction: one `threading.RLock` wraps every public
method, so a daemon can share a single `Memory` across several agents.
`Memory` is also a context manager — `with Memory(...) as mem:` closes
every backing store on exit.
"""

from __future__ import annotations

import hashlib
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Self

from . import export as export_module
from . import ranking
from .config import Settings, get_settings
from .embeddings import Reranker, make_embedder
from .pipeline.consolidate import record_review
from .pipeline.extract import LLMUnavailable, chat_completion
from .schema import (
    Category,
    MemoryAtom,
    MemoryEdge,
    Provenance,
    RelationType,
    ScopedQuery,
)
from .stores.base import GraphStore, KeywordIndex, SessionStore, VectorStore
from .stores.graph_sqlite import SqliteGraphStore
from .stores.keyword_sqlite import SqliteKeywordIndex
from .stores.sqlite import SqliteSessionStore
from .stores.vector_qdrant import QdrantVectorStore

_RECALL_RECENCY_HALF_LIFE_DAYS = 90.0

# Sentinel default for `Memory.forget_source(task_id=..., host=...)`: distinct
# from `None` (which means "global scope"), so the unscoped (all-scopes)
# default used by the CLI/REST "wipe this source everywhere" action can be
# told apart from an explicit "only the global scope" request.
_ALL_SCOPES = object()


class Memory:
    """Local-first typed memory. All LLM calls go to a local endpoint."""

    def __init__(
        self,
        home: str | Path | None = None,
        qdrant_url: str | None = None,
        kuzu_path: str | None = None,
        sqlite_path: str | None = None,
        llm_base_url: str | None = None,
        llm_model: str | None = None,
        # dependency injection: real backends by default, swappable in tests
        embedder=None,
        vector_store: VectorStore | None = None,
        graph_store: GraphStore | None = None,
        session_store: SessionStore | None = None,
        keyword_index: KeywordIndex | None = None,
        reranker=None,
        settings: Settings | None = None,
    ):
        self._lock = threading.RLock()
        self._improve_lock = threading.Lock()
        self.settings = settings or get_settings()
        self.home = Path(home).expanduser() if home is not None else self.settings.home

        self.llm_base_url = llm_base_url or self.settings.llm_base_url
        self.llm_model = llm_model or self.settings.llm_model
        self.llm_api_key = self.settings.llm_api_key

        self.embedder = embedder or make_embedder(self.settings)

        if vector_store is not None:
            self.vectors = vector_store
        else:
            dim = len(self._embed("dim probe"))
            url = qdrant_url if qdrant_url is not None else self.settings.qdrant_url
            if url:
                self.vectors = QdrantVectorStore(url=url, dim=dim)
            else:
                self.vectors = QdrantVectorStore(path=str(self.home / "qdrant"), dim=dim)

        default_db_path = sqlite_path or str(self.home / "memstrand.db")

        if graph_store is not None:
            self.graph = graph_store
        elif self.settings.graph_backend == "kuzu":
            from .stores.graph_kuzu import KuzuGraphStore

            self.graph = KuzuGraphStore(path=kuzu_path or str(self.home / "kuzu"))
        else:
            self.graph = SqliteGraphStore(path=default_db_path)

        self.sessions = session_store or SqliteSessionStore(path=default_db_path)

        if keyword_index is not None:
            self.keyword = keyword_index
        else:
            # Co-locate with an injected session store's DB file when
            # possible (test fixtures inject stores but rarely a keyword
            # index); fall back to a private in-memory index otherwise.
            sibling_path = getattr(self.sessions, "path", None)
            self.keyword = SqliteKeywordIndex(sibling_path or ":memory:")

        self.reranker = reranker
        if self.reranker is None and self.settings.rerank_model:
            self.reranker = Reranker(self.settings.rerank_model)

    @classmethod
    def in_memory(cls, **kwargs) -> Memory:
        """Zero-dependency memory for demos and tests: Qdrant runs in
        `:memory:` mode, the graph/session/keyword stores live in a temp
        SQLite file. `pip install memstrand` is the entire setup — no
        Docker, no server, no LLM.

        Data is ephemeral (it dies with the process). For persistence,
        use the default constructor (on-disk Qdrant + SQLite under
        `MEMSTRAND_HOME`), with an optional local LLM endpoint for the
        `improve()` distillation pass.
        """
        import tempfile

        from qdrant_client import QdrantClient

        tmp = Path(tempfile.mkdtemp(prefix="memstrand-"))
        db_path = str(tmp / "memstrand.db")
        kwargs.setdefault("home", tmp)
        kwargs.setdefault("vector_store", QdrantVectorStore(client=QdrantClient(":memory:")))
        kwargs.setdefault("graph_store", SqliteGraphStore(path=db_path))
        kwargs.setdefault("session_store", SqliteSessionStore(path=db_path))
        return cls(**kwargs)

    # -- internal helpers ---------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        return next(iter(self.embedder.embed([text])))

    @staticmethod
    def _atom_id(content: str, task_id: str | None = None, host: str | None = None) -> str:
        """Scope-aware content hash: the same text in two different
        task/host scopes gets two different ids, so one doesn't shadow
        the other. Re-remembering the same (content, scope) is a no-op id."""
        key = f"{content}\x1f{task_id or ''}\x1f{host or ''}"
        return f"mem:{hashlib.sha256(key.encode()).hexdigest()[:16]}"

    @staticmethod
    def _visible(atom: MemoryAtom, task_id: str | None, host: str | None, now: datetime) -> bool:
        """THE single scope-visibility rule, used everywhere a graph-hydrated
        atom (neighbor, contradiction, standing instruction, ...) needs to be
        checked against a query scope: visible if its task_id/host is that
        exact scope OR global (None), and it isn't expired. Mirrors the hard
        filter `VectorStore.search`/`KeywordIndex.search` already apply to
        primary recall hits — this is what makes it apply to everything
        *reached through* a hit (graph neighbors) too, closing the
        global-atom-as-bridge leak between scopes."""
        if atom.task_id is not None and atom.task_id != task_id:
            return False
        if atom.host is not None and atom.host != host:
            return False
        return atom.expires_at is None or atom.expires_at > now

    def _iter_all_atoms(self):
        offset = 0
        page = 500
        while True:
            batch = self.graph.list_atoms(limit=page, offset=offset)
            if not batch:
                return
            yield from batch
            offset += len(batch)
            if len(batch) < page:
                return

    # -- write path -----------------------------------------------------------

    def remember(
        self,
        content: str,
        category: Category = Category.fact,
        provenance: Provenance = Provenance.explicit_statement,
        confidence: float = 0.8,
        session_id: str | None = None,
        task_id: str | None = None,
        host: str | None = None,
        ttl_days: float | None = None,
        source: str | None = None,
        source_ref: str | None = None,
        agent: str | None = None,
    ) -> MemoryAtom:
        with self._lock:
            atom_id = self._atom_id(content, task_id, host)
            existing = self.graph.get_atom(atom_id)
            now = datetime.now(UTC)
            atom = MemoryAtom(
                id=atom_id,
                category=category,
                content=content,
                provenance=provenance,
                confidence=confidence,
                session_id=session_id,
                task_id=task_id,
                host=host,
                agent=agent,
                source=source,
                source_ref=source_ref,
                expires_at=(now + timedelta(days=ttl_days)) if ttl_days else None,
                created_at=existing.created_at if existing else now,
                updated_at=now,
                access_count=existing.access_count if existing else 0,
                last_accessed_at=existing.last_accessed_at if existing else None,
                fsrs_difficulty=existing.fsrs_difficulty if existing else 0.5,
                fsrs_stability_days=existing.fsrs_stability_days if existing else 30.0,
                fsrs_next_review=existing.fsrs_next_review if existing else None,
            )
            self.vectors.upsert(atom, self._embed(content))
            self.graph.add_atom(atom)
            self.keyword.add(atom)
            self.sessions.log_provenance(atom.id, "remember", provenance.value)
            return atom

    def log_turn(
        self, session_id: str, role: str, content: str, task_id: str | None = None
    ) -> None:
        with self._lock:
            self.sessions.append_turn(session_id, role, content, task_id=task_id)

    def forget(self, atom_id: str) -> bool:
        """Delete with provenance-chain cleanup. Returns False if the
        atom never existed (never silently pretends to have deleted it)."""
        with self._lock:
            if self.graph.get_atom(atom_id) is None:
                return False
            self.vectors.delete(atom_id)
            self.keyword.delete(atom_id)
            self.graph.delete_atom(atom_id)
            self.sessions.log_provenance(atom_id, "forget", "explicit user delete")
            return True

    def forget_source(
        self,
        source: str,
        task_id: str | None = _ALL_SCOPES,
        host: str | None = _ALL_SCOPES,
    ) -> int:
        """Delete every atom ingested from `source`. Returns the count removed.

        By default (`task_id`/`host` omitted) this matches every scope —
        the unscoped "wipe this source everywhere" behavior the CLI and
        REST surface use. Passing `task_id`/`host` narrows the match to
        atoms in EXACTLY that scope (equality, not the recall `_visible`
        OR-global rule — this is a deletion, not a read), so re-ingesting
        a changed file under one task never deletes another task's (or
        the global scope's) copy of the same source path. `ingest.py`
        always passes its own `task_id` for this reason.
        """
        with self._lock:
            ids = self.graph.ids_by_source(source)
            if task_id is not _ALL_SCOPES or host is not _ALL_SCOPES:
                atoms = self.graph.get_atoms(ids)
                ids = [
                    aid
                    for aid in ids
                    if aid in atoms
                    and (task_id is _ALL_SCOPES or atoms[aid].task_id == task_id)
                    and (host is _ALL_SCOPES or atoms[aid].host == host)
                ]
            for atom_id in ids:
                self.vectors.delete(atom_id)
                self.keyword.delete(atom_id)
                self.graph.delete_atom(atom_id)
                self.sessions.log_provenance(atom_id, "forget", f"source removed: {source}")
            return len(ids)

    def update(
        self,
        atom_id: str,
        content: str | None = None,
        category: Category | None = None,
        confidence: float | None = None,
    ) -> MemoryAtom:
        """An edit is a correction: if the content actually changes, a
        NEW atom is created (provenance=corrected) and linked back to the
        old one by a `supersedes` edge — the old atom stays queryable. If
        only category/confidence change (content identical), the atom is
        updated in place instead, since a new id would hash identically
        anyway."""
        with self._lock:
            old = self.graph.get_atom(atom_id)
            if old is None:
                raise KeyError(f"no such atom: {atom_id}")

            if content is None or content == old.content:
                now = datetime.now(UTC)
                updated = old.model_copy(
                    update={
                        "category": category if category is not None else old.category,
                        "confidence": confidence if confidence is not None else old.confidence,
                        "updated_at": now,
                    }
                )
                self.vectors.upsert(updated, self._embed(updated.content))
                self.graph.add_atom(updated)
                self.keyword.add(updated)
                self.sessions.log_provenance(updated.id, "update", "in-place edit")
                return updated

            new_atom = self.remember(
                content,
                category=category if category is not None else old.category,
                provenance=Provenance.corrected,
                confidence=confidence if confidence is not None else old.confidence,
                session_id=old.session_id,
                task_id=old.task_id,
                host=old.host,
                source=old.source,
                source_ref=old.source_ref,
                agent=old.agent,
            )
            self.graph.add_edge(
                MemoryEdge(
                    source_id=new_atom.id, target_id=old.id, relation=RelationType.supersedes,
                    reason="correction via update()",
                )
            )
            self.sessions.log_provenance(new_atom.id, "update", f"supersedes {old.id}")
            return new_atom

    # -- scoped read path -------------------------------------------------------

    def recall(
        self,
        query: str,
        session_id: str | None = None,
        task_id: str | None = None,
        host: str | None = None,
        categories: list[Category] | None = None,
        limit: int = 10,
        include_superseded: bool = False,
        track: bool = True,
        min_relevance: float | None = None,
    ) -> list[dict]:
        """Hybrid (dense + keyword) scoped recall. Scope is a hard filter:
        a scoped query sees its own scope PLUS globals; an unscoped query
        sees only globals — the Playwright-bug fix, generalized. Results
        are hydrated with graph neighbors; contradictions are surfaced,
        never hidden. `track=False` is for internal callers (judge,
        consolidation) that must not mutate access stats or edge weights.

        Relevance is an ABSOLUTE signal, not normalized within this result
        set — a lone candidate with no real connection to the query must
        score low, not 1.0 just because it's "the best of one". With a
        reranker configured, relevance is `sigmoid(rerank_score)`;
        otherwise it's `max(clipped cosine similarity, keyword strength)`
        plus a small bonus when both retrievers agree (see
        `ranking.absolute_relevance`). `min_relevance` drops hits below the
        threshold and defaults to `settings.min_relevance`
        (`MEMSTRAND_MIN_RELEVANCE`, 0.0 by default — tune per embedding
        model, see `Settings.min_relevance`)."""
        with self._lock:
            now = datetime.now(UTC)
            q = ScopedQuery(
                text=query, session_id=session_id, task_id=task_id,
                host=host, categories=categories, limit=limit,
            )
            candidate_limit = max(limit * 3, limit)
            dense_hits = self.vectors.search(self._embed(query), q, now=now, limit=candidate_limit)
            keyword_hits = self.keyword.search(query, q, limit=candidate_limit, now=now)

            dense_by_id = dict(dense_hits)
            keyword_by_id = dict(keyword_hits)
            keyword_ids = set(keyword_by_id)

            # Order-preserving union of both retrievers' hits. The old RRF
            # fusion score was only ever used to seed relevance below, which
            # is now computed directly from each retriever's own absolute
            # signal instead.
            candidate_ids = list(dict.fromkeys([aid for aid, _ in dense_hits] + [aid for aid, _ in keyword_hits]))
            if not candidate_ids:
                return []

            atoms = self.graph.get_atoms(candidate_ids)
            superseded = self.graph.superseded_ids(candidate_ids)

            working_ids = [
                aid
                for aid in candidate_ids
                if aid in atoms and (include_superseded or aid not in superseded)
            ]
            if not working_ids:
                return []

            if self.reranker is not None:
                docs = [atoms[aid].content for aid in working_ids]
                rerank_scores = self.reranker.rerank(query, docs)
                relevance = {aid: ranking.sigmoid(s) for aid, s in zip(working_ids, rerank_scores)}
            else:
                relevance = {
                    aid: ranking.absolute_relevance(
                        dense_score=dense_by_id.get(aid), keyword_score=keyword_by_id.get(aid)
                    )
                    for aid in working_ids
                }

            threshold = self.settings.min_relevance if min_relevance is None else min_relevance
            working_ids = [aid for aid in working_ids if relevance[aid] >= threshold]
            if not working_ids:
                return []

            scored: list[tuple[str, float, bool]] = []
            for aid in working_ids:
                atom = atoms[aid]
                last_touched = max(
                    (d for d in (atom.updated_at, atom.last_accessed_at) if d is not None),
                    default=atom.created_at,
                )
                recency = ranking.recency_weight(last_touched, now, _RECALL_RECENCY_HALF_LIFE_DAYS)
                score = ranking.blended_score(
                    relevance=relevance[aid],
                    confidence=atom.confidence,
                    recency=recency,
                    provenance=ranking.provenance_weight(atom.provenance),
                    access_count=atom.access_count,
                )
                is_superseded = aid in superseded
                if is_superseded:
                    score *= 0.5
                scored.append((aid, score, is_superseded))

            scored.sort(key=lambda t: t[1], reverse=True)
            scored = scored[:limit]
            top_ids = [aid for aid, _, _ in scored]

            # Neighbor hydration MUST re-apply the scope filter: an atom can
            # be a top hit because it's visible, but a global atom linked to
            # a task-scoped atom (or two different tasks both linked to the
            # same global atom) would otherwise leak that atom's content
            # into a query scope that never should have seen it — the
            # graph-neighbor bridge leak. Superseded neighbors are kept
            # (the edge itself is the useful signal, not a new disclosure).
            neighbor_map: dict[str, list[dict]] = {}
            for aid in top_ids:
                rows = []
                for n_atom, edge in self.graph.neighbors(aid, depth=1):
                    if not self._visible(n_atom, q.task_id, q.host, now):
                        continue
                    direction = "out" if edge.source_id == aid else "in"
                    rows.append(
                        {
                            "id": n_atom.id, "relation": edge.relation.value,
                            "content": n_atom.content, "weight": edge.weight,
                            "direction": direction,
                        }
                    )
                neighbor_map[aid] = rows

            results = []
            for aid, score, is_superseded in scored:
                atom = atoms[aid]
                neighbor_rows = neighbor_map[aid]
                results.append(
                    {
                        "id": aid,
                        "score": score,
                        "content": atom.content,
                        "category": atom.category.value,
                        "provenance": atom.provenance.value,
                        "confidence": atom.confidence,
                        "task_id": atom.task_id,
                        "host": atom.host,
                        "source": atom.source,
                        "source_ref": atom.source_ref,
                        "created_at": atom.created_at.isoformat(),
                        "similarity": dense_by_id.get(aid),
                        "keyword": aid in keyword_ids,
                        "superseded": is_superseded,
                        "neighbors": neighbor_rows,
                        "contradictions": [
                            n for n in neighbor_rows if n["relation"] == RelationType.contradicts.value
                        ],
                    }
                )

            if track and top_ids:
                self.graph.record_access(top_ids, now)
                for aid in top_ids:
                    atom = atoms[aid]
                    record_review(atom, success=True)
                    self.graph.update_fsrs(
                        aid, atom.fsrs_difficulty, atom.fsrs_stability_days, atom.fsrs_next_review
                    )
                    for n in neighbor_map[aid]:
                        self.graph.strengthen_edge(aid, n["id"])

            return results

    def get(
        self,
        atom_id: str,
        task_id: str | None = None,
        host: str | None = None,
        scoped: bool = False,
    ) -> dict | None:
        """Fetch one atom by id with its graph neighbors and provenance
        history. This is an OWNER-FACING browse surface (CLI `show`, the web
        UI drawer) — by default (`scoped=False`) it returns the atom and ALL
        its neighbors regardless of scope, because whoever can call `get()`
        directly already owns the whole store, same as `list()`. Pass
        `scoped=True` with `task_id`/`host` to instead apply the same
        `_visible` recall rule as `recall()`: returns `None` for an atom
        outside that scope, and drops out-of-scope neighbors — use this for
        any caller that is NOT the trusted owner (no MCP tool exposes this;
        don't add one that calls `get()` unscoped)."""
        with self._lock:
            now = datetime.now(UTC)
            atom = self.graph.get_atom(atom_id)
            if atom is None:
                return None
            if scoped and not self._visible(atom, task_id, host, now):
                return None
            superseded = atom_id in self.graph.superseded_ids([atom_id])
            d = self.atom_to_dict(atom, superseded=superseded)
            neighbor_rows = []
            for n_atom, edge in self.graph.neighbors(atom_id, depth=1):
                if scoped and not self._visible(n_atom, task_id, host, now):
                    continue
                direction = "out" if edge.source_id == atom_id else "in"
                neighbor_rows.append(
                    {
                        "id": n_atom.id, "relation": edge.relation.value,
                        "content": n_atom.content, "weight": edge.weight,
                        "direction": direction,
                    }
                )
            d["neighbors"] = neighbor_rows
            d["history"] = self.sessions.history(atom_id)
            return d

    def list(
        self,
        category: Category | None = None,
        task_id: str | None = None,
        host: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        with self._lock:
            cat_value = category.value if isinstance(category, Category) else category
            atoms = self.graph.list_atoms(
                category=cat_value, task_id=task_id, host=host, limit=limit, offset=offset
            )
            if not atoms:
                return []
            superseded = self.graph.superseded_ids([a.id for a in atoms])
            return [self.atom_to_dict(a, superseded=a.id in superseded) for a in atoms]

    def count(self) -> int:
        with self._lock:
            return self.graph.count_atoms()

    def stats(self) -> dict:
        with self._lock:
            by_category: dict[str, int] = {}
            by_task: dict[str, int] = {}
            for atom in self._iter_all_atoms():
                by_category[atom.category.value] = by_category.get(atom.category.value, 0) + 1
                key = atom.task_id or "global"
                by_task[key] = by_task.get(key, 0) + 1
            return {
                "atoms": self.graph.count_atoms(),
                "edges": len(self.graph.all_edges()),
                "by_category": by_category,
                "by_task": by_task,
                "sessions": len(self.sessions.list_sessions()),
                "undistilled_sessions": len(self.sessions.undistilled_sessions()),
            }

    @staticmethod
    def atom_to_dict(atom: MemoryAtom, superseded: bool = False) -> dict:
        return {
            "id": atom.id,
            "category": atom.category.value,
            "content": atom.content,
            "provenance": atom.provenance.value,
            "confidence": atom.confidence,
            "session_id": atom.session_id,
            "task_id": atom.task_id,
            "host": atom.host,
            "source": atom.source,
            "source_ref": atom.source_ref,
            "created_at": atom.created_at.isoformat(),
            "updated_at": atom.updated_at.isoformat(),
            "expires_at": atom.expires_at.isoformat() if atom.expires_at else None,
            "access_count": atom.access_count,
            "last_accessed_at": atom.last_accessed_at.isoformat() if atom.last_accessed_at else None,
            "superseded": superseded,
        }

    # -- context injection -------------------------------------------------------

    def _standing_instructions(
        self, task_id: str | None, host: str | None, now: datetime, cap: int
    ) -> list[dict]:
        """instruction + preference atoms visible in scope (globals + this
        task/host), non-expired, non-superseded, highest confidence first.
        Scope visibility goes through `_visible` — the one rule, same as
        `recall()`'s neighbor hydration and `get(scoped=True)` — so a
        host-scoped instruction never leaks to a query for a different
        host (or an unscoped one)."""
        scopes = {"", task_id} if task_id else {""}
        atoms: dict[str, MemoryAtom] = {}
        for category in (Category.instruction, Category.preference):
            for scope in scopes:
                # host is intentionally NOT passed to list_atoms here (it
                # would hard-filter instead of OR-with-global): `_visible`
                # below applies the real global-or-this-host rule.
                for atom in self.graph.list_atoms(category=category.value, task_id=scope, limit=1000):
                    atoms[atom.id] = atom
        if not atoms:
            return []
        superseded = self.graph.superseded_ids(list(atoms.keys()))
        visible = [
            a
            for aid, a in atoms.items()
            if aid not in superseded and self._visible(a, task_id, host, now)
        ]
        visible.sort(key=lambda a: a.confidence, reverse=True)
        return [self.atom_to_dict(a) for a in visible[:cap]]

    def context(
        self,
        query: str | None = None,
        task_id: str | None = None,
        host: str | None = None,
        limit: int = 8,
        max_chars: int = 4000,
    ) -> str:
        """Markdown block for a system prompt: standing instructions and
        preferences first, then query-relevant memories. Empty string if
        there is nothing to say."""
        with self._lock:
            now = datetime.now(UTC)
            standing = self._standing_instructions(task_id, host, now, cap=25)
            listed_ids = {a["id"] for a in standing}

            relevant_lines: list[str] = []
            if query:
                hits = self.recall(query, task_id=task_id, host=host, limit=limit, track=True)
                for h in hits:
                    if h["id"] in listed_ids:
                        continue
                    line = f"- [{h['category']}] {h['content']}"
                    contradicted_by = ", ".join(
                        c["content"] for c in h["contradictions"] if c.get("content")
                    )
                    if contradicted_by:
                        line += f" (contradicted by: {contradicted_by})"
                    relevant_lines.append(line)

            if not standing and not relevant_lines:
                return ""

            parts = ["## Memory (memstrand)"]
            if standing:
                parts.append("\n### Standing instructions & preferences")
                parts.extend(f"- [{a['category']}] {a['content']}" for a in standing)
            if relevant_lines:
                parts.append("\n### Relevant memories")
                parts.extend(relevant_lines)

            return "\n".join(parts)[:max_chars]

    # -- grounded answering -------------------------------------------------------

    def grounded_answer(self, question: str, **recall_kwargs) -> str:
        """Answer ONLY from stored memories. No retrieval = 'I don't know'.
        A down local LLM returns a clear message instead of raising."""
        with self._lock:
            hits = self.recall(question, **recall_kwargs)
            if not hits:
                return "I don't know — nothing in memory covers this."
            context_lines = "\n".join(
                f"- [{h['category']}] {h['content']} (score {h['score']:.2f})"
                for h in hits
                if h["content"]
            )
            if not context_lines:
                return "I don't know — nothing in memory covers this."
            try:
                return chat_completion(
                    self.llm_base_url,
                    self.llm_model,
                    [
                        {
                            "role": "system",
                            "content": "Answer ONLY from the memories below. "
                            "If they don't cover the question, say so.",
                        },
                        {"role": "user", "content": f"Memories:\n{context_lines}\n\nQ: {question}"},
                    ],
                    timeout=120.0,
                    api_key=self.llm_api_key,
                )
            except LLMUnavailable as e:
                # Surface the real reason (down vs. rejected the request)
                # instead of a generic "unavailable" that hides a 4xx like
                # "model not found" behind the wrong diagnosis.
                return str(e)

    # -- sessions / distillation / consolidation ---------------------------------

    def improve(self, session_ids: list[str] | None = None, background: bool = False) -> dict:
        """Distill raw sessions into typed permanent memory. `None` means
        every undistilled session. `background=True` runs on a daemon
        thread and returns `{"queued": [...]}` immediately."""
        from .pipeline import distill

        with self._lock:
            if background:
                return distill.improve_async(self, session_ids)
            return distill.distill_sessions(self, session_ids)

    def consolidate(self, now: datetime | None = None) -> dict:
        """Expire, decay, dedup, replay, and distill pending sessions."""
        from .pipeline import consolidate as consolidate_pipeline

        with self._lock:
            return consolidate_pipeline.run(self, now=now)

    # -- export / import / ingest -------------------------------------------------

    def export(self, fmt: str = "jsonl") -> str:
        with self._lock:
            if fmt == "jsonl":
                return export_module.to_jsonl(self)
            if fmt == "markdown":
                return export_module.to_markdown(self)
            raise ValueError(f"unknown export format: {fmt!r} (expected 'jsonl' or 'markdown')")

    def import_jsonl(self, text: str) -> dict:
        with self._lock:
            return export_module.from_jsonl(self, text)

    def ingest(self, path, task_id: str | None = None, force: bool = False) -> dict:
        from . import ingest as ingest_module

        with self._lock:
            return ingest_module.ingest_path(self, path, task_id=task_id, force=force)

    # -- lifecycle ----------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            for store in (self.vectors, self.graph, self.sessions, self.keyword):
                closer = getattr(store, "close", None)
                if closer:
                    closer()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
