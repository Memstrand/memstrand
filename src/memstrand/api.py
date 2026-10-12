"""memstrand public API: remember / recall / improve / forget."""

from __future__ import annotations

import hashlib
import os
from datetime import UTC

import httpx

from .schema import Category, MemoryAtom, Provenance, RelationType, ScopedQuery
from .stores.base import GraphStore, SessionStore, VectorStore
from .stores.graph_kuzu import KuzuGraphStore
from .stores.sqlite import SqliteSessionStore
from .stores.vector_qdrant import QdrantVectorStore


class Memory:
    """Local-first typed memory. All LLM calls go to a local endpoint."""

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        kuzu_path: str = "./kuzu_data",
        sqlite_path: str = "./memstrand.db",
        llm_base_url: str | None = None,
        llm_model: str | None = None,
        # dependency injection: real backends by default, swappable in tests
        embedder=None,
        vector_store: VectorStore | None = None,
        graph_store: GraphStore | None = None,
        session_store: SessionStore | None = None,
    ):
        if embedder is None:
            from fastembed import TextEmbedding

            embedder = TextEmbedding()  # local ONNX, no API key
        self.embedder = embedder
        self.vectors = vector_store or QdrantVectorStore(url=qdrant_url)
        self.graph = graph_store or KuzuGraphStore(path=kuzu_path)
        self.sessions = session_store or SqliteSessionStore(path=sqlite_path)
        self.llm_base_url = (
            llm_base_url or os.environ.get("MEMSTRAND_LLM_BASE_URL", "http://localhost:8081/v1")
        )
        self.llm_model = llm_model or os.environ.get("MEMSTRAND_LLM_MODEL", "local")

    @classmethod
    def in_memory(cls, **kwargs) -> "Memory":
        """Zero-dependency memory for demos and tests: Qdrant runs in
        `:memory:` mode, Kuzu + SQLite live in a temp dir. `pip install
        memstrand` is the entire setup — no Docker, no server, no LLM.

        Data is ephemeral (it dies with the process). For persistence, use
        the default constructor: Docker Qdrant + on-disk Kuzu/SQLite, with an
        optional local LLM endpoint for the `improve()` distillation pass.
        """
        import tempfile

        from qdrant_client import QdrantClient

        tmp = tempfile.mkdtemp(prefix="memstrand-")
        return cls(
            vector_store=QdrantVectorStore(client=QdrantClient(":memory:")),
            graph_store=KuzuGraphStore(path=f"{tmp}/kuzu"),
            session_store=SqliteSessionStore(path=f"{tmp}/memstrand.db"),
            **kwargs,
        )

    def _embed(self, text: str) -> list[float]:
        return next(iter(self.embedder.embed([text])))

    @staticmethod
    def _atom_id(content: str) -> str:
        """Content-hash id: identical re-adds are no-ops by construction."""
        return f"mem:{hashlib.sha256(content.encode()).hexdigest()[:12]}"

    # -- fast write path (L0/L3): no extraction, ~ms -------------------------
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
    ) -> MemoryAtom:
        from datetime import datetime, timedelta

        atom = MemoryAtom(
            id=self._atom_id(content),
            category=category,
            content=content,
            provenance=provenance,
            confidence=confidence,
            session_id=session_id,
            task_id=task_id,
            host=host,
            expires_at=(
                datetime.now(UTC) + timedelta(days=ttl_days)
                if ttl_days
                else None
            ),
        )
        self.vectors.upsert(atom, self._embed(content))
        self.graph.add_atom(atom)
        self.sessions.log_provenance(atom.id, "remember", provenance.value)
        return atom

    # -- scoped read path ----------------------------------------------------
    def recall(
        self,
        query: str,
        session_id: str | None = None,
        task_id: str | None = None,
        host: str | None = None,
        categories: list[Category] | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Scope filter is a hard filter. Unscoped queries never see
        task-scoped atoms — the Playwright-bug fix. Results are hydrated:
        content, type, provenance, and graph neighbors ride along, and
        contradictions are surfaced, never hidden."""
        q = ScopedQuery(
            text=query, session_id=session_id, task_id=task_id,
            host=host, categories=categories, limit=limit,
        )
        hits = self.vectors.search(self._embed(query), q)
        results = []
        for atom_id, score in hits:
            atom = self.graph.get_atom(atom_id)
            neighbor_rows = []
            for n_atom, edge in self.graph.neighbors(atom_id, depth=1):
                neighbor_rows.append(
                    {
                        "id": n_atom.id,
                        "relation": edge.relation.value,
                        "content": n_atom.content,
                        "weight": edge.weight,
                    }
                )
                self.graph.strengthen_edge(atom_id, n_atom.id)  # Hebbian
            results.append(
                {
                    "id": atom_id,
                    "score": score,
                    "content": atom.content if atom else None,
                    "category": atom.category.value if atom else None,
                    "provenance": atom.provenance.value if atom else None,
                    "confidence": atom.confidence if atom else None,
                    "task_id": atom.task_id if atom else None,
                    "neighbors": neighbor_rows,
                    "contradictions": [
                        n for n in neighbor_rows if n["relation"] == RelationType.contradicts.value
                    ],
                }
            )
        return results

    # -- session lifecycle ---------------------------------------------------
    def improve(self, session_ids: list[str]) -> dict:
        """Distill raw sessions into typed permanent memory."""
        from .pipeline import distill

        return distill.distill_sessions(self, session_ids)

    def forget(self, atom_id: str) -> None:
        """Delete with provenance-chain cleanup. Never silent."""
        self.vectors.delete(atom_id)
        self.graph.delete_atom(atom_id)
        self.sessions.log_provenance(atom_id, "forget", "explicit user delete")
        # TODO: propagate to session turns that cited this atom

    # -- grounded answering ---------------------------------------------------
    def grounded_answer(self, question: str, **recall_kwargs) -> str:
        """Answer ONLY from stored memories. No retrieval = 'I don't know'."""
        hits = self.recall(question, **recall_kwargs)
        if not hits:
            return "I don't know — nothing in memory covers this."
        context = "\n".join(
            f"- [{h['category']}] {h['content']} (score {h['score']:.2f})"
            for h in hits
            if h["content"]
        )
        if not context:
            return "I don't know — nothing in memory covers this."
        resp = httpx.post(
            f"{self.llm_base_url}/chat/completions",
            headers={"Authorization": "Bearer <redacted>"},
            json={
                "model": self.llm_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Answer ONLY from the memories below. "
                        "If they don't cover the question, say so.",
                    },
                    {"role": "user", "content": f"Memories:\n{context}\n\nQ: {question}"},
                ],
            },
            timeout=120,
        )
        return resp.json()["choices"][0]["message"]["content"]
