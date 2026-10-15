"""Store interfaces. All backends are swappable.

The GraphStore protocol exists because Kuzu upstream was archived after
the Oct 2025 acquisition: pin kuzu==0.11.3 today, swap tomorrow without
touching anything above this interface.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from ..schema import MemoryAtom, MemoryEdge, ScopedQuery


class GraphStore(Protocol):
    """Typed relationship graph. SQLite is the default (Kuzu upstream was
    archived and embedded Kuzu only allows one writer process); Kuzu
    remains available as an optional backend behind this same interface."""

    def add_atom(self, atom: MemoryAtom) -> None: ...  # upsert by id; all fields round-trip
    def add_edge(self, edge: MemoryEdge) -> None: ...  # no-op if either endpoint missing; dedupes by (source,target,relation)
    def get_atom(self, atom_id: str) -> MemoryAtom | None: ...
    def get_atoms(self, atom_ids: list[str]) -> dict[str, MemoryAtom]: ...
    def list_atoms(
        self,
        category: str | None = None,
        task_id: str | None = None,
        host: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryAtom]: ...  # newest created_at first. None = no filter; "" = only global (unset) values
    def count_atoms(self) -> int: ...
    def all_edges(self) -> list[MemoryEdge]: ...
    def neighbors(
        self, atom_id: str, relation: str | None = None, depth: int = 1
    ) -> list[tuple[MemoryAtom, MemoryEdge]]: ...  # both directions; edge keeps its true source/target
    def contradicting(self, atom_id: str) -> list[tuple[MemoryAtom, MemoryEdge]]: ...
    def superseded_ids(self, atom_ids: list[str]) -> set[str]: ...  # subset that are TARGET of a "supersedes" edge
    def strengthen_edge(self, a_id: str, b_id: str, amount: float = 0.1) -> None: ...  # either direction
    def decay_edges(self, factor: float = 0.98, prune_below: float = 0.1) -> int: ...  # returns #edges pruned
    def record_access(self, atom_ids: list[str], when: datetime) -> None: ...
    def update_fsrs(
        self, atom_id: str, difficulty: float, stability_days: float, next_review: datetime | None
    ) -> None: ...
    def expired_ids(self, now: datetime) -> list[str]: ...
    def ids_by_source(self, source: str) -> list[str]: ...
    def delete_atom(self, atom_id: str) -> None: ...  # atom + all its edges
    def close(self) -> None: ...


class VectorStore(Protocol):
    """Semantic similarity over atoms. Qdrant today.

    Scope (task_id/host) is a hard filter baked into `search`: a scoped
    query sees its own scope PLUS globals, an unscoped query sees only
    globals. TTL (`expires_at`) is filtered the same way everywhere else
    — expired atoms never come back from `search`.
    """

    def upsert(self, atom: MemoryAtom, embedding: list[float]) -> None: ...
    def search(
        self,
        embedding: list[float],
        query: ScopedQuery,
        now: datetime | None = None,
        limit: int | None = None,
    ) -> list[tuple[str, float]]: ...
    def exists(self, atom_id: str) -> bool: ...
    def delete(self, atom_id: str) -> None: ...
    def close(self) -> None: ...


class KeywordIndex(Protocol):
    """Lexical (BM25) index over atom content. SQLite FTS5 today.

    Complements VectorStore in hybrid recall: catches exact tokens
    (ids, codes) that a semantic embedding ranks poorly. Same scope/TTL
    rules as VectorStore.search.
    """

    def add(self, atom: MemoryAtom) -> None: ...
    def delete(self, atom_id: str) -> None: ...
    def search(
        self,
        text: str,
        query: ScopedQuery,
        limit: int = 30,
        now: datetime | None = None,
    ) -> list[tuple[str, float]]: ...  # (atom_id, normalized_score) best first
    def close(self) -> None: ...


class SessionStore(Protocol):
    """Fast raw session cache (L1) + append-only provenance ledger. SQLite today."""

    def append_turn(
        self, session_id: str, role: str, content: str, task_id: str | None = None,
        ts: float | None = None,
    ) -> None: ...  # ts: override the recorded time (e.g. re-importing an export); None = now
    def get_session(self, session_id: str, after_id: int = 0) -> list[dict]: ...  # dicts include "id"
    def list_sessions(self, task_id: str | None = None) -> list[str]: ...
    def clear_session(self, session_id: str) -> None: ...
    def log_provenance(
        self, atom_id: str, action: str, detail: str = "", ts: float | None = None
    ) -> None: ...  # append-only; ts override for re-importing an export, None = now
    def history(self, atom_id: str) -> list[dict]: ...  # oldest first: [{"action","detail","ts"}]
    def dump_ledger(self) -> list[dict]: ...  # full ledger, for export
    def last_distilled_turn_id(self, session_id: str) -> int: ...
    def mark_distilled(self, session_id: str, last_turn_id: int) -> None: ...
    def undistilled_sessions(self) -> list[str]: ...  # sessions with turns newer than last_turn_id
    def close(self) -> None: ...
