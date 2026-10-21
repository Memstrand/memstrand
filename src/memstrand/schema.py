"""memstrand schema: the typed memory contract.

Adapted from Memanto's 13-category typed schema (moorcheh-ai/memanto),
reimplemented local-first. Every memory atom carries its type, its
provenance, and its confidence — the three things flat vector memory
throws away.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

# -- scopes ---------------------------------------------------------------------
#
# An atom belongs to a SET of scope keys (see docs/design/scoping-v2.md):
# repo keys like "github.com/org/repo", "local:host:/path", or named tasks
# like "notes". GLOBAL_SCOPE is a member like any other -- {"", "notes"}
# means "global, and also contributed by the notes task" -- so merging is
# plain set union and removing a scope can never silently make an atom
# global (an atom whose set becomes empty is deleted instead).

GLOBAL_SCOPE = ""
_SCOPE_DELIM = "|"


def normalize_scopes(scopes: Iterable[str | None] | None) -> list[str]:
    """Deduped, sorted (GLOBAL_SCOPE first), whitespace-stripped. `None`
    entries mean global. An empty result means global too -- every atom
    belongs to at least one scope. Rejects keys containing the storage
    delimiter or control characters."""
    out: set[str] = set()
    for s in scopes or ():
        key = (s or "").strip()
        if _SCOPE_DELIM in key or any(ord(ch) < 32 for ch in key):
            raise ValueError(f"invalid scope key {key!r}: must not contain '|' or control characters")
        out.add(key)
    return sorted(out) if out else [GLOBAL_SCOPE]


def encode_scopes(scopes: Iterable[str]) -> str:
    """Storage form: "|a|b|", global member first ("||a|" = global + a,
    "||" = global only). Membership test is a substring match on
    f"|{key}|" -- correct because keys never contain "|" and the global
    member (the only one producing "||") always sorts first."""
    return _SCOPE_DELIM + _SCOPE_DELIM.join(normalize_scopes(scopes)) + _SCOPE_DELIM


def decode_scopes(value: str | None) -> list[str]:
    """Inverse of `encode_scopes`. A value that isn't in encoded form is a
    legacy single `task_id` ("" / None = global)."""
    if not value:
        return [GLOBAL_SCOPE]
    if len(value) >= 2 and value.startswith(_SCOPE_DELIM) and value.endswith(_SCOPE_DELIM):
        return normalize_scopes(value[1:-1].split(_SCOPE_DELIM))
    return normalize_scopes([value])


def scopes_visible(atom_scopes: Iterable[str], query_scopes: Iterable[str]) -> bool:
    """THE scope rule: an atom is visible to a query if it's global or
    shares at least one scope with the query."""
    atom_set = set(atom_scopes)
    return GLOBAL_SCOPE in atom_set or bool(atom_set & set(query_scopes))


def _legacy_task_id_to_scopes(data, *, query: bool):
    """Accept the pre-v2 `task_id=` keyword: it becomes a one-element scope
    set (an atom with no task is global; a query with no task sees only
    global)."""
    if isinstance(data, dict) and "task_id" in data:
        data = dict(data)
        task = data.pop("task_id")
        if data.get("scopes") is None:
            data["scopes"] = ([task] if task else []) if query else [task or GLOBAL_SCOPE]
    return data


def _assume_utc(dt: datetime | None) -> datetime | None:
    """A naive datetime (no tzinfo) is assumed UTC rather than rejected.
    Without this, one naive `created_at` slipping in through `import_jsonl`
    (hand-edited JSONL, an older export, a third-party tool) permanently
    breaks every later `recall()`/`consolidate()` call that subtracts it
    from an aware `datetime.now(UTC)` (`TypeError: can't subtract
    offset-naive and offset-aware datetimes`)."""
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


class Category(str, Enum):
    """The 13 typed memory categories."""

    instruction = "instruction"      # durable directives ("always do X")
    fact = "fact"                    # stable world knowledge
    decision = "decision"            # a choice that was made + rationale
    goal = "goal"                    # an outcome being worked toward
    commitment = "commitment"        # a promise (often with a deadline)
    preference = "preference"        # likes, dislikes, defaults
    relationship = "relationship"    # people, roles, how they connect
    context = "context"              # situational background
    event = "event"                  # something that happened (timestamped)
    learning = "learning"            # distilled lesson from experience
    observation = "observation"      # noticed, not yet verified
    artifact = "artifact"            # a produced thing (file, doc, link)
    error = "error"                  # a mistake + what it taught


class Provenance(str, Enum):
    """How this memory was learned. Never trust a memory without one."""

    explicit_statement = "explicit_statement"  # the user said it directly
    inferred = "inferred"                      # derived by the agent
    corrected = "corrected"                    # supersedes an earlier atom
    validated = "validated"                    # confirmed against a source
    observed = "observed"                      # seen in tool output / behavior
    imported = "imported"                      # bulk-loaded from elsewhere


class RelationType(str, Enum):
    """Typed edges between memories. Contradictions are edges, never deletes."""

    informs = "informs"
    supports = "supports"
    contradicts = "contradicts"      # additive only; both sides stay readable
    extends = "extends"
    supersedes = "supersedes"        # new atom replaces old; old stays queryable
    related_to = "related_to"
    precedes = "precedes"            # temporal ordering
    caused_by = "caused_by"


class MemoryAtom(BaseModel):
    """One typed unit of memory. The fundamental unit of the system."""

    id: str = Field(default_factory=lambda: f"mem:{uuid4().hex[:12]}")
    category: Category
    content: str
    provenance: Provenance = Provenance.inferred
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)

    # scope: the Playwright-bug fix. Retrieval filters on these FIRST.
    session_id: str | None = None
    scopes: list[str] = Field(default_factory=lambda: [GLOBAL_SCOPE])
    host: str | None = None
    agent: str | None = None

    # lifecycle
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None   # None = permanent
    access_count: int = 0

    # FSRS memory-strength state (per-fact forgetting curve)
    fsrs_difficulty: float = 0.5
    fsrs_stability_days: float = 30.0
    fsrs_next_review: datetime | None = None

    # provenance origin: where this memory came from, not why it was learned
    source: str | None = None        # absolute file path, URL, "claude-code", etc.
    source_ref: str | None = None    # locator within source, e.g. "notes/x.md#Heading [2]"
    last_accessed_at: datetime | None = None

    @field_validator(
        "created_at", "updated_at", "expires_at", "last_accessed_at", "fsrs_next_review",
        mode="after",
    )
    @classmethod
    def _naive_datetimes_assume_utc(cls, v: datetime | None) -> datetime | None:
        return _assume_utc(v)

    @model_validator(mode="before")
    @classmethod
    def _accept_task_id(cls, data):
        return _legacy_task_id_to_scopes(data, query=False)

    @field_validator("scopes", mode="before")
    @classmethod
    def _normalize_scopes(cls, v):
        return normalize_scopes(v)

    @property
    def is_global(self) -> bool:
        return GLOBAL_SCOPE in self.scopes

    @property
    def task_id(self) -> str | None:
        """Legacy single-scope view: None when global, else the first scope.
        Lossy for multi-scope atoms -- new code should use `scopes`."""
        return None if self.is_global else self.scopes[0]


class MemoryEdge(BaseModel):
    """One typed relationship between two atoms."""

    source_id: str
    target_id: str
    relation: RelationType
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    reason: str = ""                 # why this edge exists (esp. contradicts)
    weight: float = Field(default=1.0, ge=0.0)  # Hebbian: use strengthens
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("created_at", mode="after")
    @classmethod
    def _naive_created_at_assumes_utc(cls, v: datetime) -> datetime:
        return _assume_utc(v)


class ScopedQuery(BaseModel):
    """A recall query. Scope is a hard filter, not a ranking signal."""

    text: str
    session_id: str | None = None
    # the query's active scopes; global atoms are always visible on top
    scopes: list[str] = Field(default_factory=list)
    host: str | None = None
    categories: list[Category] | None = None
    limit: int = 10

    @model_validator(mode="before")
    @classmethod
    def _accept_task_id(cls, data):
        return _legacy_task_id_to_scopes(data, query=True)

    @field_validator("scopes", mode="before")
    @classmethod
    def _normalize_scopes(cls, v):
        return [s for s in normalize_scopes(v) if s != GLOBAL_SCOPE]

    @property
    def task_id(self) -> str | None:
        """Legacy single-scope view (first scope, or None)."""
        return self.scopes[0] if self.scopes else None
