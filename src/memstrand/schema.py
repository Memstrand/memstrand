"""memstrand schema: the typed memory contract.

Adapted from Memanto's 13-category typed schema (moorcheh-ai/memanto),
reimplemented local-first. Every memory atom carries its type, its
provenance, and its confidence — the three things flat vector memory
throws away.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


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
    task_id: str | None = None
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


class MemoryEdge(BaseModel):
    """One typed relationship between two atoms."""

    source_id: str
    target_id: str
    relation: RelationType
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    reason: str = ""                 # why this edge exists (esp. contradicts)
    weight: float = Field(default=1.0, ge=0.0)  # Hebbian: use strengthens
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ScopedQuery(BaseModel):
    """A recall query. Scope is a hard filter, not a ranking signal."""

    text: str
    session_id: str | None = None
    task_id: str | None = None
    host: str | None = None
    categories: list[Category] | None = None
    limit: int = 10
