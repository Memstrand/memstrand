"""Recall ranking: hybrid fusion (dense + keyword) and the final blended score.

Kept separate from api.py so `Memory.recall` stays readable: everything
here is pure functions over plain data, easy to unit-test without a store.
"""

from __future__ import annotations

import math
from datetime import datetime

from .schema import Provenance

_PROVENANCE_WEIGHT = {
    Provenance.explicit_statement: 1.0,
    Provenance.validated: 1.0,
    Provenance.corrected: 1.0,
    Provenance.observed: 0.8,
    Provenance.imported: 0.7,
    Provenance.inferred: 0.6,
}


def provenance_weight(provenance: Provenance) -> float:
    return _PROVENANCE_WEIGHT.get(provenance, 0.6)


def recency_weight(dt: datetime | None, now: datetime, half_life_days: float) -> float:
    """Exponential decay: 1.0 at age 0, 0.5 at `half_life_days`."""
    if dt is None:
        return 0.0
    age_days = max((now - dt).total_seconds() / 86400.0, 0.0)
    return math.exp(-math.log(2) * age_days / half_life_days)


def reciprocal_rank_fusion(
    dense: list[tuple[str, float]],
    keyword: list[tuple[str, float]],
    k: int = 60,
) -> dict[str, float]:
    """RRF: fuse two ranked lists into one score per id, by RANK rather
    than raw score — dense cosine and BM25 live on unrelated scales."""
    fused: dict[str, float] = {}
    for rank, (atom_id, _score) in enumerate(dense):
        fused[atom_id] = fused.get(atom_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, (atom_id, _score) in enumerate(keyword):
        fused[atom_id] = fused.get(atom_id, 0.0) + 1.0 / (k + rank + 1)
    return fused


def normalize(values: dict[str, float]) -> dict[str, float]:
    """Min-max normalize to 0..1 within this result set. A single value
    (or a tie across all of them) normalizes to 1.0 — it's the best thing
    available in this set, not an arbitrary midpoint.

    NOT used for recall relevance (see `absolute_relevance` below): a
    lone weak candidate would normalize to a perfect 1.0 here, which is
    exactly the bug this module now avoids for ranking. Kept as a
    general-purpose utility (e.g. for presenting a relative confidence
    bar across a known-good result set)."""
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi == lo:
        return {k: 1.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def sigmoid(x: float, k: float = 1.0) -> float:
    """Squash an unbounded score (e.g. a cross-encoder reranker logit)
    into an absolute 0..1 relevance signal."""
    return 1.0 / (1.0 + math.exp(-k * x))


def keyword_strength(bm25_score: float | None, k: float = 2.0) -> float:
    """Absolute 0..1 keyword-match strength from SQLite's bm25() value
    (more negative = a better match; magnitude is unbounded). `s/(s+k)`
    saturates smoothly as the match gets stronger, without ever reading
    1.0 just because a candidate happens to be the only keyword hit in
    this query's result set (the per-set min-max normalization bug).
    `None` (no keyword hit at all) -> 0.0."""
    if bm25_score is None:
        return 0.0
    s = max(-bm25_score, 0.0)
    return s / (s + k)


def absolute_relevance(
    *,
    dense_score: float | None,
    keyword_score: float | None,
    agreement_bonus: float = 0.05,
) -> float:
    """Recall relevance as an ABSOLUTE signal, not normalized within the
    current result set. A lone irrelevant candidate (e.g. the only atom
    visible in scope, with near-zero cosine similarity to the query) must
    score low — not 1.0 just because it's "the best of one".

    `dense_score` is a raw cosine similarity (Qdrant's COSINE distance can
    read slightly negative for near-orthogonal vectors, so it's clipped to
    0..1); `keyword_score` is already an absolute 0..1 value (e.g. from
    `keyword_strength`), or `None` if the atom wasn't a keyword hit at all.
    A small bonus rewards atoms both retrievers agree on, since independent
    agreement is itself evidence of relevance."""
    dense = max(min(dense_score, 1.0), 0.0) if dense_score is not None else 0.0
    keyword = keyword_score if keyword_score is not None else 0.0
    base = max(dense, keyword)
    if dense_score is not None and keyword_score is not None:
        base = min(base + agreement_bonus, 1.0)
    return base


def blended_score(
    *,
    relevance: float,
    confidence: float,
    recency: float,
    provenance: float,
    access_count: int,
) -> float:
    """The final recall ranking signal. Weights sum to 1.0."""
    return (
        0.65 * relevance
        + 0.12 * confidence
        + 0.10 * recency
        + 0.08 * provenance
        + 0.05 * min(access_count / 10.0, 1.0)
    )
