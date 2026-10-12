"""Sleep consolidation: FSRS + replay + Hebbian decay.

ZenBrain's stealable mechanics, adapted:
  - per-fact FSRS state {difficulty, stability_days, next_review};
    retrieval success counts as a review (stability grows)
  - replay: nightly, pick top-k by
      0.3*access_freq + 0.3*emotion + 0.2*recency + 0.2*instability,
    restabilize (stability x1.5, cap 365d)
  - Hebbian edge decay on a schedule; prune weak EDGES, never facts

Runs as a cron job, not in the request path.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta


def fsrs_recall_probability(stability_days: float, days_since_review: float) -> float:
    """R = e^(-t/S): probability the fact is still retrievable."""
    if stability_days <= 0:
        return 0.0
    return math.exp(-days_since_review / stability_days)


def record_review(atom, success: bool) -> None:
    """A successful recall strengthens; a miss weakens. Desirable difficulty."""
    if success:
        atom.fsrs_stability_days = min(atom.fsrs_stability_days * 1.5, 365.0)
        atom.fsrs_difficulty = max(atom.fsrs_difficulty - 0.05, 0.1)
    else:
        atom.fsrs_stability_days = max(atom.fsrs_stability_days * 0.7, 1.0)
        atom.fsrs_difficulty = min(atom.fsrs_difficulty + 0.05, 0.9)
    atom.fsrs_next_review = datetime.now(UTC) + timedelta(
        days=atom.fsrs_stability_days
    )


def replay_priority(atom) -> float:
    """Which memories get replayed tonight."""
    return (
        0.3 * min(atom.access_count / 10.0, 1.0)
        + 0.3 * atom.confidence
        + 0.2 * 1.0  # recency term: compute from updated_at in real impl
        + 0.2 * atom.fsrs_difficulty
    )
