"""Sleep consolidation: expire, Hebbian decay, near-duplicate merge,
spaced-repetition replay, and pending-session distillation.

ZenBrain's stealable mechanics, adapted:
  - per-fact FSRS state {difficulty, stability_days, next_review};
    retrieval success counts as a review (stability grows)
  - replay: pick the top-k due facts by usage/confidence/recency/
    instability, restabilize (stability x1.5, cap 365d)
  - Hebbian edge decay on a schedule; prune weak EDGES, never facts
  - near-duplicate atoms (same scope, same category, cosine >= 0.95) get
    linked newer-supersedes-older rather than deleted — nothing is ever
    silently destroyed by consolidation

Runs as `Memory.consolidate()` — a cron job, not the request path.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from ..schema import MemoryEdge, RelationType, ScopedQuery
from .extract import LLMUnavailable, chat_completion

_PAGE = 200
_DEDUP_THRESHOLD = 0.95
_REPLAY_HALF_LIFE_DAYS = 30.0
_REPLAY_BATCH = 50


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
    atom.fsrs_next_review = datetime.now(UTC) + timedelta(days=atom.fsrs_stability_days)


def replay_priority(atom, now: datetime) -> float:
    """Which memories get replayed tonight: usage, confidence, recency
    (exp decay off whichever of updated_at/last_accessed_at is newer,
    half-life 30d), and instability (FSRS difficulty)."""
    last_touched = max(
        (d for d in (atom.updated_at, atom.last_accessed_at) if d is not None),
        default=atom.created_at,
    )
    # Defensive naive-datetime coercion: MemoryAtom's own validators already
    # normalize these to aware UTC, but this stays defensive in case either
    # ever arrives from an untyped source without going through the model.
    if last_touched.tzinfo is None:
        last_touched = last_touched.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    age_days = max((now - last_touched).total_seconds() / 86400.0, 0.0)
    recency = math.exp(-math.log(2) * age_days / _REPLAY_HALF_LIFE_DAYS)
    return (
        0.3 * min(atom.access_count / 10.0, 1.0)
        + 0.3 * atom.confidence
        + 0.2 * recency
        + 0.2 * atom.fsrs_difficulty
    )


def _is_llm_reachable(mem) -> bool:
    """A non-transient `LLMUnavailable` (4xx: bad model, bad key, ...)
    still means the endpoint ANSWERED — just rejected the ping request —
    so distillation should still be attempted and report the real
    per-session error, instead of this gate silently skipping it with a
    misleading "local LLM unreachable"."""
    try:
        chat_completion(
            mem.llm_base_url,
            mem.llm_model,
            [{"role": "user", "content": "ping"}],
            timeout=3.0,
            api_key=mem.llm_api_key,
        )
        return True
    except LLMUnavailable as e:
        return not e.transient


def _iter_all_atoms(mem):
    offset = 0
    while True:
        batch = mem.graph.list_atoms(limit=_PAGE, offset=offset)
        if not batch:
            return
        yield from batch
        offset += len(batch)
        if len(batch) < _PAGE:
            return


def _link_near_duplicates(mem) -> int:
    """For each atom, vector-search its own exact scope (never globals
    pulled in by the task-or-global OR-filter — post-filtered below) for
    a same-category near-duplicate. Links newer -> older via `supersedes`;
    never deletes. Skips pairs already linked either direction."""
    linked_pairs: set[tuple[str, str]] = set()
    for edge in mem.graph.all_edges():
        if edge.relation == RelationType.supersedes:
            linked_pairs.add((edge.source_id, edge.target_id))
            linked_pairs.add((edge.target_id, edge.source_id))

    linked = 0
    for atom in _iter_all_atoms(mem):
        if atom.id in mem.graph.superseded_ids([atom.id]):
            continue
        q = ScopedQuery(
            text=atom.content, task_id=atom.task_id, host=atom.host,
            categories=[atom.category], limit=5,
        )
        for cand_id, cosine in mem.vectors.similar_to(atom.id, q, limit=5):
            if cosine < _DEDUP_THRESHOLD or cand_id == atom.id:
                continue
            if (atom.id, cand_id) in linked_pairs:
                continue
            cand = mem.graph.get_atom(cand_id)
            if cand is None or cand.category != atom.category:
                continue
            if cand.task_id != atom.task_id or cand.host != atom.host:
                continue  # exact-scope match only; the OR-global filter can leak globals in
            if cand_id in mem.graph.superseded_ids([cand_id]):
                continue
            newer, older = (atom, cand) if atom.created_at >= cand.created_at else (cand, atom)
            mem.graph.add_edge(
                MemoryEdge(
                    source_id=newer.id, target_id=older.id, relation=RelationType.supersedes,
                    reason="near-duplicate (consolidate)",
                )
            )
            linked_pairs.add((newer.id, older.id))
            linked_pairs.add((older.id, newer.id))
            linked += 1
    return linked


def _replay(mem, now: datetime) -> int:
    due = [
        atom
        for atom in _iter_all_atoms(mem)
        if atom.fsrs_next_review is None or atom.fsrs_next_review <= now
    ]
    due.sort(key=lambda a: replay_priority(a, now), reverse=True)
    top = due[:_REPLAY_BATCH]
    for atom in top:
        atom.fsrs_stability_days = min(atom.fsrs_stability_days * 1.5, 365.0)
        atom.fsrs_next_review = now + timedelta(days=atom.fsrs_stability_days)
        mem.graph.update_fsrs(
            atom.id, atom.fsrs_difficulty, atom.fsrs_stability_days, atom.fsrs_next_review
        )
    return len(top)


def run(mem, now: datetime | None = None) -> dict:
    """Full consolidation pass. Never raises — EACH stage (expire, decay,
    dedup, replay, distill) is wrapped in its own try/except and recorded
    as `{"error": ...}` in the report instead of aborting the rest: a
    broken store call in one stage (or one bad atom within expire) must
    never prevent the others from running, since this is a cron job with
    no one watching it fail."""
    now = now or datetime.now(UTC)
    report: dict = {}

    try:
        expired_ids = mem.graph.expired_ids(now)
    except Exception as e:  # noqa: BLE001 - never abort the rest of consolidation
        report["expired"] = {"error": str(e)}
    else:
        expired_count = 0
        expire_errors = []
        for atom_id in expired_ids:
            # Per-atom try: one atom whose vector/keyword/graph delete
            # blows up must not stop the rest of the expired batch from
            # being cleaned up.
            try:
                mem.vectors.delete(atom_id)
                mem.keyword.delete(atom_id)
                mem.graph.delete_atom(atom_id)
                mem.sessions.log_provenance(atom_id, "expire", "ttl elapsed")
                expired_count += 1
            except Exception as e:  # noqa: BLE001 - one bad atom must not stop the others
                expire_errors.append({"atom_id": atom_id, "error": str(e)})
        report["expired"] = expired_count
        if expire_errors:
            report["expire_errors"] = expire_errors

    try:
        report["edges_pruned"] = mem.graph.decay_edges()
    except Exception as e:  # noqa: BLE001
        report["edges_pruned"] = {"error": str(e)}

    try:
        report["duplicates_linked"] = _link_near_duplicates(mem)
    except Exception as e:  # noqa: BLE001
        report["duplicates_linked"] = {"error": str(e)}

    try:
        report["replayed"] = _replay(mem, now)
    except Exception as e:  # noqa: BLE001
        report["replayed"] = {"error": str(e)}

    if _is_llm_reachable(mem):
        try:
            report["distilled"] = mem.improve(None)
        except Exception as e:  # noqa: BLE001 - consolidation must never crash the cron job
            report["distilled"] = {"error": str(e)}
    else:
        report["distilled"] = {"skipped": "local LLM unreachable"}

    return report
