"""Session-end distillation (the 'improve' step).

Cognee's improve() pattern, local-first:
  1. extract candidate atoms from raw session turns (only the turns after
     the session's last distilled turn id — incremental, not a full replay)
  2. dedup by scope-aware content hash (identical re-add = no-op)
  3. conflict check: genuinely contradictory atoms become
     `contradicts` EDGES — additive, both sides stay readable
  4. promote survivors into the graph + vector + keyword stores, scoped
     global (task_id=None) or task (task_id=session's task) per the
     extractor's scope call

Runs async at session end / pre-compact hook, or as part of
`Memory.consolidate()`. Never blocks the chat loop when `background=True`.
Idempotent: scope-aware content-hash ids make a crashed run safe to re-run,
and `mark_distilled` means a clean run never re-sends the same turns twice.
"""

from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING

from ..schema import MemoryEdge, Provenance, RelationType
from .extract import LLMUnavailable, _extract_json_array, chat_completion, extract_atoms

if TYPE_CHECKING:
    from ..api import Memory

JUDGE_SYSTEM_PROMPT = """\
You resolve one candidate memory against existing memories.
Return a single JSON object, nothing else:
{"action": "add" | "update" | "delete" | "none",
 "target_id": "<id of the existing memory, or null>",
 "relation": "supports" | "extends" | "contradicts" | "supersedes" | "related_to",
 "reason": "<one line>"}
Rules:
- "none": the candidate is already covered by an existing memory, or is noise.
- "update": the candidate replaces or refines the target. The old memory
  stays queryable, linked by a supersedes edge. Nothing is deleted.
- "delete": the candidate shows the target memory is WRONG. Store the
  correction as a new memory linked by a supersedes edge. Nothing is deleted.
- "contradicts" relation: genuinely contradictory memories both stay
  readable, joined by a contradicts edge.
- Default "add" for new information; link it "related_to" the most similar
  existing memory.\
"""


def _judge(mem: Memory, candidate: dict, existing: list[dict]) -> dict:
    """ADD/UPDATE/DELETE/NONE resolution (Mem0's pattern)."""
    if not existing:
        return {"action": "add", "target_id": None, "relation": "related_to", "reason": "no similar memories"}
    lines = "\n".join(
        f"- [{h['id']}] ({h['category']}) {h['content']}" for h in existing if h["content"]
    )
    raw = chat_completion(
        mem.llm_base_url,
        mem.llm_model,
        [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Candidate ({candidate['category'].value}): {candidate['content']}\n"
                    f"Existing memories:\n{lines}"
                ),
            },
        ],
        timeout=120.0,
        api_key=mem.llm_api_key,
    )
    try:
        verdict = _extract_json_array(f"[{raw}]")[0]
    except (ValueError, json.JSONDecodeError, IndexError, TypeError, AttributeError):
        return {"action": "add", "target_id": None, "relation": "related_to", "reason": "judge unparseable; defaulting to add"}
    action = verdict.get("action", "add")
    if action not in ("add", "update", "delete", "none"):
        action = "add"
    return {
        "action": action,
        "target_id": verdict.get("target_id"),
        "relation": verdict.get("relation", "related_to"),
        "reason": str(verdict.get("reason", ""))[:200],
    }


def _target_task_id(candidate: dict, session_task_id: str | None) -> str | None:
    """Durable facts about the user (preference/instruction/relationship,
    or anything the extractor explicitly scoped "global") are stored
    global; everything else inherits the session's task."""
    return None if candidate.get("scope") == "global" else session_task_id


def _distill_one_session(mem: Memory, session_id: str) -> dict:
    report: dict = {
        "session": session_id, "atoms": 0, "edges": 0, "contradictions": 0,
        "skipped_dedup": 0, "skipped_none": 0, "errors": [],
    }
    after_id = mem.sessions.last_distilled_turn_id(session_id)
    turns = mem.sessions.get_session(session_id, after_id=after_id)
    if not turns:
        return report
    session_task_id = next((t.get("task_id") for t in turns if t.get("task_id")), None)

    try:
        candidates = extract_atoms(turns, mem.llm_base_url, mem.llm_model, api_key=mem.llm_api_key)
    except LLMUnavailable as e:
        report["errors"].append({"session": session_id, "error": str(e)})
        return report  # don't mark distilled: a later run retries these same turns

    last_turn_id = max(t["id"] for t in turns)
    for cand in candidates:
        target_task_id = _target_task_id(cand, session_task_id)
        atom_id = mem._atom_id(cand["content"], target_task_id, None)
        if mem.vectors.exists(atom_id):
            report["skipped_dedup"] += 1
            continue
        similar = [
            h
            for h in mem.recall(cand["content"], task_id=target_task_id, limit=5, track=False)
            if h["similarity"] is not None and h["similarity"] >= 0.6 and h["id"] != atom_id
        ]
        verdict = _judge(mem, cand, similar)
        if verdict["action"] == "none":
            report["skipped_none"] += 1
            continue
        # Prompt-injection defense: the judge is told about `similar`
        # candidates and nothing else, but an adversarial session
        # transcript could still coax it into naming an unrelated id
        # (e.g. one from a different task/scope) as `target_id`. Only
        # ever link to an id that was actually shown to it.
        similar_ids = {h["id"] for h in similar}
        target = verdict["target_id"]
        if target is not None and target not in similar_ids:
            target = None
        provenance = cand["provenance"]
        if verdict["action"] == "delete":
            # A retraction is a first-class memory, not a deletion.
            provenance = Provenance.corrected
        atom = mem.remember(
            cand["content"],
            category=cand["category"],
            provenance=provenance,
            confidence=cand["confidence"],
            session_id=session_id,
            task_id=target_task_id,
        )
        report["atoms"] += 1
        if verdict["action"] in ("update", "delete") and target:
            mem.graph.add_edge(
                MemoryEdge(
                    source_id=atom.id, target_id=target, relation=RelationType.supersedes,
                    reason=verdict["reason"] or "session distillation",
                )
            )
            report["edges"] += 1
        elif verdict["relation"] == "contradicts" and target:
            mem.graph.add_edge(
                MemoryEdge(
                    source_id=atom.id, target_id=target, relation=RelationType.contradicts,
                    reason=verdict["reason"] or "session distillation",
                )
            )
            report["edges"] += 1
            report["contradictions"] += 1
        elif target and similar:
            # new information, linked to its neighborhood
            mem.graph.add_edge(
                MemoryEdge(
                    source_id=atom.id, target_id=target, relation=RelationType.related_to,
                    reason=verdict["reason"] or "same topic",
                )
            )
            report["edges"] += 1

    mem.sessions.mark_distilled(session_id, last_turn_id)
    return report


def _merge_reports(session_ids: list[str], results: list[dict]) -> dict:
    merged: dict = {
        "sessions": session_ids, "atoms": 0, "edges": 0, "contradictions": 0,
        "skipped_dedup": 0, "skipped_none": 0, "errors": [],
    }
    for r in results:
        for key in ("atoms", "edges", "contradictions", "skipped_dedup", "skipped_none"):
            merged[key] += r[key]
        merged["errors"].extend(r["errors"])
    return merged


def distill_sessions(mem: Memory, session_ids: list[str] | None) -> dict:
    """Distill `session_ids`, or every undistilled session when None.
    Each session is distilled incrementally (only new turns)."""
    ids = session_ids if session_ids is not None else mem.sessions.undistilled_sessions()
    results = [_distill_one_session(mem, sid) for sid in ids]
    return _merge_reports(ids, results)


def improve_async(mem: Memory, session_ids: list[str] | None) -> dict:
    """Non-blocking `improve()`: runs `distill_sessions` on a daemon
    thread and returns immediately. A second call while one is already
    running for this `Memory` gets `{"queued": [], "busy": True}` instead
    of racing a concurrent distillation pass."""
    if not mem._improve_lock.acquire(blocking=False):
        return {"queued": [], "busy": True}
    ids = session_ids if session_ids is not None else mem.sessions.undistilled_sessions()

    def _run() -> None:
        try:
            distill_sessions(mem, session_ids)
        finally:
            mem._improve_lock.release()

    threading.Thread(target=_run, daemon=True).start()
    return {"queued": ids}
