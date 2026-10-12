"""Session-end distillation (the 'improve' step).

Cognee's improve() pattern, local-first:
  1. extract candidate atoms from raw session turns
  2. dedup by content hash (identical re-add = no-op)
  3. conflict check: genuinely contradictory atoms become
     `contradicts` EDGES — additive, both sides stay readable
  4. promote survivors into L2 (episodic, task-scoped) and
     L3/L5 (semantic + graph)

Runs async at session end / pre-compact hook. Never blocks the chat loop.
Idempotent: content-hash ids make a crashed run safe to re-run.
"""

from __future__ import annotations

import json
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


def distill_sessions(mem: Memory, session_ids: list[str]) -> dict:
    """Full implementation. Returns a per-stage report."""
    report: dict = {
        "sessions": session_ids,
        "atoms": 0,
        "edges": 0,
        "contradictions": 0,
        "skipped_dedup": 0,
        "skipped_none": 0,
        "errors": [],
    }
    for sid in session_ids:
        turns = mem.sessions.get_session(sid)
        if not turns:
            continue
        task_id = turns[0].get("task_id") or None
        try:
            candidates = extract_atoms(turns, mem.llm_base_url, mem.llm_model)
        except LLMUnavailable as e:
            report["errors"].append({"session": sid, "error": str(e)})
            continue  # content-hash ids make a later re-run safe
        for cand in candidates:
            atom_id = mem._atom_id(cand["content"])
            if mem.vectors.exists(atom_id):
                report["skipped_dedup"] += 1
                continue
            similar = [
                h
                for h in mem.recall(cand["content"], task_id=task_id, limit=5)
                if h["score"] is not None and h["score"] >= 0.6 and h["id"] != atom_id
            ]
            verdict = _judge(mem, cand, similar)
            if verdict["action"] == "none":
                report["skipped_none"] += 1
                continue
            provenance = cand["provenance"]
            if verdict["action"] == "delete":
                # A retraction is a first-class memory, not a deletion.
                provenance = Provenance.corrected
            atom = mem.remember(
                cand["content"],
                category=cand["category"],
                provenance=provenance,
                confidence=cand["confidence"],
                session_id=sid,
                task_id=task_id,
            )
            report["atoms"] += 1
            target = verdict["target_id"]
            if verdict["action"] in ("update", "delete") and target:
                mem.graph.add_edge(
                    MemoryEdge(
                        source_id=atom.id,
                        target_id=target,
                        relation=RelationType.supersedes,
                        reason=verdict["reason"] or "session distillation",
                    )
                )
                report["edges"] += 1
            elif verdict["relation"] == "contradicts" and target:
                mem.graph.add_edge(
                    MemoryEdge(
                        source_id=atom.id,
                        target_id=target,
                        relation=RelationType.contradicts,
                        reason=verdict["reason"] or "session distillation",
                    )
                )
                report["edges"] += 1
                report["contradictions"] += 1
            elif target and similar:
                # new information, linked to its neighborhood
                mem.graph.add_edge(
                    MemoryEdge(
                        source_id=atom.id,
                        target_id=target,
                        relation=RelationType.related_to,
                        reason=verdict["reason"] or "same topic",
                    )
                )
                report["edges"] += 1
    return report
