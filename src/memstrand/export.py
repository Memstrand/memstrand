"""Export / import: a full-fidelity snapshot of one `Memory`.

Used for backup/restore, for migrating into a fresh `MEMSTRAND_HOME`
(e.g. after the embedding model changes — vectors are never trusted on
import, everything gets re-embedded), and for a human-readable Markdown
view for Obsidian.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from .schema import MemoryAtom, MemoryEdge

FORMAT_VERSION = "0.2"

_PAGE = 1000


def _list_all_atoms(mem) -> list:
    atoms = []
    offset = 0
    while True:
        batch = mem.graph.list_atoms(limit=_PAGE, offset=offset)
        if not batch:
            break
        atoms.extend(batch)
        offset += len(batch)
        if len(batch) < _PAGE:
            break
    return atoms


def to_jsonl(mem) -> str:
    """One JSON object per line, `"type"` in {meta, atom, edge, turn,
    ledger, distill_state}. Vectors are deliberately NOT included —
    import re-embeds.

    `distill_state` (fix #9) round-trips each session's incremental-
    distillation watermark (session_id, last_turn_id) — without it, a
    restored backup would re-distill every session from scratch on the
    next consolidate() run (duplicate LLM calls; de-duped by content-hash
    id, so not data-corrupting, but wasteful and user-visibly slow)."""
    lines = [
        json.dumps(
            {
                "type": "meta",
                "version": FORMAT_VERSION,
                "embed_backend": mem.settings.embed_backend,
                "embed_model": mem.settings.embed_model,
                "exported_at": datetime.now(UTC).isoformat(),
            }
        )
    ]
    for atom in _list_all_atoms(mem):
        lines.append(json.dumps({"type": "atom", **atom.model_dump(mode="json")}))
    for edge in mem.graph.all_edges():
        lines.append(json.dumps({"type": "edge", **edge.model_dump(mode="json")}))
    for session_id in mem.sessions.list_sessions():
        for turn in mem.sessions.get_session(session_id):
            lines.append(json.dumps({"type": "turn", "session_id": session_id, **turn}))
    for entry in mem.sessions.dump_ledger():
        lines.append(json.dumps({"type": "ledger", **entry}))
    for state in mem.sessions.dump_distill_state():
        lines.append(json.dumps({"type": "distill_state", **state}))
    return "\n".join(lines) + "\n"


def to_markdown(mem) -> str:
    """Grouped by category, readable in Obsidian: scope/source/confidence
    ride along as an italic footer on each bullet."""
    by_category: dict[str, list] = {}
    for atom in _list_all_atoms(mem):
        by_category.setdefault(atom.category.value, []).append(atom)

    lines = ["# memstrand export", ""]
    for category in sorted(by_category):
        lines.append(f"## {category}")
        lines.append("")
        for atom in sorted(by_category[category], key=lambda a: a.created_at, reverse=True):
            scope = atom.task_id or "global"
            bits = [f"scope: {scope}", f"confidence: {atom.confidence:.2f}", f"provenance: {atom.provenance.value}"]
            if atom.source:
                bits.append(f"source: {atom.source}")
            lines.append(f"- {atom.content}  _({', '.join(bits)})_")
        lines.append("")
    return "\n".join(lines)


def from_jsonl(mem, text: str) -> dict:
    """Re-embeds every atom (never trusts stored vectors). Idempotent:
    atoms/edges upsert by id; turns/ledger entries are skipped if an
    identical (session, ts, content) / (atom_id, action, detail, ts)
    row already exists, so re-running the same export twice is a no-op
    the second time.

    Atom id collisions (fix #8c): an atom id is a content-hash of
    (content, task_id, host), so two atoms legitimately share an id only
    when they're the same memory. If an incoming atom's id already exists
    locally with DIFFERENT content, that's not your own backup being
    restored — it's either state corruption or (worst case) a crafted
    import trying to silently overwrite a trusted memory's text under an
    id it doesn't own. Such atoms are skipped and reported in
    `report["errors"]`, never overwritten. Identical content under the
    same id is a true idempotent no-op (metadata such as confidence/
    access_count naturally gets the incoming values via the normal
    upsert, which is harmless since the content — the only thing anyone
    actually reads — did not change).

    Import edge trust gap: a crafted JSONL can include an `edge` line
    whose endpoints are atom ids that were never legitimately part of
    this import (e.g. ids that happen to already exist locally) — that
    lets one planted `supersedes` edge silently hide an existing, trusted
    atom by pointing an untrusted/attacker-controlled "head" at it,
    without the import ever having vouched for either endpoint. An
    incoming edge is only accepted if BOTH its endpoints are atoms this
    *same* import batch actually created or matched identically (i.e.
    counted in `report["atoms"]` above — collided-with-different-content
    atoms, skipped above, do NOT count), or the identical edge
    (source_id, target_id, relation) already existed locally before this
    import started (so restoring your own full backup — where every atom
    line is reprocessed and every edge's endpoints are therefore always
    in-batch — round-trips every edge as before). Otherwise the edge is
    skipped and reported in `report["errors"]`."""
    report: dict = {"atoms": 0, "edges": 0, "turns": 0, "ledger": 0, "distill_state": 0, "errors": []}
    turns_seen: dict[str, set] = {}
    ledger_seen: set | None = None
    batch_atom_ids: set[str] = set()
    existing_edges: set[tuple[str, str, str]] | None = None

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            kind = obj.get("type")
            if kind == "meta":
                continue
            if kind == "atom":
                atom = MemoryAtom(**{k: v for k, v in obj.items() if k != "type"})
                existing = mem.graph.get_atom(atom.id)
                if existing is not None and existing.content != atom.content:
                    report["errors"].append(
                        {"line": lineno, "error": f"id collision with different content: {atom.id}"}
                    )
                    continue
                embedding = mem._embed(atom.content)
                mem.vectors.upsert(atom, embedding)
                mem.graph.add_atom(atom)
                mem.keyword.add(atom)
                batch_atom_ids.add(atom.id)
                report["atoms"] += 1
            elif kind == "edge":
                edge = MemoryEdge(**{k: v for k, v in obj.items() if k != "type"})
                if existing_edges is None:
                    existing_edges = {
                        (e.source_id, e.target_id, e.relation.value if hasattr(e.relation, "value") else e.relation)
                        for e in mem.graph.all_edges()
                    }
                edge_key = (
                    edge.source_id,
                    edge.target_id,
                    edge.relation.value if hasattr(edge.relation, "value") else edge.relation,
                )
                in_batch = edge.source_id in batch_atom_ids and edge.target_id in batch_atom_ids
                # NOTE (round 3): there is deliberately no third "both
                # endpoints already exist locally" acceptance path here —
                # today's format is a full-fidelity snapshot, so a
                # legitimate export always re-sends every atom it
                # references, meaning "in_batch" already covers that case.
                # A future incremental/partial export (one that omits
                # unchanged atoms and sends only new/changed atoms + edges)
                # would need such a path, since a new edge could then
                # legitimately point at an old, pre-existing, not-in-batch
                # atom on the importing side — see
                # /tmp/review_core/round3_partial_export.py, which shows
                # that scenario being silently (and, for the CURRENT
                # full-snapshot format, correctly) rejected today.
                if not (in_batch or edge_key in existing_edges):
                    report["errors"].append(
                        {"line": lineno, "error": "edge references atoms outside this import"}
                    )
                    continue
                mem.graph.add_edge(edge)
                report["edges"] += 1
            elif kind == "turn":
                sid = obj["session_id"]
                if sid not in turns_seen:
                    turns_seen[sid] = {
                        (t["ts"], t["role"], t["content"]) for t in mem.sessions.get_session(sid)
                    }
                key = (obj.get("ts"), obj["role"], obj["content"])
                if key in turns_seen[sid]:
                    continue
                mem.sessions.append_turn(
                    sid, obj["role"], obj["content"], task_id=obj.get("task_id"), ts=obj.get("ts")
                )
                turns_seen[sid].add(key)
                report["turns"] += 1
            elif kind == "ledger":
                if ledger_seen is None:
                    ledger_seen = {
                        (e["atom_id"], e["action"], e["detail"], e["ts"]) for e in mem.sessions.dump_ledger()
                    }
                key = (obj["atom_id"], obj["action"], obj.get("detail", ""), obj.get("ts"))
                if key in ledger_seen:
                    continue
                mem.sessions.log_provenance(
                    obj["atom_id"], obj["action"], obj.get("detail", ""), ts=obj.get("ts")
                )
                ledger_seen.add(key)
                report["ledger"] += 1
            elif kind == "distill_state":
                # max() semantics, not overwrite: importing an older
                # export after a newer local consolidate() run must never
                # rewind a session's distillation watermark backwards (an
                # idempotent no-op in the common restore-your-own-backup
                # case, since the two values are then equal).
                sid = obj["session_id"]
                incoming = int(obj["last_turn_id"])
                current = mem.sessions.last_distilled_turn_id(sid)
                if incoming > current:
                    mem.sessions.mark_distilled(sid, incoming)
                report["distill_state"] += 1
        except Exception as e:  # noqa: BLE001 - one bad line must never abort import
            report["errors"].append({"line": lineno, "error": str(e)})
    return report
