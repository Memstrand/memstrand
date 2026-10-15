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
    ledger}. Vectors are deliberately NOT included — import re-embeds."""
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
    the second time."""
    report: dict = {"atoms": 0, "edges": 0, "turns": 0, "ledger": 0, "errors": []}
    turns_seen: dict[str, set] = {}
    ledger_seen: set | None = None

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
                mem.vectors.upsert(atom, mem._embed(atom.content))
                mem.graph.add_atom(atom)
                mem.keyword.add(atom)
                report["atoms"] += 1
            elif kind == "edge":
                edge = MemoryEdge(**{k: v for k, v in obj.items() if k != "type"})
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
        except Exception as e:  # noqa: BLE001 - one bad line must never abort import
            report["errors"].append({"line": lineno, "error": str(e)})
    return report
