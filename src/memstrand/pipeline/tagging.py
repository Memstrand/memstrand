"""Retroactive tagging: give existing memories rich topic tags.

New memories get tags at extraction time (`extract.py`). Memories created
before that — or tagged poorly — are re-tagged here, with the SOURCE
CONVERSATION as context: an atomic memory like "run multi-tenant changes
one tenant at a time" doesn't say it came out of a GCP new-region build or
an IPv6 reprovisioning; its session does. One LLM call per session sees the
transcript plus that session's memories (numbered) and returns tags per
memory. Memories with no source session are tagged from their text alone,
in batches.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ..schema import normalize_tags
from . import usage
from .extract import ExtractionFailed, chat_completion

if TYPE_CHECKING:
    from ..api import Memory

TAG_SYSTEM_PROMPT = """\
You assign topic tags to memories that were extracted from a conversation.
Each memory is one atomic sentence; the sentence alone lost the context it
came from. Use the conversation to restore it, so that someone searching
later by system, by initiative, or by concept finds the memory.

Step 1 — read the whole conversation and name its 1-3 main initiatives:
what work was being done, as reusable kebab-case terms (e.g.
new-region-build, ipv6-reprovisioning, incident-response, cost-cleanup,
ci-pipeline, tenant-migration).

Step 2 — give EACH memory 3-6 lowercase-kebab-case tags covering ALL of:
(1) at least one system/technology it involves (e.g. gcp, aws, oci, ipv6,
    terraform, k8s, lambda, postgres),
(2) at least one initiative/activity it came out of — usually one of the
    step-1 initiatives, even when the memory's own words don't mention it,
(3) at least one domain concept (e.g. multi-tenant, prod, rate-limiting,
    auth, networking, resource-contention, rollback).
Tags are reusable categories, not sentence fragments: "deployment-manager"
yes, "multi-field-updates" no; "resumability" yes, "resume-logic" no. No
ticket numbers, IDs, people, or one-off names, and not the repo name.
Reuse terms from the existing-tags list where they fit instead of
inventing synonyms.

Output ONLY a JSON object:
{"initiatives": ["new-region-build", ...],
 "tags": {"1": ["gcp", "new-region-build", "multi-tenant"], "2": [...]}}\
"""

# Haiku 4.5 has a 200K-token context; keep the transcript well under it.
MAX_TRANSCRIPT_CHARS = 450_000
UNSOURCED_BATCH = 40
VOCABULARY_SIZE = 80


def _vocabulary(mem: Memory) -> list[str]:
    return [t for t, _ in mem.graph.tag_counts(limit=VOCABULARY_SIZE)]


def _render_transcript(turns: list[dict]) -> str:
    text = "\n".join(f"{t.get('role', 'user').upper()}: {t.get('content', '')}" for t in turns)
    if len(text) <= MAX_TRANSCRIPT_CHARS:
        return text
    head = text[: MAX_TRANSCRIPT_CHARS // 2]
    tail = text[-MAX_TRANSCRIPT_CHARS // 2 :]
    return f"{head}\n[... middle of a long conversation omitted ...]\n{tail}"


def _parse_tag_map(raw: str, count: int) -> dict[int, list[str]]:
    cleaned = raw.strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ExtractionFailed(f"tagger did not return a JSON object: {raw[:120]!r}")
    try:
        obj = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as e:
        raise ExtractionFailed(f"tagger returned invalid JSON: {raw[:120]!r}") from e
    if isinstance(obj, dict) and isinstance(obj.get("tags"), dict):
        obj = obj["tags"]  # {"initiatives": [...], "tags": {...}}; also accept a bare map
    out: dict[int, list[str]] = {}
    for key, tags in obj.items() if isinstance(obj, dict) else ():
        try:
            n = int(str(key).strip().rstrip("."))
        except ValueError:
            continue
        if 1 <= n <= count and isinstance(tags, list):
            out[n] = normalize_tags(tags)
    return out


def _tag_call(mem: Memory, atoms: list, transcript: str | None) -> dict[int, list[str]]:
    from ..scopes import short_name

    listing = "\n".join(
        f"{i}. [{a.category.value}] ({', '.join(short_name(s) for s in a.scopes)}) {a.content}"
        for i, a in enumerate(atoms, start=1)
    )
    parts = []
    if transcript:
        parts.append(f"<transcript>\n{transcript}\n</transcript>")
    parts.append(f"<memories>\n{listing}\n</memories>")
    vocabulary = _vocabulary(mem)
    if vocabulary:
        parts.append(f"Existing tags (reuse where they fit): {', '.join(vocabulary)}")
    parts.append("Output ONLY the JSON object of tags per memory number. Do not reply to the conversation.")
    with usage.purpose("tag"):
        raw = chat_completion(
            mem.llm_base_url,
            mem.llm_model,
            [{"role": "system", "content": TAG_SYSTEM_PROMPT}, {"role": "user", "content": "\n\n".join(parts)}],
            timeout=180.0,
            api_key=mem.llm_api_key,
        )
    return _parse_tag_map(raw, len(atoms))


def retag(mem: Memory, atom_ids: list[str], session_id: str | None = None) -> dict:
    """Re-tag `atom_ids`. With `session_id`, the session's logged transcript
    is the context (one call); without, memories are tagged from their text
    alone, in batches. Replaces each atom's tags. Never raises for LLM or
    parse failures — they're reported in `errors` and those atoms keep
    their current tags."""
    report = {"session": session_id, "atoms": 0, "tagged": 0, "errors": [],
              "usage": usage.empty_totals()}
    atoms = [a for a in mem.graph.get_atoms(atom_ids).values()]
    atoms.sort(key=lambda a: a.created_at)
    report["atoms"] = len(atoms)
    if not atoms:
        return report
    transcript = None
    if session_id:
        turns = mem.sessions.get_session(session_id)
        transcript = _render_transcript(turns) if turns else None
    batches = [atoms] if transcript else [atoms[i : i + UNSOURCED_BATCH] for i in range(0, len(atoms), UNSOURCED_BATCH)]
    with usage.collect() as totals, usage.for_session(session_id):
        for batch in batches:
            try:
                tag_map = _tag_call(mem, batch, transcript)
            except Exception as e:  # noqa: BLE001 - report, keep going
                report["errors"].append({"session": session_id, "error": str(e)})
                continue
            for i, atom in enumerate(batch, start=1):
                tags = tag_map.get(i)
                if tags:
                    mem.set_tags(atom.id, tags=tags)
                    report["tagged"] += 1
    report["usage"] = dict(totals)
    return report
