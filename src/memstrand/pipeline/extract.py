"""ECL-lite EXTRACT: session turns -> typed MemoryAtoms.

The 13-category schema IS the extraction contract: the local LLM is
instructed to emit only valid Category/Provenance values, with a
confidence per atom. Resolution follows Mem0's pattern:
ADD / UPDATE / DELETE / NONE per candidate atom (resolved in distill.py).

Every LLM call goes to a local OpenAI-compatible endpoint. Zero keys.
"""

from __future__ import annotations

import json
import re

import httpx

from ..schema import Category, Provenance

EXTRACTION_SYSTEM_PROMPT = """\
You extract durable memories from a conversation session.
Emit a JSON list. Each item: {"category": <one of the 13 categories>,
"content": "<atomic fact>", "provenance": "<how learned>",
"confidence": <0.0-1.0>}.
Valid categories: instruction, fact, decision, goal, commitment,
preference, relationship, context, event, learning, observation,
artifact, error.
Valid provenance: explicit_statement, inferred, corrected, validated,
observed, imported.
One atomic fact per item. Skip chit-chat and throwaway context.
Prefer high-signal items. Prefer explicit_statement when the user said
it directly, observed when it came from tool output, inferred otherwise.
Keep confidence honest: 0.9+ only for directly stated facts.\
"""


class LLMUnavailable(Exception):
    """The local LLM endpoint didn't answer. Nothing was written."""


def chat_completion(
    base_url: str, model: str, messages: list[dict], timeout: float = 120.0
) -> str:
    """One blocking chat call against a local OpenAI-compatible endpoint."""
    try:
        resp = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": "Bearer <redacted>"},
            json={"model": model, "messages": messages, "temperature": 0.1},
            timeout=timeout,
        )
        resp.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        raise LLMUnavailable(f"local LLM at {base_url} unreachable: {e}") from e
    try:
        return resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as e:
        raise LLMUnavailable(f"malformed LLM response from {base_url}: {e}") from e


def _extract_json_array(text: str) -> list:
    """Pull the first JSON array out of free-form model output."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON array in model output: {text[:200]!r}")
    return json.loads(cleaned[start : end + 1])


def _valid_atom(item: dict) -> dict | None:
    """Schema-contract check. Drops (never invents) malformed candidates."""
    try:
        category = Category(item["category"])
        provenance = Provenance(item.get("provenance", "inferred"))
        content = str(item["content"]).strip()
        if not content:
            return None
        confidence = float(item.get("confidence", 0.7))
        confidence = max(0.0, min(1.0, confidence))
    except (KeyError, ValueError, TypeError):
        return None
    return {
        "category": category,
        "content": content,
        "provenance": provenance,
        "confidence": confidence,
    }


def extract_atoms(
    session_turns: list[dict],
    llm_base_url: str,
    llm_model: str,
    timeout: float = 180.0,
) -> list[dict]:
    """Turns -> validated candidate atom dicts. LLM call goes here."""
    transcript = "\n".join(
        f"{t.get('role', 'user').upper()}: {t.get('content', '')}" for t in session_turns
    )
    if not transcript.strip():
        return []
    raw = chat_completion(
        llm_base_url,
        llm_model,
        [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Session transcript:\n{transcript}"},
        ],
        timeout=timeout,
    )
    try:
        items = _extract_json_array(raw)
    except (ValueError, json.JSONDecodeError):
        return []  # unparseable extraction = no atoms, not a crash
    atoms = []
    for item in items:
        if isinstance(item, dict):
            valid = _valid_atom(item)
            if valid:
                atoms.append(valid)
    return atoms
