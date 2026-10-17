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
"confidence": <0.0-1.0>, "scope": "global" | "task"}.
Valid categories: instruction, fact, decision, goal, commitment,
preference, relationship, context, event, learning, observation,
artifact, error.
Valid provenance: explicit_statement, inferred, corrected, validated,
observed, imported.
Scope rule: durable facts about the user themself (preferences,
instructions, relationships, personal facts) are "global" — true no
matter what project/task is active. Things only meaningful inside the
current project/task are "task".
One atomic fact per item. Skip chit-chat and throwaway context.
Prefer high-signal items. Prefer explicit_statement when the user said
it directly, observed when it came from tool output, inferred otherwise.
Keep confidence honest: 0.9+ only for directly stated facts.\
"""

# Categories that default to global scope when the model omits "scope".
_GLOBAL_DEFAULT_CATEGORIES = {Category.preference, Category.instruction, Category.relationship}


class LLMUnavailable(Exception):
    """The local LLM endpoint didn't answer, or rejected the request.
    Nothing was written either way.

    `transient` (default `True`) distinguishes the two: a down endpoint,
    a timeout, or a 5xx means retrying later might work; `transient=False`
    means the server answered but rejected the request itself (bad model,
    bad key, malformed body, ...) — retrying the same request won't help,
    and the message carries the real reason instead of a generic
    "unreachable"."""

    def __init__(self, message: str, transient: bool = True):
        super().__init__(message)
        self.transient = transient


def _error_detail(resp: httpx.Response) -> str:
    """Pull the most useful one-line explanation out of an HTTP error
    response: an OpenAI-style {"error": {"message": ...}} body if present,
    otherwise the raw (truncated) response text."""
    try:
        data = resp.json()
        message = data.get("error")
        if isinstance(message, dict):
            message = message.get("message")
        if message:
            return str(message)
    except (ValueError, AttributeError):
        pass
    return resp.text[:200]


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict],
    timeout: float = 120.0,
    api_key: str = "local",
) -> str:
    """One blocking chat call against a local OpenAI-compatible endpoint."""
    try:
        resp = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": messages, "temperature": 0.1},
            timeout=timeout,
        )
        resp.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        raise LLMUnavailable(f"local LLM at {base_url} unreachable: {e}") from e
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        detail = _error_detail(e.response)
        if status >= 500:
            # the endpoint answered but is having a bad time — transient.
            raise LLMUnavailable(
                f"local LLM at {base_url} returned a server error ({status}): {detail}"
            ) from e
        # 4xx: the request itself was rejected (bad model, bad key, bad
        # body, ...) — retrying the identical request will not help.
        raise LLMUnavailable(
            f"LLM rejected request ({status}): {detail}", transient=False
        ) from e
    except (httpx.HTTPError, httpx.InvalidURL) as e:
        # Everything else httpx can raise before a response exists —
        # InvalidURL (a plain Exception, NOT an HTTPError subclass) from a
        # malformed proxy env, ProxyError, UnsupportedProtocol, ... — is
        # environmental, not a verdict on the endpoint. Transient: a later
        # attempt (or a fixed environment) may succeed. Without this,
        # _is_llm_reachable() let these escape raw and crashed
        # consolidate(), breaking its never-raises contract.
        raise LLMUnavailable(
            f"local LLM request failed ({type(e).__name__}): {e}", transient=True
        ) from e
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
    scope = item.get("scope")
    if scope not in ("global", "task"):
        scope = "global" if category in _GLOBAL_DEFAULT_CATEGORIES else "task"
    return {
        "category": category,
        "content": content,
        "provenance": provenance,
        "confidence": confidence,
        "scope": scope,
    }


def extract_atoms(
    session_turns: list[dict],
    llm_base_url: str,
    llm_model: str,
    timeout: float = 180.0,
    api_key: str = "local",
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
        api_key=api_key,
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
