"""LLM token accounting: every chat call's reported usage, in one ledger.

`chat_completion` calls `record()` with the `usage` block the endpoint
returns (OpenAI shape: prompt_tokens / completion_tokens). Each record is
appended to `{MEMSTRAND_HOME}/llm_usage.jsonl` (one JSON object per line)
and, if a `collect()` block is active on this thread, added to its totals.

What a call is FOR, and which session it served, ride along via context
variables (`purpose()` / `for_session()`) so call sites deep in the
pipeline don't need extra parameters. Context variables don't cross into
new threads, so code that spawns one (background distillation) sets them
inside the thread.

Accounting must never break the call it accounts for: every failure here
is swallowed.
"""

from __future__ import annotations

import contextvars
import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

_purpose: contextvars.ContextVar[str] = contextvars.ContextVar("memstrand_llm_purpose", default="chat")
_session: contextvars.ContextVar[str | None] = contextvars.ContextVar("memstrand_llm_session", default=None)
_totals: contextvars.ContextVar[dict | None] = contextvars.ContextVar("memstrand_llm_totals", default=None)
_ledger_lock = threading.Lock()

LEDGER_NAME = "llm_usage.jsonl"


def empty_totals() -> dict:
    return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


@contextmanager
def purpose(name: str) -> Iterator[None]:
    token = _purpose.set(name)
    try:
        yield
    finally:
        _purpose.reset(token)


@contextmanager
def for_session(session_id: str | None) -> Iterator[None]:
    token = _session.set(session_id)
    try:
        yield
    finally:
        _session.reset(token)


@contextmanager
def collect() -> Iterator[dict]:
    """Totals of every call recorded inside the block (this thread)."""
    totals = empty_totals()
    token = _totals.set(totals)
    try:
        yield totals
    finally:
        _totals.reset(token)


def record(model: str, usage: dict | None) -> None:
    try:
        prompt = int((usage or {}).get("prompt_tokens") or 0)
        completion = int((usage or {}).get("completion_tokens") or 0)
        totals = _totals.get()
        if totals is not None:
            totals["calls"] += 1
            totals["prompt_tokens"] += prompt
            totals["completion_tokens"] += completion
        from ..config import ensure_private_dir, get_settings, secure_file

        path = get_settings().home / LEDGER_NAME
        ensure_private_dir(path.parent)
        line = json.dumps(
            {
                "ts": time.time(), "model": model, "purpose": _purpose.get(),
                "session": _session.get(), "prompt_tokens": prompt,
                "completion_tokens": completion, "reported": usage is not None,
            }
        )
        with _ledger_lock, path.open("a") as f:
            f.write(line + "\n")
        secure_file(path)
    except Exception:  # noqa: BLE001, S110 - accounting must never break the LLM call
        pass


def read_ledger(since: float | None = None) -> list[dict]:
    from ..config import get_settings

    path = get_settings().home / LEDGER_NAME
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and (since is None or row.get("ts", 0) >= since):
                rows.append(row)
    return rows


def summarize(rows: list[dict], by: str = "purpose") -> dict[str, dict]:
    """Totals grouped by "purpose", "session", "model", or "day"."""
    out: dict[str, dict] = {}
    for row in rows:
        if by == "day":
            key = time.strftime("%Y-%m-%d", time.localtime(row.get("ts", 0)))
        else:
            key = str(row.get(by) or "-")
        bucket = out.setdefault(key, empty_totals())
        bucket["calls"] += 1
        bucket["prompt_tokens"] += int(row.get("prompt_tokens") or 0)
        bucket["completion_tokens"] += int(row.get("completion_tokens") or 0)
    return out
