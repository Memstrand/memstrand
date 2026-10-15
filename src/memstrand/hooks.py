"""Claude Code integration: turn its hook events into `Memory` calls.

Claude Code runs a hook command once per event and feeds it a JSON object
on stdin (`session_id`, `transcript_path`, `cwd`, `hook_event_name`, ...).
Two shapes are implemented here:

- `claude_code_hook` for Stop / SessionEnd / PreCompact: read whatever is
  new in the session's transcript since last time, log it as raw turns,
  then kick off a background `improve()` so it gets distilled into typed
  memory without blocking anything.
- `claude_code_context_hook` for UserPromptSubmit / SessionStart: return a
  context block for the CLI to print — Claude Code appends hook stdout to
  the model's context.

Both MUST NEVER fail the user's session: every exception is caught here
and written to `{home}/hook.log` instead of propagating. The `memstrand
hook ...` CLI commands add a second layer of the same guarantee (see
cli.py) in case something fails before `claude_code_hook` is even
reached (e.g. `get_memory()` itself raising).

`claude_code_context_hook` takes whatever `mem` it's given and queries it
directly — it does not decide whether to go local or remote. The `cli.py`
caller is the one that keeps this fast: it only ever passes a `mem` when
the daemon is already confirmed to be up, since opening a local `Memory`
(first-run fastembed model load, disk-backed Qdrant/SQLite) is too slow
for a hook that runs before every user prompt. When no daemon is running,
the CLI skips calling this function entirely and prints nothing.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from .config import get_settings

# -- error logging --------------------------------------------------------------


def log_error(message: str) -> None:
    """Best-effort error log for a hook that swallowed an exception. Never
    raises itself — a broken logger must not break the hook it's logging for."""
    try:
        path = get_settings().home / "hook.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(f"{datetime.now(UTC).isoformat()} {message}\n")
    except Exception:  # noqa: BLE001, S110 - logging must never itself raise
        pass


# -- per-session progress state --------------------------------------------------


def _state_path() -> Path:
    return get_settings().home / "hook_state.json"


def _load_state() -> dict:
    path = _state_path()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {"sessions": {}}
    return {"sessions": {}}


def _save_state(state: dict) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, path)


# -- transcript parsing -----------------------------------------------------------


def _read_transcript_entries(path: Path) -> list[dict]:
    """One JSON object per line. Malformed lines are skipped, never fatal."""
    entries: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                entries.append(obj)
    return entries


def _extract_text(message: dict) -> str | None:
    """`content` is either a plain string or a list of blocks; only
    `{"type": "text"}` blocks count — tool_use/tool_result/thinking are
    skipped. `None` (not "") means "nothing worth logging"."""
    content = message.get("content")
    if isinstance(content, str):
        text = content.strip()
        return text or None
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        text = "\n".join(p for p in parts if p).strip()
        return text or None
    return None


def _new_turns(entries: list[dict]) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    for entry in entries:
        if entry.get("isMeta"):
            continue
        if entry.get("type") not in ("user", "assistant"):
            continue
        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        text = _extract_text(message)
        if not text:
            continue
        role = message.get("role") or entry["type"]
        turns.append((role, text))
    return turns


# -- task scoping -----------------------------------------------------------------


def _git_root(cwd: str) -> Path | None:
    try:
        start = Path(cwd).expanduser().resolve()
    except OSError:
        return None
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _resolve_task_id(task_from: str, cwd: str) -> str | None:
    if task_from == "none":
        return None
    if task_from == "env":
        return get_settings().default_task
    # "cwd" (default): the enclosing git repo's name, else the cwd's own name.
    root = _git_root(cwd)
    if root is not None:
        return root.name
    try:
        return Path(cwd).expanduser().resolve().name
    except OSError:
        return None


# -- Stop / SessionEnd / PreCompact -----------------------------------------------


def claude_code_hook(
    stdin_json: dict, mem, task_from: str = "cwd", improve: bool = True
) -> dict:
    """Log any transcript turns new since the last call, then (if any were
    new) kick off `mem.improve(..., background=True)`. Returns a small
    report dict for callers/tests; on any failure returns {"error": ...}
    and logs to `{home}/hook.log` — it never raises."""
    try:
        session_id = stdin_json.get("session_id")
        transcript_path = stdin_json.get("transcript_path")
        cwd = stdin_json.get("cwd") or "."
        if not session_id or not transcript_path:
            return {"logged": 0}

        transcript = Path(transcript_path)
        if not transcript.exists():
            return {"logged": 0}

        entries = _read_transcript_entries(transcript)

        state = _load_state()
        sessions = state.setdefault("sessions", {})
        last_line = sessions.get(session_id, {}).get("last_line", 0)
        new_entries = entries[last_line:]
        turns = _new_turns(new_entries)

        task_id = _resolve_task_id(task_from, cwd)
        mem_session_id = f"claude-code:{session_id}"

        if turns:
            log_many = getattr(mem, "log_turns", None)
            if callable(log_many):
                log_many(
                    [
                        {
                            "session_id": mem_session_id,
                            "role": role,
                            "content": content,
                            "task_id": task_id,
                        }
                        for role, content in turns
                    ]
                )
            else:
                for role, content in turns:
                    mem.log_turn(mem_session_id, role, content, task_id=task_id)

        sessions[session_id] = {"transcript_path": str(transcript), "last_line": len(entries)}
        _save_state(state)

        if improve and turns:
            mem.improve([mem_session_id], background=True)

        return {"logged": len(turns), "session_id": mem_session_id, "task_id": task_id}
    except Exception as exc:  # noqa: BLE001 - a hook must never fail the user's session
        log_error(f"claude_code_hook: {exc}")
        return {"error": str(exc)}


# -- UserPromptSubmit / SessionStart -----------------------------------------------


def claude_code_context_hook(
    stdin_json: dict, mem, task_from: str = "cwd", limit: int = 8
) -> str:
    """Returns the context block for the CLI to print to stdout. `""` on
    any failure or when there's nothing to say — this hook must stay fast
    and must never fail the user's prompt."""
    try:
        query = stdin_json.get("prompt")  # UserPromptSubmit has it; SessionStart doesn't
        cwd = stdin_json.get("cwd") or "."
        task_id = _resolve_task_id(task_from, cwd)
        return mem.context(query, task_id=task_id, limit=limit) or ""
    except Exception as exc:  # noqa: BLE001 - a hook must never fail the user's session
        log_error(f"claude_code_context_hook: {exc}")
        return ""


# -- settings.json snippet ---------------------------------------------------------

_CLAUDE_CODE_HOOKS = {
    "hooks": {
        "Stop": [{"hooks": [{"type": "command", "command": "memstrand hook claude-code"}]}],
        "PreCompact": [{"hooks": [{"type": "command", "command": "memstrand hook claude-code"}]}],
        "UserPromptSubmit": [
            {"hooks": [{"type": "command", "command": "memstrand hook claude-code-context"}]}
        ],
    }
}

CLAUDE_CODE_SETTINGS_SNIPPET = json.dumps(_CLAUDE_CODE_HOOKS, indent=2)
"""Paste the `"hooks"` key into `~/.claude/settings.json`. Printed by
`memstrand hook print-config`. Schema: event name -> list of
`{"matcher"?: str, "hooks": [{"type": "command", "command": str, "timeout"?: int}]}`
— `matcher` is only meaningful for tool-scoped events (PreToolUse etc.)
and is omitted here since Stop/PreCompact/UserPromptSubmit fire regardless."""
