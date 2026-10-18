"""Tests for memstrand.hooks: Claude Code transcript parsing and the two
hook bodies. No network, no real Memory — fake objects stand in for `mem`
so these stay fast and exercise the contract hooks.py codes against:
mem.log_turn / mem.log_turns / mem.improve / mem.context.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from memstrand import hooks

# -- fakes ------------------------------------------------------------------------


class FakeMem:
    def __init__(self):
        self.turns: list[tuple[str, str, str, str | None]] = []
        self.improve_calls: list[tuple[list[str], bool]] = []

    def log_turn(self, session_id, role, content, task_id=None):
        self.turns.append((session_id, role, content, task_id))

    def improve(self, session_ids, background=False):
        self.improve_calls.append((session_ids, background))
        return {"queued": session_ids}

    def context(self, query, task_id=None, limit=8):
        return f"ctx:{query}:{task_id}:{limit}"


class BatchMem:
    """A remote-like mem exposing the batch `log_turns` form."""

    def __init__(self):
        self.batches: list[list[dict]] = []
        self.improve_calls: list[tuple[list[str], bool]] = []

    def log_turns(self, turns):
        self.batches.append(turns)

    def log_turn(self, *a, **k):  # pragma: no cover - must not be used when log_turns exists
        raise AssertionError("log_turn called despite log_turns being available")

    def improve(self, session_ids, background=False):
        self.improve_calls.append((session_ids, background))


class BrokenMem:
    """Every call raises — used to prove hooks never propagate exceptions."""

    def log_turn(self, *a, **k):
        raise RuntimeError("boom")

    def improve(self, *a, **k):
        raise RuntimeError("boom")

    def context(self, *a, **k):
        raise RuntimeError("boom")


def _entry(etype: str, role: str, content, is_meta: bool = False) -> dict:
    e = {"type": etype, "message": {"role": role, "content": content}}
    if is_meta:
        e["isMeta"] = True
    return e


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    with path.open("a") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    """Every test gets its own hook_state.json / hook.log location."""
    monkeypatch.setenv("MEMSTRAND_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("MEMSTRAND_TASK", raising=False)
    return tmp_path / "home"


# -- transcript parsing / incremental logging --------------------------------------


def test_claude_code_hook_logs_new_turns_only(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    _write_jsonl(
        transcript,
        [
            _entry("user", "user", "hello there"),
            _entry("assistant", "assistant", [{"type": "text", "text": "hi!"}]),
            _entry("assistant", "assistant", [{"type": "tool_use", "id": "1", "name": "Bash", "input": {}}]),
            _entry("user", "user", [{"type": "tool_result", "tool_use_id": "1", "content": "ok"}]),
            _entry("assistant", "assistant", [{"type": "thinking", "thinking": "hmm"}]),
            {"type": "user", "isMeta": True, "message": {"role": "user", "content": "meta stuff"}},
        ],
    )

    mem = FakeMem()
    stdin_json = {"session_id": "sess1", "transcript_path": str(transcript), "cwd": str(tmp_path)}

    result = hooks.claude_code_hook(stdin_json, mem, task_from="none", improve=True)

    assert result["logged"] == 2
    assert [(t[1], t[2]) for t in mem.turns] == [("user", "hello there"), ("assistant", "hi!")]
    assert all(t[0] == "claude-code:sess1" for t in mem.turns)
    assert mem.improve_calls == [(["claude-code:sess1"], True)]

    # second run before anything new is appended: no new turns, no re-improve
    result2 = hooks.claude_code_hook(stdin_json, mem, task_from="none", improve=True)
    assert result2["logged"] == 0
    assert len(mem.turns) == 2
    assert len(mem.improve_calls) == 1

    # append one more real turn -> only the new one gets logged this time
    _write_jsonl(transcript, [_entry("user", "user", "one more thing")])
    result3 = hooks.claude_code_hook(stdin_json, mem, task_from="none", improve=True)
    assert result3["logged"] == 1
    assert mem.turns[-1][1:3] == ("user", "one more thing")
    assert len(mem.improve_calls) == 2


def test_claude_code_hook_no_improve_when_flag_off(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    _write_jsonl(transcript, [_entry("user", "user", "hello")])
    mem = FakeMem()
    stdin_json = {"session_id": "sess-no-improve", "transcript_path": str(transcript), "cwd": str(tmp_path)}

    result = hooks.claude_code_hook(stdin_json, mem, task_from="none", improve=False)

    assert result["logged"] == 1
    assert mem.improve_calls == []


def test_claude_code_hook_missing_transcript_is_quiet(tmp_path):
    stdin_json = {
        "session_id": "s3",
        "transcript_path": str(tmp_path / "missing.jsonl"),
        "cwd": str(tmp_path),
    }
    result = hooks.claude_code_hook(stdin_json, FakeMem(), task_from="none")
    assert result == {"logged": 0}


def test_claude_code_hook_batches_when_log_turns_available(tmp_path):
    transcript = tmp_path / "t.jsonl"
    _write_jsonl(transcript, [_entry("user", "user", "hello"), _entry("assistant", "assistant", "hi")])
    mem = BatchMem()
    stdin_json = {"session_id": "s4", "transcript_path": str(transcript), "cwd": str(tmp_path)}

    result = hooks.claude_code_hook(stdin_json, mem, task_from="none")

    assert result["logged"] == 2
    assert len(mem.batches) == 1
    assert mem.batches[0][0] == {
        "session_id": "claude-code:s4",
        "role": "user",
        "content": "hello",
        "task_id": None,
    }
    assert mem.improve_calls == [(["claude-code:s4"], True)]


# -- exceptions never propagate -----------------------------------------------------


def test_claude_code_hook_never_raises_and_logs_error(tmp_path):
    transcript = tmp_path / "t.jsonl"
    _write_jsonl(transcript, [_entry("user", "user", "hi")])
    stdin_json = {"session_id": "s2", "transcript_path": str(transcript), "cwd": str(tmp_path)}

    result = hooks.claude_code_hook(stdin_json, BrokenMem(), task_from="none", improve=True)

    assert "error" in result
    log_path = hooks.get_settings().home / "hook.log"
    assert log_path.exists()
    assert "boom" in log_path.read_text()


def test_claude_code_context_hook_empty_on_failure():
    result = hooks.claude_code_context_hook({"prompt": "q"}, BrokenMem(), task_from="none")
    assert result == ""


# -- private file/dir permissions -----------------------------------------------------


def test_hook_log_and_state_file_are_private(tmp_path):
    """`hook.log` and `hook_state.json` (and their parent, MEMSTRAND_HOME)
    must be created with restrictive permissions: 0700 for the directory,
    0600 for the files. These can contain conversation content, so they
    must not be world/group readable."""
    transcript = tmp_path / "t.jsonl"
    _write_jsonl(transcript, [_entry("user", "user", "hi")])
    stdin_json = {"session_id": "s-priv", "transcript_path": str(transcript), "cwd": str(tmp_path)}

    # state file, written on a normal successful run
    hooks.claude_code_hook(stdin_json, FakeMem(), task_from="none")
    state_path = hooks.get_settings().home / "hook_state.json"
    assert state_path.exists()
    assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(state_path.parent.stat().st_mode) == 0o700

    # log file, written when mem calls raise -- append a fresh turn so
    # there's something new for BrokenMem.log_turn to choke on (the state
    # file above already advanced past the existing entries).
    _write_jsonl(transcript, [_entry("assistant", "assistant", "boom time")])
    hooks.claude_code_hook(stdin_json, BrokenMem(), task_from="none")
    log_path = hooks.get_settings().home / "hook.log"
    assert log_path.exists()
    assert stat.S_IMODE(log_path.stat().st_mode) == 0o600


# -- task_id resolution ---------------------------------------------------------------


def test_resolve_task_id_variants(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "sub" / "dir").mkdir(parents=True)
    (repo / ".git").mkdir()

    assert hooks._resolve_task_id("cwd", str(repo / "sub" / "dir")) == "repo"
    assert hooks._resolve_task_id("none", str(repo)) is None

    monkeypatch.setenv("MEMSTRAND_TASK", "env-task")
    assert hooks._resolve_task_id("env", str(repo)) == "env-task"

    no_git = tmp_path / "standalone"
    no_git.mkdir()
    assert hooks._resolve_task_id("cwd", str(no_git)) == "standalone"


# -- context hook --------------------------------------------------------------------


def test_claude_code_context_hook_returns_context():
    mem = FakeMem()
    stdin_json = {"prompt": "what is x", "cwd": "/doesnotexist"}
    result = hooks.claude_code_context_hook(stdin_json, mem, task_from="none", limit=5)
    assert result == "ctx:what is x:None:5"


def test_claude_code_context_hook_session_start_has_no_prompt():
    mem = FakeMem()
    stdin_json = {"cwd": "/doesnotexist"}
    result = hooks.claude_code_context_hook(stdin_json, mem, task_from="none")
    assert result == "ctx:None:None:8"


# -- settings.json snippet ------------------------------------------------------------


def test_claude_code_settings_snippet_is_valid_json_with_expected_shape():
    data = json.loads(hooks.CLAUDE_CODE_SETTINGS_SNIPPET)
    assert data["hooks"]["Stop"][0]["hooks"][0]["command"] == "memstrand hook claude-code"
    assert data["hooks"]["PreCompact"][0]["hooks"][0]["command"] == "memstrand hook claude-code"
    assert (
        data["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"] == "memstrand hook claude-code-context"
    )
    for event_hooks in data["hooks"].values():
        for matcher_block in event_hooks:
            for h in matcher_block["hooks"]:
                assert h["type"] == "command"
