"""`memstrand hook backfill`: distill existing Claude Code transcripts."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_scoping_v2 import _make_mem

from memstrand import cli, hooks
from memstrand.pipeline import distill as distill_mod


def _entry(role: str, text: str, cwd: str = "/work/repo", **extra) -> dict:
    return {"type": role, "cwd": cwd, "message": {"role": role, "content": [{"type": "text", "text": text}]},
            **extra}


def _write(path: Path, entries: list[dict], mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in entries))
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


class RecordingMem:
    def __init__(self, fail_improve_after: int | None = None):
        self.turns: list[dict] = []
        self.improve_calls: list[tuple[list[str], bool]] = []
        self.fail_improve_after = fail_improve_after

    def log_turns(self, turns):
        self.turns.extend(turns)

    def improve(self, session_ids, background=False):
        self.improve_calls.append((session_ids, background))
        if self.fail_improve_after is not None and len(self.improve_calls) > self.fail_improve_after:
            raise RuntimeError("gateway down")
        return {"atoms": 1, "scope_reviews": 0, "errors": []}

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMSTRAND_HOME", str(tmp_path / "home"))


# -- discovery ------------------------------------------------------------------------------


def test_discover_layouts_filters_and_order(tmp_path):
    root = tmp_path / "projects"
    old, mid, new = time.time() - 30 * 86400, time.time() - 5 * 86400, time.time() - 3600
    a = _write(root / "-Users-me-projects-alpha" / "s-old.jsonl", [_entry("user", "a")], old)
    b = _write(root / "-Users-me-projects-beta" / "s-mid.jsonl", [_entry("user", "b")], mid)
    c = _write(root / "-Users-me-projects-alpha" / "s-new.jsonl", [_entry("user", "c")], new)
    _write(root / "-Users-me-projects-alpha" / "s-new" / "subagents" / "agent-1.jsonl", [_entry("user", "x")], new)
    live = _write(root / "-Users-me-projects-alpha" / "s-live.jsonl", [_entry("user", "live")])

    assert hooks.discover_transcripts([root]) == [a, b, c]  # oldest first; no subagents; live skipped
    assert hooks.discover_transcripts([root], skip_active_minutes=0)[-1] == live
    assert hooks.discover_transcripts([root], projects=["beta"]) == [b]
    assert hooks.discover_transcripts([root], since=datetime.fromtimestamp(mid - 60, UTC)) == [b, c]
    assert hooks.discover_transcripts([root], limit=2) == [a, b]
    assert hooks.discover_transcripts([root / "-Users-me-projects-alpha"]) == [a, c]  # project folder
    assert hooks.discover_transcripts([a, a]) == [a]  # a file, deduped


# -- backfill one transcript ----------------------------------------------------------------


def _transcript(tmp_path, n_pairs: int, **kw) -> Path:
    entries = []
    for i in range(n_pairs):
        entries.append(_entry("user", f"question {i}"))
        entries.append({"type": "assistant", "cwd": "/work/repo", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Read", "input": {"file_path": "/nope.py"}}]}})
        entries.append(_entry("assistant", f"answer {i}"))
    return _write(tmp_path / "proj" / "sess-1.jsonl", entries, **kw)


def test_backfill_chunks_and_distills_synchronously(tmp_path):
    path = _transcript(tmp_path, 5)  # 10 text turns
    mem = RecordingMem()
    r = hooks.backfill_transcript(mem, path, task_from="none", chunk=4)
    assert (r["turns"], r["chunks"], r["atoms"]) == (10, 3, 3)
    assert len(mem.turns) == 10
    assert mem.improve_calls == [(["claude-code:sess-1"], False)] * 3
    assert {t["session_id"] for t in mem.turns} == {"claude-code:sess-1"}
    assert hooks._load_state()["sessions"]["sess-1"]["last_line"] == 15

    again = hooks.backfill_transcript(mem, path, task_from="none", chunk=4)
    assert again["turns"] == 0 and len(mem.turns) == 10  # idempotent


def test_backfill_resumes_after_interruption_without_duplicates(tmp_path):
    path = _transcript(tmp_path, 5)
    mem = RecordingMem(fail_improve_after=1)
    with pytest.raises(RuntimeError):
        hooks.backfill_transcript(mem, path, task_from="none", chunk=4)
    assert len(mem.turns) == 8  # two chunks logged, second distill failed
    mem.fail_improve_after = None
    r = hooks.backfill_transcript(mem, path, task_from="none", chunk=4)
    assert r["turns"] == 2
    assert [t["content"] for t in mem.turns] == [
        f"{kind} {i}" for i in range(5) for kind in ("question", "answer")
    ]


def test_live_hook_after_backfill_logs_only_new_entries(tmp_path):
    path = _transcript(tmp_path, 2)
    mem = RecordingMem()
    hooks.backfill_transcript(mem, path, task_from="none")
    with path.open("a") as f:
        f.write(json.dumps(_entry("user", "follow-up after resuming")) + "\n")
    hooks.claude_code_hook(
        {"session_id": "sess-1", "transcript_path": str(path), "cwd": "/work/repo"},
        mem, task_from="none", improve=False,
    )
    assert [t["content"] for t in mem.turns][-2:] == ["answer 1", "follow-up after resuming"]
    assert len(mem.turns) == 5


def test_backfill_dry_run_writes_nothing_and_skips_sidechains(tmp_path):
    path = _write(tmp_path / "p" / "s.jsonl", [
        _entry("user", "real question"),
        _entry("user", "subagent prompt written by claude", isSidechain=True),
        _entry("assistant", "real answer"),
    ])
    mem = RecordingMem()
    r = hooks.backfill_transcript(mem, path, dry_run=True)
    assert (r["turns"], r["cwd"]) == (2, "/work/repo")
    assert mem.turns == [] and "s" not in hooks._load_state().get("sessions", {})


def test_backfill_with_real_memory_records_cwd_scope_and_agent(tmp_path):
    from memstrand.scopes import agent_name

    path = _transcript(tmp_path, 1)
    mem = _make_mem(tmp_path)
    hooks.backfill_transcript(mem, path, task_from="none", improve=False)
    turns = mem.sessions.get_session("claude-code:sess-1")
    assert [t["content"] for t in turns] == ["question 0", "answer 0"]
    assert {t["agent"] for t in turns} == {agent_name("claude-code")}


# -- sync improve no longer blocks the store -------------------------------------------------


def test_sync_improve_does_not_hold_the_store_lock(tmp_path, monkeypatch):
    mem = _make_mem(tmp_path)
    mem.log_turn("s", "user", "hello", task_id="r")
    entered, release = threading.Event(), threading.Event()

    def slow_extract(*a, **k):
        entered.set()
        release.wait(5)
        return []

    monkeypatch.setattr(distill_mod, "extract_atoms", slow_extract)
    worker = threading.Thread(target=lambda: mem.improve(["s"]))
    worker.start()
    assert entered.wait(5)
    t0 = time.monotonic()
    assert mem.count() == 0  # would block until release if improve held mem._lock
    assert time.monotonic() - t0 < 1.0
    assert mem.improve(["s"], background=True) == {"queued": [], "busy": True}  # still serialized
    release.set()
    worker.join(5)


# -- CLI ------------------------------------------------------------------------------------------


def test_cli_backfill_dry_run_and_real(tmp_path, capsys, monkeypatch):
    root = tmp_path / "projects"
    _write(root / "-Users-me-alpha" / "aaaaaaaa-1.jsonl",
           [_entry("user", "q"), _entry("assistant", "a")], time.time() - 7200)
    mem = RecordingMem()
    monkeypatch.setattr(cli, "get_memory", lambda *a, **k: mem)

    assert cli.main(["hook", "backfill", str(root), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "turns=2" in out and "dry run: nothing logged" in out and mem.turns == []

    assert cli.main(["hook", "backfill", str(root), "--task-from", "none"]) == 0
    out = capsys.readouterr().out
    assert "chunks=1 atoms=1" in out and "1 memories" in out
    assert len(mem.turns) == 2

    assert cli.main(["hook", "backfill", str(root), "--since", "nope"]) == 1
    assert "--since" in capsys.readouterr().err
