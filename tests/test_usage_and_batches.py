"""Token accounting (ledger, purposes, per-run totals) and batch-friendly
backfill (`--limit` = next N unprocessed, `--min-turns`, original timestamps)."""

from __future__ import annotations

import json
import time
from datetime import datetime

import httpx
import pytest
from test_backfill import RecordingMem, _entry, _write
from test_scoping_v2 import _make_mem

from memstrand import cli, hooks
from memstrand.pipeline import distill as distill_mod
from memstrand.pipeline import extract as extract_mod
from memstrand.pipeline import usage
from memstrand.schema import Category, Provenance


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMSTRAND_HOME", str(tmp_path / "home"))


def _fake_post(reply: str, prompt_tokens=100, completion_tokens=20):
    def post(url, headers=None, json=None, timeout=None):
        body = {"choices": [{"message": {"content": reply}}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))
    return post


# -- ledger ---------------------------------------------------------------------------------------


def test_chat_completion_records_usage_with_purpose_and_session(monkeypatch):
    monkeypatch.setattr(extract_mod.httpx, "post", _fake_post("hi", 120, 30))
    with usage.collect() as totals, usage.purpose("judge"), usage.for_session("s1"):
        assert extract_mod.chat_completion("http://x/v1", "haiku", [{"role": "user", "content": "q"}]) == "hi"
    assert totals == {"calls": 1, "prompt_tokens": 120, "completion_tokens": 30}
    (row,) = usage.read_ledger()
    assert (row["purpose"], row["session"], row["model"], row["prompt_tokens"]) == ("judge", "s1", "haiku", 120)
    assert usage.summarize(usage.read_ledger(), by="purpose")["judge"]["completion_tokens"] == 30


def test_missing_usage_block_is_recorded_as_unreported(monkeypatch):
    def post(url, headers=None, json=None, timeout=None):
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]},
                              request=httpx.Request("POST", url))
    monkeypatch.setattr(extract_mod.httpx, "post", post)
    extract_mod.chat_completion("http://x/v1", "m", [])
    assert usage.read_ledger()[0]["reported"] is False


def test_distill_tags_purposes_and_reports_totals(tmp_path, monkeypatch):
    mem = _make_mem(tmp_path)
    mem.remember("deploys go through argo rollouts", scopes=["r"])  # a similar memory -> judge runs
    mem.log_turn("s", "user", "we deploy with argo rollouts now", task_id="r")
    replies = iter([
        json.dumps([{"category": "fact", "content": "deploys use argo rollouts", "scope": "task",
                     "provenance": "explicit_statement", "confidence": 0.9}]),
        json.dumps({"action": "add", "target_id": None, "relation": "related_to", "reason": ""}),
    ])
    monkeypatch.setattr(extract_mod.httpx, "post", lambda *a, **k: _fake_post(next(replies), 200, 40)(*a, **k))
    report = distill_mod.distill_sessions(mem, ["s"])
    assert report["usage"]["calls"] == 2 and report["usage"]["prompt_tokens"] == 400
    by_purpose = usage.summarize(usage.read_ledger(), by="purpose")
    assert set(by_purpose) == {"extract", "judge"}
    assert {r["session"] for r in usage.read_ledger()} == {"s"}


def test_usage_command(capsys, monkeypatch):
    monkeypatch.setattr(extract_mod.httpx, "post", _fake_post("x", 1000, 50))
    for p in ("extract", "judge", "judge"):
        with usage.purpose(p):
            extract_mod.chat_completion("http://x/v1", "haiku", [])
    assert cli.main(["usage", "--by", "purpose"]) == 0
    out = capsys.readouterr().out
    assert "judge" in out and "2,000" in out and "TOTAL" in out and "3,150" in out
    assert cli.main(["usage", "--since", "bad"]) == 1


# -- batches --------------------------------------------------------------------------------------


def _sessions(root, n, turns=2):
    for i in range(n):
        entries = []
        for t in range(turns):
            entries.append(_entry("user" if t % 2 == 0 else "assistant", f"s{i} turn {t}",
                                  timestamp=f"2026-09-{i + 1:02d}T10:00:{t:02d}Z"))
        _write(root / "-Users-me-proj" / f"sess{i:02d}.jsonl", entries, time.time() - 86400 * (30 - i))


def test_limit_takes_the_next_unprocessed_sessions(tmp_path, capsys, monkeypatch):
    root = tmp_path / "projects"
    _sessions(root, 5)
    mem = RecordingMem()
    monkeypatch.setattr(cli, "get_memory", lambda *a, **k: mem)
    assert cli.main(["hook", "backfill", str(root), "--limit", "2", "--task-from", "none"]) == 0
    assert cli.main(["hook", "backfill", str(root), "--limit", "2", "--task-from", "none"]) == 0
    sessions = [t["session_id"] for t in mem.turns]
    assert list(dict.fromkeys(sessions)) == [f"claude-code:sess{i:02d}" for i in range(4)]
    capsys.readouterr()
    assert cli.main(["hook", "backfill", str(root), "--limit", "2", "--task-from", "none"]) == 0
    assert cli.main(["hook", "backfill", str(root), "--task-from", "none"]) == 0
    assert "no transcripts with unprocessed turns" in capsys.readouterr().out


def test_min_turns_logs_small_sessions_without_llm(tmp_path, capsys, monkeypatch):
    root = tmp_path / "projects"
    _sessions(root, 1, turns=2)
    mem = _make_mem(tmp_path)
    monkeypatch.setattr(mem, "close", lambda: None)
    monkeypatch.setattr(cli, "get_memory", lambda *a, **k: mem)
    calls = []
    monkeypatch.setattr(mem, "improve", lambda *a, **k: calls.append(a) or {})
    assert cli.main(["hook", "backfill", str(root), "--min-turns", "5", "--task-from", "none"]) == 0
    out = capsys.readouterr().out
    assert "logged only (below --min-turns)" in out and "1 small sessions logged without distilling" in out
    assert calls == []
    assert len(mem.sessions.get_session("claude-code:sess00")) == 2  # still searchable
    assert mem.sessions.undistilled_sessions() == []  # consolidation won't spend tokens on it later


def test_backfilled_turns_keep_original_timestamps(tmp_path):
    root = tmp_path / "projects"
    _sessions(root, 1, turns=2)
    mem = _make_mem(tmp_path)
    (path,) = hooks.discover_transcripts([root])
    hooks.backfill_transcript(mem, path, task_from="none", improve=False)
    stamps = [t["ts"] for t in mem.sessions.get_session("claude-code:sess00")]
    assert stamps == [datetime.fromisoformat("2026-09-01T10:00:00Z").timestamp(),
                      datetime.fromisoformat("2026-09-01T10:00:01Z").timestamp()]


def test_mark_distilled_over_rest(tmp_path):
    from dataclasses import replace

    from starlette.testclient import TestClient

    from memstrand.client import RemoteMemory
    from memstrand.config import get_settings
    from memstrand.server import create_app

    mem = _make_mem(tmp_path)
    mem.log_turn("claude-code:x", "user", "hi")
    settings = replace(get_settings(), home=tmp_path, server_host="127.0.0.1", server_port=7777)
    with TestClient(create_app(mem, settings), base_url="http://127.0.0.1:7777") as http:
        remote = RemoteMemory("http://127.0.0.1:7777", client=http)
        assert remote.mark_distilled("claude-code:x") >= 1
        remote.log_turn("claude-code:y", "user", "later", ts=1_700_000_000.0)
        assert http.post("/api/turns", json={"session_id": "z", "role": "user", "content": "c",
                                             "ts": "yesterday"}).status_code == 400
    assert mem.sessions.undistilled_sessions() == ["claude-code:y"]
    assert mem.sessions.get_session("claude-code:y")[0]["ts"] == 1_700_000_000.0


def test_trust_pass_is_tagged(tmp_path, monkeypatch):
    mem = _make_mem(tmp_path)
    turns = [{"role": "user", "content": "always use metric units", "id": 1, "task_id": None}]
    cands = [{"category": Category.instruction, "content": "always use metric units",
              "provenance": Provenance.explicit_statement, "confidence": 0.9, "scope": "global"}]
    monkeypatch.setattr(extract_mod.httpx, "post", _fake_post("[]", 50, 5))
    distill_mod._enforce_global_trust_boundary(cands, turns, mem)
    assert [r["purpose"] for r in usage.read_ledger()] == ["trust"]
