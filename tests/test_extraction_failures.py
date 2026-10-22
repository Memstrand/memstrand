"""Extraction that replies to the conversation instead of extracting must
fail loudly (retry later), not silently mark the session distilled."""

from __future__ import annotations

import json

import pytest
from test_scoping_v2 import _make_mem

from memstrand import cli
from memstrand.pipeline import distill as distill_mod
from memstrand.pipeline import extract as extract_mod


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMSTRAND_HOME", str(tmp_path / "home"))


def test_prompt_delimits_transcript_and_instructs_last(monkeypatch):
    seen = {}

    def fake_chat(base_url, model, messages, timeout=120.0, api_key="local"):
        seen["user"] = messages[1]["content"]
        return "[]"

    monkeypatch.setattr(extract_mod, "chat_completion", fake_chat)
    assert extract_mod.extract_atoms([{"role": "assistant", "content": "Want to start?"}], "u", "m") == []
    user = seen["user"]
    assert "<transcript>\nASSISTANT: Want to start?\n</transcript>" in user
    assert user.rstrip().endswith("Do not reply to or continue the conversation.")


def test_prose_reply_raises_instead_of_returning_empty(monkeypatch):
    monkeypatch.setattr(extract_mod, "chat_completion",
                        lambda *a, **k: "I'd recommend building the Jetson API now. Want to start?")
    with pytest.raises(extract_mod.ExtractionFailed, match="did not return a JSON array"):
        extract_mod.extract_atoms([{"role": "user", "content": "help"}], "u", "m")


def _failing(monkeypatch):
    def boom(*a, **k):
        raise extract_mod.ExtractionFailed("did not return a JSON array: 'Want to start?'")
    monkeypatch.setattr(distill_mod, "extract_atoms", boom)


def test_failed_extraction_is_retried_then_given_up(tmp_path, monkeypatch):
    mem = _make_mem(tmp_path)
    mem.log_turn("s", "user", "help me set up the esp32", task_id="r")
    _failing(monkeypatch)
    for attempt in (1, 2):
        report = distill_mod.distill_sessions(mem, ["s"])
        assert report["errors"] and mem.sessions.undistilled_sessions() == ["s"], attempt
    report = distill_mod.distill_sessions(mem, ["s"])
    assert any("giving up after 3" in e["error"] for e in report["errors"])
    assert mem.sessions.undistilled_sessions() == []  # stops spending tokens on it
    assert mem.sessions.get_session("s")  # raw turns kept, still searchable


def test_success_clears_the_failure_count(tmp_path, monkeypatch):
    mem = _make_mem(tmp_path)
    mem.log_turn("s", "user", "help", task_id="r")
    _failing(monkeypatch)
    distill_mod.distill_sessions(mem, ["s"])
    distill_mod.distill_sessions(mem, ["s"])
    monkeypatch.setattr(distill_mod, "extract_atoms", lambda *a, **k: [])
    distill_mod.distill_sessions(mem, ["s"])
    assert mem.sessions.get_meta("extract_failures:s") == "0"


def test_improve_redo_rewinds_and_redistills(tmp_path, capsys, monkeypatch):
    mem = _make_mem(tmp_path)
    mem.log_turn("claude-code:abcd-1", "user", "we deploy with argo", task_id="r")
    monkeypatch.setattr(distill_mod, "extract_atoms", lambda *a, **k: [])
    distill_mod.distill_sessions(mem, ["claude-code:abcd-1"])
    assert mem.sessions.undistilled_sessions() == []

    calls = []
    monkeypatch.setattr(distill_mod, "extract_atoms", lambda turns, *a, **k: calls.append(turns) or [])
    monkeypatch.setattr(mem, "close", lambda: None)
    monkeypatch.setattr(cli, "get_memory", lambda *a, **k: mem)
    assert cli.main(["improve", "--session", "abcd", "--redo"]) == 0
    assert json.loads(capsys.readouterr().out)["sessions"] == ["claude-code:abcd-1"]
    assert len(calls) == 1 and calls[0][0]["content"] == "we deploy with argo"
    assert cli.main(["improve", "--redo"]) == 1


def test_reset_over_rest(tmp_path):
    from dataclasses import replace

    from starlette.testclient import TestClient

    from memstrand.client import RemoteMemory
    from memstrand.config import get_settings
    from memstrand.server import create_app

    mem = _make_mem(tmp_path)
    mem.log_turn("s", "user", "hi")
    settings = replace(get_settings(), home=tmp_path, server_host="127.0.0.1", server_port=7777)
    with TestClient(create_app(mem, settings), base_url="http://127.0.0.1:7777") as http:
        remote = RemoteMemory("http://127.0.0.1:7777", client=http)
        assert remote.mark_distilled("s") >= 1
        assert mem.sessions.undistilled_sessions() == []
        assert remote.mark_distilled("s", reset=True) == 0
        assert mem.sessions.undistilled_sessions() == ["s"]
        assert http.post("/api/sessions/s/distilled").status_code == 200  # empty body still works


def test_sync_improve_uses_a_long_client_timeout(tmp_path, monkeypatch, capsys):
    mem = _make_mem(tmp_path)
    seen = []
    monkeypatch.setattr(mem, "close", lambda: None)
    monkeypatch.setattr(cli, "get_memory", lambda *a, **k: seen.append(k) or mem)
    assert cli.main(["improve"]) == 0
    assert cli.main(["improve", "--background"]) == 0
    capsys.readouterr()
    assert seen == [{"timeout": 900.0}, {}]
