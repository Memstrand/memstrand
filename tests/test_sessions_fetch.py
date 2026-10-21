"""Fetching raw sessions (`memstrand session`, `search-sessions`) and the
attribution fixes found by the backfill pilot."""

from __future__ import annotations

import sqlite3
import subprocess
from dataclasses import replace

import pytest
from starlette.testclient import TestClient
from test_scoping_v2 import _make_mem

from memstrand import cli, hooks
from memstrand.client import RemoteMemory
from memstrand.config import get_settings
from memstrand.pipeline import extract as extract_mod
from memstrand.scopes import ScopeResolver, git_root
from memstrand.server import create_app
from memstrand.stores.sqlite import SqliteSessionStore

# -- pilot fixes: executables and symlinks are not "touched repos" ------------------------


def test_bash_skips_the_program_being_run():
    assert hooks._bash_paths("~/.claude/skills/g/.venv/bin/python3 x.py --out /tmp/y") == ["/tmp/y"]
    assert hooks._bash_paths("FOO=1 /usr/bin/env python /work/a.py") == ["/work/a.py"]
    assert hooks._bash_paths("(cd /repo/x; make) && ls") == ["/repo/x"]
    assert hooks._bash_paths("cd -P /repo/y") == ["/repo/y"]


def test_git_root_does_not_follow_symlinks_into_another_repo(tmp_path):
    tool_repo = tmp_path / "opt" / "homebrew"
    (tool_repo / "bin").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(tool_repo)], check=True)
    real_python = tool_repo / "bin" / "python3"
    real_python.write_text("")
    venv_bin = tmp_path / "skills" / "g" / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python3").symlink_to(real_python)

    assert git_root(venv_bin / "python3") is None
    resolver = ScopeResolver(tmp_path / "c.json", hostname="h")
    assert hooks._scopes_for_paths([str(venv_bin / "python3")], str(tmp_path), resolver) == []
    assert git_root(real_python) == tool_repo  # the real file still resolves normally


def test_extraction_prompt_asks_for_durable_items_only():
    assert "one-off events" in extract_mod.EXTRACTION_SYSTEM_PROMPT
    assert "fewer, better memories" in extract_mod.EXTRACTION_SYSTEM_PROMPT


# -- session store: lookup and search ------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    s = SqliteSessionStore(str(tmp_path / "s.db"))
    s.append_turn("claude-code:aaaa1111-x", "user", "how do we deploy the helm charts to S3?", ts=1.0)
    s.append_turn("claude-code:aaaa1111-x", "assistant", "kaniko builds the image, then helm-sync.sh uploads", ts=2.0)
    s.append_turn("claude-code:aaaa2222-y", "user", "kaniko failed on the publish stage", ts=3.0)
    s.append_turn("proxy-2026-10-21", "user", "unrelated chat 100% done", ts=4.0)
    return s


def test_find_sessions_forms(store):
    assert store.find_sessions("claude-code:aaaa1111-x") == ["claude-code:aaaa1111-x"]
    assert store.find_sessions("aaaa1111-x") == ["claude-code:aaaa1111-x"]
    assert store.find_sessions("aaaa1") == ["claude-code:aaaa1111-x"]
    assert store.find_sessions("aaaa") == ["claude-code:aaaa1111-x", "claude-code:aaaa2222-y"]
    assert store.find_sessions("%") == []  # LIKE wildcards are escaped
    assert store.find_sessions("nope") == []


def test_search_turns_all_words_with_snippets(store):
    hits = store.search_turns("kaniko")
    assert {h["session_id"] for h in hits} == {"claude-code:aaaa1111-x", "claude-code:aaaa2222-y"}
    assert [h["role"] for h in store.search_turns("kaniko publish")] == ["user"]  # every word must match
    assert "[kaniko]" in store.search_turns("kaniko publish")[0]["snippet"]
    assert store.search_turns("kaniko", session_id="claude-code:aaaa1111-x")[0]["role"] == "assistant"
    assert store.search_turns("!!!") == []


def test_existing_turns_are_indexed_on_first_open(tmp_path):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE turns(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, task_id TEXT, "
                 "role TEXT, content TEXT, ts REAL)")
    conn.execute("INSERT INTO turns(session_id, task_id, role, content, ts) "
                 "VALUES ('s','','user','legacy terraform state bucket',1.0)")
    conn.commit()
    conn.close()
    store = SqliteSessionStore(str(db))
    assert [h["content"] for h in store.search_turns("terraform bucket")] == ["legacy terraform state bucket"]
    store.clear_session("s")
    assert store.search_turns("terraform") == []


# -- Memory / REST / client / CLI ---------------------------------------------------------


def _seed(mem):
    mem.log_turn("claude-code:abcd1234-1", "user", "where is the CI image built?", task_id="r")
    mem.log_turn("claude-code:abcd1234-1", "assistant", "kaniko pushes it to artifactory", task_id="r")
    mem.log_turn("claude-code:abcd9999-2", "user", "something else", task_id="r")


def test_memory_session_turns_and_errors(tmp_path):
    mem = _make_mem(tmp_path)
    _seed(mem)
    got = mem.session_turns("abcd1234")
    assert got["session_id"] == "claude-code:abcd1234-1" and len(got["turns"]) == 2
    assert len(mem.session_turns("abcd1234", limit=1)["turns"]) == 1
    with pytest.raises(KeyError):
        mem.session_turns("zzz")
    with pytest.raises(ValueError, match="ambiguous"):
        mem.session_turns("abcd")
    assert [h["content"] for h in mem.search_turns("artifactory", session="abcd1234")] == [
        "kaniko pushes it to artifactory"
    ]


def test_rest_client_and_cli(tmp_path, capsys, monkeypatch):
    mem = _make_mem(tmp_path)
    _seed(mem)
    atom = mem.remember("CI image is built with kaniko", session_id="claude-code:abcd1234-1")
    settings = replace(get_settings(), home=tmp_path, server_host="127.0.0.1", server_port=7777)
    with TestClient(create_app(mem, settings), base_url="http://127.0.0.1:7777") as http:
        remote = RemoteMemory("http://127.0.0.1:7777", client=http)
        assert remote.session_turns("claude-code:abcd1234-1")["session_id"] == "claude-code:abcd1234-1"
        assert len(remote.session_turns("abcd1234", limit=1)["turns"]) == 1
        with pytest.raises(KeyError):
            remote.session_turns("zzz")
        assert http.get("/api/sessions/abcd/turns").status_code == 400  # ambiguous
        assert remote.search_turns("artifactory")[0]["session_id"] == "claude-code:abcd1234-1"
        assert http.get("/api/turns/search").status_code == 400

        monkeypatch.setattr(cli, "get_memory", lambda *a, **k: remote)
        monkeypatch.setattr(remote, "close", lambda: None)
        assert cli.main(["session", "abcd1234"]) == 0
        out = capsys.readouterr().out
        assert "session claude-code:abcd1234-1 — 2 turns" in out and "kaniko pushes it" in out
        assert cli.main(["search-sessions", "artifactory"]) == 0
        out = capsys.readouterr().out
        assert "[artifactory]" in out and "claude-code:abcd1234-1" in out
        assert cli.main(["show", atom["id"] if isinstance(atom, dict) else atom.id]) == 0
        assert "session:    claude-code:abcd1234-1" in capsys.readouterr().out
        assert cli.main(["session", "zzz"]) == 1
