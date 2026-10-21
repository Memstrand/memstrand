"""Scoping v2 phase 2: attribute session turns to the repos they touched,
and let distillation pick memory scopes from those candidates."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from test_scoping_v2 import HashEmbedder, _make_mem

from memstrand import Memory, hooks
from memstrand.pipeline import distill as distill_mod
from memstrand.pipeline import extract as extract_mod
from memstrand.schema import Category, Provenance
from memstrand.scopes import ScopeResolver


def _git_repo(path: Path, remote: str | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    if remote:
        subprocess.run(["git", "-C", str(path), "remote", "add", "origin", remote], check=True)
    return path


def _assistant_tool(name: str, **inp) -> dict:
    return {
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "tool_use", "name": name, "input": inp}]},
    }


def _text(role: str, text: str) -> dict:
    return {"type": role, "message": {"role": role, "content": [{"type": "text", "text": text}]}}


# -- path harvesting ------------------------------------------------------------------------


def test_bash_paths():
    assert hooks._bash_paths("cd ~/projects/memstrand && git log -3") == ["~/projects/memstrand"]
    assert hooks._bash_paths("cat /etc/hosts | grep x; ls ~/a") == ["/etc/hosts", "~/a"]
    assert hooks._bash_paths("cd src && pytest") == ["src"]  # relative cd target, resolved vs cwd later
    assert hooks._bash_paths("echo 'unterminated") == []  # shlex error -> whitespace split, no paths


def test_tool_paths_carried_forward_to_next_text_turn():
    entries = [
        _text("user", "please read ~/projects/foo/README.md and summarize"),
        _assistant_tool("Read", file_path="/repo/b/x.py"),
        _assistant_tool("Grep", pattern="def", path="/repo/c"),
        _text("assistant", "Here is the summary."),
        _assistant_tool("Bash", command="cd /repo/d && ls"),
    ]
    turns = hooks._new_turns_with_paths(entries)
    assert [(r, p) for r, _, p in turns] == [
        ("user", ["~/projects/foo/README.md"]),
        ("assistant", ["/repo/b/x.py", "/repo/c", "/repo/d"]),  # trailing tool call -> last turn
    ]
    assert hooks._new_turns(entries) == [(r, t) for r, t, _ in turns]  # legacy shape unchanged


def test_scopes_for_paths_only_counts_git_repos(tmp_path):
    repo = _git_repo(tmp_path / "work" / "widgets", "git@github.com:acme/widgets.git")
    (repo / "src").mkdir()
    (repo / "src" / "a.py").write_text("x")
    plain = tmp_path / "plain"
    plain.mkdir()
    resolver = ScopeResolver(tmp_path / "cache.json", hostname="box")
    got = hooks._scopes_for_paths(
        [str(repo / "src" / "a.py"), "src/a.py", str(plain), "/does/not/exist", str(repo)],
        cwd=str(repo), resolver=resolver,
    )
    assert got == ["github.com/acme/widgets"]  # deduped; plain dir and missing path ignored


# -- hook end to end ----------------------------------------------------------------------------


def test_hook_auto_attributes_turns_to_touched_repos(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMSTRAND_HOME", str(tmp_path / "home"))
    home_repo = _git_repo(tmp_path / "dotclaude")  # no remote -> local: key
    other = _git_repo(tmp_path / "projects" / "memstrand", "https://github.com/Memstrand/memstrand")
    (other / "api.py").write_text("x")

    transcript = tmp_path / "t.jsonl"
    transcript.write_text("\n".join(json.dumps(e) for e in [
        _text("user", f"read {other} and tell me how recall works"),
        _assistant_tool("Read", file_path=str(other / "api.py")),
        _text("assistant", "recall applies the scope filter first"),
        _text("user", "thanks"),
    ]) + "\n")

    mem = Memory.in_memory(embedder=HashEmbedder())
    stdin = {"session_id": "s1", "transcript_path": str(transcript), "cwd": str(home_repo)}
    report = hooks.claude_code_hook(stdin, mem, task_from="auto", improve=False)
    assert report["logged"] == 3
    primary = report["task_id"]
    assert primary.startswith("local:") and primary.endswith(str(home_repo.resolve()))

    turns = mem.sessions.get_session("claude-code:s1")
    assert [t["scopes"] for t in turns] == [
        ["github.com/memstrand/memstrand"],  # path in the user's prompt
        ["github.com/memstrand/memstrand"],  # the Read tool call
        [],
    ]
    assert all(t["task_id"] == primary for t in turns)


def test_hook_cwd_mode_does_not_harvest(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMSTRAND_HOME", str(tmp_path / "home"))
    other = _git_repo(tmp_path / "other", "git@github.com:a/b.git")
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(json.dumps(_text("user", f"look at {other}")) + "\n")
    mem = Memory.in_memory(embedder=HashEmbedder())
    stdin = {"session_id": "s2", "transcript_path": str(transcript), "cwd": str(tmp_path)}
    hooks.claude_code_hook(stdin, mem, task_from="cwd", improve=False)
    assert mem.sessions.get_session("claude-code:s2")[0]["scopes"] == []


# -- distillation picks from candidates ------------------------------------------------------------


def _cand(content, scope):
    return {"category": Category.fact, "content": content, "provenance": Provenance.explicit_statement,
            "confidence": 0.8, "scope": scope}


@pytest.fixture()
def session(tmp_path, monkeypatch):
    mem = _make_mem(tmp_path)
    primary = "local:box:/Users/me/.claude"
    mem.log_turn("s", "user", "read memstrand and widgets", task_id=primary,
                 scopes=["github.com/me/memstrand", "github.com/acme/widgets"])
    mem.log_turn("s", "assistant", "done", task_id=primary)
    monkeypatch.setattr(distill_mod, "_enforce_global_trust_boundary", lambda c, t, m: c)
    monkeypatch.setattr(
        distill_mod, "_judge",
        lambda m, c, similar: {"action": "add", "target_id": None, "relation": "related_to", "reason": ""},
    )
    return mem, primary


def test_distill_offers_candidates_and_honors_listed_repos(session, monkeypatch):
    mem, primary = session
    seen = {}

    def fake_extract(turns, base_url, model, timeout=180.0, api_key="local", scope_candidates=None):
        seen["candidates"] = scope_candidates
        seen["touched"] = [t.get("touched") for t in turns]
        return [
            _cand("recall filters scope before ranking", ["memstrand"]),
            _cand("both repos use ruff", ["memstrand", "widgets"]),
            _cand("the dotfiles live in .claude", "task"),
            _cand("made up repo fact", ["nonexistent"]),
        ]

    monkeypatch.setattr(distill_mod, "extract_atoms", fake_extract)
    report = distill_mod.distill_sessions(mem, ["s"])
    assert report["errors"] == [] and report["atoms"] == 4
    # session's own scope first, then touched scopes (stored sorted by key)
    assert seen["candidates"] == [".claude", "widgets", "memstrand"]
    assert seen["touched"] == [["widgets", "memstrand"], []]

    def scopes_of(text):
        return mem.graph.get_atom(mem._atom_id(text)).scopes

    assert scopes_of("recall filters scope before ranking") == ["github.com/me/memstrand"]
    assert scopes_of("both repos use ruff") == ["github.com/acme/widgets", "github.com/me/memstrand"]
    assert scopes_of("the dotfiles live in .claude") == [primary]
    assert scopes_of("made up repo fact") == [primary]  # invented name -> session's own scope


def test_distill_without_touched_repos_keeps_old_call_shape(tmp_path, monkeypatch):
    """No candidates beyond the session's own scope: extract_atoms is called
    exactly as before (no scope_candidates kwarg)."""
    mem = _make_mem(tmp_path)
    mem.log_turn("s", "user", "hello", task_id="repo-a")
    calls = []

    def fake_extract(turns, base_url, model, timeout=180.0, api_key="local"):
        calls.append(turns)
        return []

    monkeypatch.setattr(distill_mod, "extract_atoms", fake_extract)
    distill_mod.distill_sessions(mem, ["s"])
    assert len(calls) == 1


def test_labels_fall_back_to_full_keys_on_collision():
    labels = distill_mod._scope_labels(["github.com/a/infra", "gitlab.corp/b/infra", "github.com/a/web"])
    assert labels == {
        "github.com/a/infra": "github.com/a/infra",
        "gitlab.corp/b/infra": "gitlab.corp/b/infra",
        "web": "github.com/a/web",
    }


def test_extraction_prompt_lists_repos_and_annotations(monkeypatch):
    captured = {}

    def fake_chat(base_url, model, messages, timeout=120.0, api_key="local"):
        captured["user"] = messages[1]["content"]
        return json.dumps([
            {"category": "fact", "content": "a", "scope": ["memstrand", "", 3]},
            {"category": "fact", "content": "b", "scope": []},
            {"category": "preference", "content": "c", "scope": " "},
            {"category": "fact", "content": "d", "scope": "memstrand"},
        ])

    monkeypatch.setattr(extract_mod, "chat_completion", fake_chat)
    atoms = extract_mod.extract_atoms(
        [{"role": "user", "content": "hi", "touched": ["memstrand"]}], "http://x", "m",
        scope_candidates=[".claude", "memstrand"],
    )
    assert "Repos in this session:\n1. .claude (this session's own repo)\n2. memstrand" in captured["user"]
    assert "USER: hi\n[touched: memstrand]" in captured["user"]
    # a bare name is a one-element list (validated against candidates by distill)
    assert [a["scope"] for a in atoms] == [["memstrand"], "task", "global", ["memstrand"]]


# -- surfaces --------------------------------------------------------------------------------------


def test_turn_scopes_roundtrip_rest_and_export(tmp_path):
    from dataclasses import replace

    from starlette.testclient import TestClient

    from memstrand.client import RemoteMemory
    from memstrand.config import get_settings
    from memstrand.server import create_app

    mem = _make_mem(tmp_path / "a")
    settings = replace(get_settings(), home=tmp_path, server_host="127.0.0.1", server_port=7777)
    with TestClient(create_app(mem, settings), base_url="http://127.0.0.1:7777") as http:
        remote = RemoteMemory("http://127.0.0.1:7777", client=http)
        remote.log_turn("s", "user", "one", task_id="p", scopes=["x"])
        remote.log_turns([{"session_id": "s", "role": "assistant", "content": "two", "task_id": "p",
                           "scopes": ["y", "z"]}])
    assert [t["scopes"] for t in mem.sessions.get_session("s")] == [["x"], ["y", "z"]]

    dst = _make_mem(tmp_path / "b")
    dst.import_jsonl(mem.export("jsonl"))
    assert [t["scopes"] for t in dst.sessions.get_session("s")] == [["x"], ["y", "z"]]


def test_legacy_turns_table_gains_scopes_column(tmp_path):
    import sqlite3

    from memstrand.stores.sqlite import SqliteSessionStore

    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE turns(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, task_id TEXT, "
                 "role TEXT, content TEXT, ts REAL)")
    conn.execute("INSERT INTO turns(session_id, task_id, role, content, ts) VALUES ('s','t','user','hi',1.0)")
    conn.commit()
    conn.close()
    store = SqliteSessionStore(str(db))
    assert store.get_session("s")[0]["scopes"] == []
    store.append_turn("s", "user", "again", task_id="t", scopes=["k"])
    assert store.get_session("s")[1]["scopes"] == ["k"]
