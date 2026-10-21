"""Scoping v2 phase 4: read-side scopes (prompt paths, proxy header) and
agent attribution."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace

import httpx
from starlette.testclient import TestClient
from test_scoping_v2 import HashEmbedder, _make_mem

from memstrand import Memory, hooks
from memstrand.config import get_settings
from memstrand.pipeline import distill as distill_mod
from memstrand.schema import Category, MemoryAtom, Provenance
from memstrand.scopes import agent_name
from memstrand.server import create_app
from memstrand.stores.graph_kuzu import KuzuGraphStore
from memstrand.stores.graph_sqlite import SqliteGraphStore


def _git_repo(path, remote):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "remote", "add", "origin", remote], check=True)
    return path


def _settings(tmp_path):
    return replace(get_settings(), home=tmp_path, server_host="127.0.0.1", server_port=7777)


# -- read side --------------------------------------------------------------------------------


def test_context_hook_adds_repos_named_in_the_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMSTRAND_HOME", str(tmp_path / "home"))
    repo = _git_repo(tmp_path / "projects" / "widgets", "git@github.com:acme/widgets.git")
    elsewhere = tmp_path / "dotclaude"
    elsewhere.mkdir()
    mem = Memory.in_memory(embedder=HashEmbedder())
    mem.remember("widgets builds with pnpm workspaces", scopes=["github.com/acme/widgets"])

    with_path = {"prompt": f"read {repo} and tell me how widgets builds", "cwd": str(elsewhere)}
    without = {"prompt": "tell me how widgets builds", "cwd": str(elsewhere)}
    assert "pnpm workspaces" in hooks.claude_code_context_hook(with_path, mem, task_from="auto")
    assert "pnpm workspaces" not in hooks.claude_code_context_hook(without, mem, task_from="auto")
    # cwd mode: prompt paths don't widen the lookup
    assert "pnpm workspaces" not in hooks.claude_code_context_hook(with_path, mem, task_from="cwd")


def test_proxy_scopes_header_and_agent(tmp_path):
    mem = _make_mem(tmp_path)
    mem.remember("widgets deploy with argo rollouts", scopes=["github.com/acme/widgets"])
    seen = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    app = create_app(mem, _settings(tmp_path), upstream_transport=httpx.MockTransport(upstream))
    with TestClient(app, base_url="http://127.0.0.1:7777") as http:
        resp = http.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "how do widgets deploy with argo"}]},
            headers={"x-memstrand-scopes": "github.com/acme/widgets, other", "x-memstrand-session": "p1"},
        )
        assert resp.status_code == 200
    system = seen["body"]["messages"][0]["content"]
    assert "argo rollouts" in system
    turns = mem.sessions.get_session("p1")
    assert [t["agent"] for t in turns] == [agent_name("openai-proxy")] * 2


# -- agent attribution ---------------------------------------------------------------------------


def test_hook_turns_carry_agent_and_distilled_atoms_inherit_it(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMSTRAND_HOME", str(tmp_path / "home"))
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(json.dumps(
        {"type": "user", "message": {"role": "user", "content": "we use pnpm"}}) + "\n")
    mem = _make_mem(tmp_path)
    hooks.claude_code_hook(
        {"session_id": "s", "transcript_path": str(transcript), "cwd": str(tmp_path)},
        mem, task_from="none", improve=False,
    )
    (turn,) = mem.sessions.get_session("claude-code:s")
    assert turn["agent"] == agent_name("claude-code")

    monkeypatch.setattr(distill_mod, "extract_atoms", lambda *a, **k: [{
        "category": Category.fact, "content": "the team uses pnpm",
        "provenance": Provenance.explicit_statement, "confidence": 0.8, "scope": "task",
    }])
    monkeypatch.setattr(distill_mod, "_enforce_global_trust_boundary", lambda c, t, m: c)
    monkeypatch.setattr(distill_mod, "_judge", lambda m, c, s: {
        "action": "add", "target_id": None, "relation": "related_to", "reason": ""})
    distill_mod.distill_sessions(mem, ["claude-code:s"])
    assert mem.graph.get_atom(mem._atom_id("the team uses pnpm")).agent == agent_name("claude-code")


def test_cli_remember_records_agent_and_list_filters(tmp_path, capsys, monkeypatch):
    from memstrand import cli

    mem = _make_mem(tmp_path)
    monkeypatch.setattr(mem, "close", lambda: None)
    monkeypatch.setattr(cli, "get_memory", lambda *a, **k: mem)
    assert cli.main(["remember", "via the cli"]) == 0
    atom_id = capsys.readouterr().out.strip()
    mem.remember("via something else", agent="mcp@box")
    assert mem.graph.get_atom(atom_id).agent == agent_name("cli")
    assert cli.main(["list", "--agent", agent_name("cli")]) == 0
    out = capsys.readouterr().out
    assert "via the cli" in out and "via something else" not in out
    assert cli.main(["show", atom_id]) == 0
    assert f"agent:      {agent_name('cli')}" in capsys.readouterr().out


def test_agent_over_rest_and_export(tmp_path):
    from memstrand.client import RemoteMemory

    mem = _make_mem(tmp_path / "a")
    with TestClient(create_app(mem, _settings(tmp_path)), base_url="http://127.0.0.1:7777") as http:
        remote = RemoteMemory("http://127.0.0.1:7777", client=http)
        atom = remote.remember("remote fact", agent="claude-code@laptop")
        assert atom["agent"] == "claude-code@laptop"
        assert [a["id"] for a in remote.list(agent="claude-code@laptop")] == [atom["id"]]
        assert remote.list(agent="nobody") == []
        remote.log_turn("s", "user", "hi", agent="claude-code@laptop")
    dst = _make_mem(tmp_path / "b")
    dst.import_jsonl(mem.export("jsonl"))
    assert dst.sessions.get_session("s")[0]["agent"] == "claude-code@laptop"
    assert dst.list(agent="claude-code@laptop")[0]["content"] == "remote fact"


def test_graph_stores_filter_by_agent(tmp_path):
    for store in (SqliteGraphStore(str(tmp_path / "g.db")), KuzuGraphStore(str(tmp_path / "k"))):
        store.add_atom(MemoryAtom(id="a", category=Category.fact, content="a", agent="cli@x"))
        store.add_atom(MemoryAtom(id="b", category=Category.fact, content="b", agent="mcp@x"))
        assert [a.id for a in store.list_atoms(agent="cli@x")] == ["a"]
        assert store.count_atoms(agent="mcp@x") == 1
        assert store.count_atoms() == 2
        store.close()
