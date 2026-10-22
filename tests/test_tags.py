"""Tags: what a memory is about (systems, activity/initiative, domain)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace

import pytest
from starlette.testclient import TestClient
from test_scoping_v2 import _make_mem

from memstrand import cli
from memstrand.client import RemoteMemory
from memstrand.config import get_settings
from memstrand.pipeline import distill as distill_mod
from memstrand.pipeline import extract as extract_mod
from memstrand.pipeline import tagging
from memstrand.schema import Category, MemoryAtom, normalize_tags
from memstrand.server import create_app
from memstrand.stores.graph_kuzu import KuzuGraphStore
from memstrand.stores.graph_sqlite import SqliteGraphStore
from memstrand.stores.keyword_sqlite import SqliteKeywordIndex

LESSON = ("When executing multi-step infrastructure changes affecting multiple tenants, run them "
          "sequentially one tenant at a time to avoid flooding the prod prefix service")


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMSTRAND_HOME", str(tmp_path / "home"))


@pytest.fixture()
def mem(tmp_path):
    return _make_mem(tmp_path)


def test_normalize_tags():
    assert normalize_tags(["Google Cloud", "GCP", "gcp", "ipv6_reprovisioning", "x", "a" * 50, None, 3]) == [
        "google-cloud", "gcp", "ipv6-reprovisioning"]
    assert len(normalize_tags([f"t{i}" for i in range(20)])) == 8


@pytest.mark.parametrize("backend", ["sqlite", "kuzu"])
def test_graph_stores_persist_and_query_tags(tmp_path, backend):
    store = SqliteGraphStore(str(tmp_path / "g.db")) if backend == "sqlite" else KuzuGraphStore(str(tmp_path / "k"))
    store.add_atom(MemoryAtom(id="a", category=Category.fact, content="a", tags=["gcp", "ipv6", "multi-tenant"]))
    store.add_atom(MemoryAtom(id="b", category=Category.fact, content="b", tags=["gcp", "ipv6"]))
    store.add_atom(MemoryAtom(id="c", category=Category.fact, content="c", tags=["gcp"], scopes=["r"]))
    assert store.get_atom("a").tags == ["gcp", "ipv6", "multi-tenant"]
    assert store.get_atoms(["b"])["b"].tags == ["gcp", "ipv6"]
    assert {x.id for x in store.list_atoms(tag="ipv6")} == {"a", "b"}
    assert store.count_atoms(tag="gcp") == 3
    assert store.tag_counts()[0] == ("gcp", 3)
    assert store.tag_counts(task_id="r") == [("gcp", 1)]
    assert store.related_by_tags("a") == [("b", 2)]
    store.add_atom(MemoryAtom(id="b", category=Category.fact, content="b", tags=["aws"]))  # replace
    assert store.get_atom("b").tags == ["aws"] and store.related_by_tags("a") == []
    store.delete_atom("a")
    assert store.count_atoms(tag="multi-tenant") == 0
    store.close()


def test_tags_make_context_findable(mem):
    """The lesson's own words never mention GCP or a region build — its tags do."""
    mem.remember(LESSON, scopes=["geoforge"], tags=["gcp", "new-region-build", "ipv6-reprovisioning", "multi-tenant"])
    mem.remember("The weekly report is generated on Fridays", scopes=["geoforge"])
    hits = mem.recall("gcp new region build", scopes=["geoforge"])
    assert hits and hits[0]["content"] == LESSON
    assert hits[0]["tags"] == ["gcp", "new-region-build", "ipv6-reprovisioning", "multi-tenant"]


def test_tag_filter_and_list(mem):
    mem.remember("deploys use argo rollouts", tags=["argo", "deploy"])
    mem.remember("deploys use helm", tags=["helm", "deploy"])
    assert [h["content"] for h in mem.recall("deploys", tags=["argo"])] == ["deploys use argo rollouts"]
    assert {a["content"] for a in mem.list(tag="deploy")} == {"deploys use argo rollouts", "deploys use helm"}


def test_remember_unions_tags_and_reembeds(mem):
    a = mem.remember("the api runs on port 9090", scopes=["r1"], tags=["api"])
    before = mem.vectors.get_vector(a.id)
    b = mem.remember("the api runs on port 9090", scopes=["r2"], tags=["networking"])
    assert b.tags == ["api", "networking"] and b.scopes == ["r1", "r2"]
    assert mem.vectors.get_vector(a.id) != before  # embedded text changed


def test_set_tags_related_and_reindex(mem):
    a = mem.remember("prefix service rate limits", tags=["ipv6", "rate-limiting"])
    b = mem.remember("reprovision prefixes per tenant", tags=["ipv6", "multi-tenant"])
    c = mem.remember("tenant quotas", tags=["multi-tenant"])
    assert mem.related(a.id) == []
    mem.set_tags(c.id, add=["ipv6"])
    related = mem.related(b.id)
    assert [r["id"] for r in related] == [c.id] and related[0]["shared_tags"] == ["multi-tenant", "ipv6"]
    assert mem.set_tags(a.id, remove=["rate-limiting"]).tags == ["ipv6"]
    assert mem.set_tags(a.id, tags=["IPv6", "Prod"]).tags == ["ipv6", "prod"]
    assert "tags" in [h["action"] for h in mem.get(a.id)["history"]]
    with pytest.raises(KeyError):
        mem.set_tags("mem:nope", add=["x"])
    assert mem.reindex()["atoms"] == 3


def test_keyword_index_rebuilt_over_old_schema(tmp_path):
    db = tmp_path / "k.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE VIRTUAL TABLE atoms_fts USING fts5(content, atom_id UNINDEXED, task_id UNINDEXED, "
                 "host UNINDEXED, category UNINDEXED, expires_at UNINDEXED)")
    conn.commit()
    conn.close()
    mem = _make_mem(tmp_path, keyword_index=SqliteKeywordIndex(str(db)))
    mem.remember("ticket QQ-17 tracks the outage", tags=["incident-response"])
    assert mem.recall("QQ-17") and mem.recall("incident response")


def test_extraction_parses_tags_and_uses_vocabulary(tmp_path, monkeypatch):
    mem = _make_mem(tmp_path)
    mem.remember("existing", scopes=["r"], tags=["gcp", "new-region-build"])
    mem.log_turn("s", "user", "run tenant changes one at a time", task_id="r")
    seen = {}

    def fake_chat(base_url, model, messages, timeout=120.0, api_key="local"):
        seen["user"] = messages[1]["content"]
        return json.dumps([{"category": "learning", "content": LESSON, "scope": "task",
                            "provenance": "explicit_statement", "confidence": 0.9,
                            "tags": ["GCP", "new region build", "multi-tenant"]}])

    monkeypatch.setattr(extract_mod, "chat_completion", fake_chat)
    monkeypatch.setattr(distill_mod, "_enforce_global_trust_boundary", lambda c, t, m: c)
    distill_mod.distill_sessions(mem, ["s"])
    assert "Existing tags (reuse where they fit): gcp, new-region-build" in seen["user"]
    assert mem.graph.get_atom(mem._atom_id(LESSON)).tags == ["gcp", "new-region-build", "multi-tenant"]
    assert "3-6 lowercase-kebab-case tags" in extract_mod.EXTRACTION_SYSTEM_PROMPT


def test_retag_uses_the_session_transcript(tmp_path, monkeypatch):
    mem = _make_mem(tmp_path)
    mem.log_turn("claude-code:s1", "user", "We're building the new GCP region; reprovision IPv6 for all tenants")
    a = mem.remember(LESSON, session_id="claude-code:s1", scopes=["geoforge"])
    b = mem.remember("Prefix service has a 10 rps limit", session_id="claude-code:s1", scopes=["geoforge"])
    seen = {}

    def fake_chat(base_url, model, messages, timeout=120.0, api_key="local"):
        seen["system"], seen["user"] = messages[0]["content"], messages[1]["content"]
        return ('{"initiatives": ["new-region-build"], "tags": {"1": ["gcp", "new-region-build", "multi-tenant"], '
                '"2": ["ipv6", "Rate Limiting"], "9": ["x"]}}')

    monkeypatch.setattr(tagging, "chat_completion", fake_chat)
    report = mem.retag([a.id, b.id], session_id="claude-code:s1")
    assert report["tagged"] == 2 and report["errors"] == []
    assert "<transcript>" in seen["user"] and "reprovision IPv6" in seen["user"]
    assert "1. [fact] (geoforge)" in seen["user"]
    assert mem.graph.get_atom(a.id).tags == ["gcp", "new-region-build", "multi-tenant"]
    assert mem.graph.get_atom(b.id).tags == ["ipv6", "rate-limiting"]


def test_retag_bad_reply_keeps_tags_and_reports(mem, monkeypatch):
    a = mem.remember("x", tags=["keep-me"])
    monkeypatch.setattr(tagging, "chat_completion", lambda *a, **k: "Sure! Here are some thoughts.")
    report = mem.retag([a.id])
    assert report["tagged"] == 0 and "did not return a JSON object" in report["errors"][0]["error"]
    assert mem.graph.get_atom(a.id).tags == ["keep-me"]


def test_rest_client_and_cli(tmp_path, capsys, monkeypatch):
    mem = _make_mem(tmp_path)
    settings = replace(get_settings(), home=tmp_path, server_host="127.0.0.1", server_port=7777)
    with TestClient(create_app(mem, settings), base_url="http://127.0.0.1:7777") as http:
        remote = RemoteMemory("http://127.0.0.1:7777", client=http)
        a = remote.remember("tenant changes run sequentially", tags=["multi-tenant", "ipv6"])
        b = remote.remember("prefix service is rate limited", tags=["ipv6", "multi-tenant", "prod"])
        assert a["tags"] == ["multi-tenant", "ipv6"]
        assert remote.recall("tenant changes", tags=["ipv6"])[0]["tags"] == ["multi-tenant", "ipv6"]
        assert [x["id"] for x in remote.list(tag="prod")] == [b["id"]]
        assert remote.tag_counts()[0][1] == 2
        assert remote.related(a["id"])[0]["id"] == b["id"]
        assert remote.set_tags(a["id"], add=["gcp"])["tags"] == ["multi-tenant", "ipv6", "gcp"]
        assert http.post("/api/retag", json={}).status_code == 400
        assert remote.reindex()["atoms"] == 2

        monkeypatch.setattr(cli, "get_memory", lambda *k, **kw: remote)
        monkeypatch.setattr(remote, "close", lambda: None)
        assert cli.main(["tags"]) == 0
        assert "multi-tenant" in capsys.readouterr().out
        assert cli.main(["tag", a["id"], "--remove", "gcp"]) == 0
        assert capsys.readouterr().out.strip() == "multi-tenant, ipv6"
        assert cli.main(["list", "--tag", "prod", "--show-tags"]) == 0
        assert "#ipv6 #multi-tenant #prod" in capsys.readouterr().out
        assert cli.main(["show", a["id"]]) == 0
        out = capsys.readouterr().out
        assert "tags:       multi-tenant, ipv6" in out and "related (shared tags):" in out
        assert cli.main(["retag", "--dry-run"]) == 0
        assert "(no session)  2 memories" in capsys.readouterr().out


def test_tag_map_accepts_bare_and_wrapped_shapes():
    assert tagging._parse_tag_map('{"1": ["gcp"]}', 1) == {1: ["gcp"]}
    assert tagging._parse_tag_map('```json\n{"initiatives": ["x"], "tags": {"1": ["aws"]}}\n```', 1) == {1: ["aws"]}


# -- review items show tags; cleanup ----------------------------------------------------------


def test_review_items_carry_tags_and_cli_shows_them(mem, capsys, monkeypatch):
    a = mem.remember("roll out one tenant at a time", scopes=["geoforge"], tags=["ipv6-reprovisioning", "multi-tenant"])
    mem.queue_scope_review(a.id, "ambiguous", "unsure")
    (review,) = mem.pending_scope_reviews()
    assert review["tags"] == ["ipv6-reprovisioning", "multi-tenant"]
    monkeypatch.setattr(mem, "close", lambda: None)
    monkeypatch.setattr(cli, "get_memory", lambda *k, **kw: mem)
    assert cli.main(["review"]) == 0
    assert "tags: ipv6-reprovisioning, multi-tenant" in capsys.readouterr().out


def test_stoplist_filters_llm_tags_but_not_manual_ones(mem, monkeypatch):
    from memstrand.schema import clean_llm_tags

    assert clean_llm_tags(["Artifact", "gcp", "best-practice", "misc", "ipv6"]) == ["gcp", "ipv6"]
    monkeypatch.setattr(extract_mod, "chat_completion", lambda *a, **k: json.dumps(
        [{"category": "fact", "content": "x", "tags": ["fact", "gcp", "best-practice"]}]))
    assert extract_mod.extract_atoms([{"role": "user", "content": "hi"}], "u", "m")[0]["tags"] == ["gcp"]
    assert tagging._parse_tag_map('{"1": ["artifact", "aws"]}', 1) == {1: ["aws"]}
    assert mem.remember("manual", tags=["artifact", "gcp"]).tags == ["artifact", "gcp"]  # user's choice kept


def test_rename_merges_and_drop_removes(mem, capsys, monkeypatch):
    a = mem.remember("a", tags=["google-cloud", "ipv6"])
    b = mem.remember("b", tags=["gcp", "google-cloud"])
    c = mem.remember("c", tags=["artifact", "aws"])
    assert mem.rename_tag("google-cloud", "gcp") == 2
    assert mem.graph.get_atom(a.id).tags == ["gcp", "ipv6"]
    assert mem.graph.get_atom(b.id).tags == ["gcp"]  # merged, no duplicate
    assert mem.rename_tag("artifact", None) == 1
    assert mem.graph.get_atom(c.id).tags == ["aws"]
    with pytest.raises(ValueError):
        mem.rename_tag("!!", "x")
    monkeypatch.setattr(mem, "close", lambda: None)
    monkeypatch.setattr(cli, "get_memory", lambda *k, **kw: mem)
    assert cli.main(["tags", "rename", "aws", "amazon-web-services"]) == 0
    assert "on 1 memories" in capsys.readouterr().out
    assert cli.main(["tags", "drop", "ipv6"]) == 0
    assert "dropped ipv6 from 1 memories" in capsys.readouterr().out
    assert dict(mem.tag_counts()) == {"gcp": 2, "amazon-web-services": 1}


def test_rename_over_rest(tmp_path):
    mem = _make_mem(tmp_path)
    mem.remember("a", tags=["k8s"])
    settings = replace(get_settings(), home=tmp_path, server_host="127.0.0.1", server_port=7777)
    with TestClient(create_app(mem, settings), base_url="http://127.0.0.1:7777") as http:
        remote = RemoteMemory("http://127.0.0.1:7777", client=http)
        assert remote.rename_tag("k8s", "kubernetes") == 1
        assert remote.rename_tag("kubernetes", None) == 1
        assert http.post("/api/tags/rename", json={"old": 3}).status_code == 400
    assert mem.tag_counts() == []
