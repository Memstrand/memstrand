"""Scoping v2 (docs/design/scoping-v2.md): repo identity, multi-scope atoms,
content-only ids, scope management, migration, and scope-aware distillation."""

from __future__ import annotations

import hashlib
import json
import math
import socket
import subprocess
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from memstrand import Memory
from memstrand.export import to_jsonl
from memstrand.pipeline import distill as distill_mod
from memstrand.schema import (
    GLOBAL_SCOPE,
    Category,
    MemoryAtom,
    MemoryEdge,
    Provenance,
    RelationType,
    ScopedQuery,
    decode_scopes,
    encode_scopes,
    normalize_scopes,
    scopes_visible,
)
from memstrand.scopes import (
    AmbiguousScope,
    ScopeResolver,
    normalize_remote_url,
    resolve_alias,
    short_name,
)
from memstrand.stores.graph_kuzu import KuzuGraphStore
from memstrand.stores.graph_sqlite import SqliteGraphStore
from memstrand.stores.keyword_sqlite import SqliteKeywordIndex
from memstrand.stores.sqlite import SqliteSessionStore
from memstrand.stores.vector_qdrant import QdrantVectorStore


class HashEmbedder:
    """Deterministic keyword-overlap embeddings. Shared words => similar."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in t.lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


def _make_mem(tmp_path: Path, **kwargs) -> Memory:
    kwargs.setdefault("embedder", HashEmbedder())
    kwargs.setdefault("vector_store", QdrantVectorStore(client=QdrantClient(":memory:")))
    kwargs.setdefault("graph_store", SqliteGraphStore(path=str(tmp_path / "g.db")))
    kwargs.setdefault("session_store", SqliteSessionStore(path=str(tmp_path / "s.db")))
    kwargs.setdefault("keyword_index", SqliteKeywordIndex(str(tmp_path / "k.db")))
    kwargs.setdefault("home", tmp_path / "home")
    return Memory(**kwargs)


@pytest.fixture()
def mem(tmp_path):
    return _make_mem(tmp_path)


def _legacy_id(content: str, task_id: str | None, host: str | None = None) -> str:
    key = f"{content}\x1f{task_id or ''}\x1f{host or ''}"
    return f"mem:{hashlib.sha256(key.encode()).hexdigest()[:16]}"


# -- schema helpers -------------------------------------------------------------------


def test_normalize_encode_decode_roundtrip():
    assert normalize_scopes(None) == [GLOBAL_SCOPE]
    assert normalize_scopes(["b", "a", "a", None]) == ["", "a", "b"]
    for scopes in (["", "a"], ["a", "b"], [""], ["github.com/o/r"]):
        assert decode_scopes(encode_scopes(scopes)) == normalize_scopes(scopes)
    assert encode_scopes(["a", ""]) == "||a|"
    # legacy single task_id values
    assert decode_scopes("notes") == ["notes"]
    assert decode_scopes("") == [GLOBAL_SCOPE]
    assert decode_scopes(None) == [GLOBAL_SCOPE]


def test_scope_keys_reject_delimiter():
    with pytest.raises(ValueError):
        normalize_scopes(["a|b"])


def test_scopes_visible_rule():
    assert scopes_visible([""], [])
    assert scopes_visible(["", "a"], ["z"])
    assert scopes_visible(["a", "b"], ["b"])
    assert not scopes_visible(["a", "b"], ["c"])
    assert not scopes_visible(["a"], [])


def test_legacy_task_id_keyword_still_accepted():
    assert MemoryAtom(category=Category.fact, content="x", task_id="t").scopes == ["t"]
    assert MemoryAtom(category=Category.fact, content="x", task_id=None).scopes == [""]
    atom = MemoryAtom(category=Category.fact, content="x", scopes=["a", "b"])
    assert atom.task_id == "a"
    assert MemoryAtom(category=Category.fact, content="x", scopes=["", "a"]).task_id is None
    assert ScopedQuery(text="q", task_id="t").scopes == ["t"]
    assert ScopedQuery(text="q", task_id=None).scopes == []
    assert ScopedQuery(text="q", scopes=["", "a"]).scopes == ["a"]


# -- repo identity -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("git@github.com:Memstrand/memstrand.git", "github.com/memstrand/memstrand"),
        ("https://github.com/Memstrand/memstrand", "github.com/memstrand/memstrand"),
        ("https://user:secret@gitlab.corp/DevOps/Infra.git/", "gitlab.corp/DevOps/Infra"),
        ("ssh://git@gitlab.corp:2222/devops/infra", "gitlab.corp/devops/infra"),
        ("git@gitlab.corp:group/sub/repo.git", "gitlab.corp/group/sub/repo"),
        ("/srv/repos/x.git", None),
        ("file:///srv/x", None),
        ("", None),
    ],
)
def test_normalize_remote_url(url, expected):
    assert normalize_remote_url(url) == expected


def test_normalized_url_never_contains_credentials():
    assert "secret" not in normalize_remote_url("https://bob:secret@host.io/a/b")


def test_short_name_and_alias_resolution():
    assert short_name("github.com/org/repo") == "repo"
    assert short_name("local:mac:/Users/me/.claude") == ".claude"
    assert short_name("") == "global"
    known = ["github.com/a/infra", "gitlab.corp/b/infra", "github.com/a/memstrand", ""]
    assert resolve_alias("memstrand", known) == "github.com/a/memstrand"
    assert resolve_alias("github.com/a/infra", known) == "github.com/a/infra"
    assert resolve_alias("notes", known) == "notes"  # unknown -> literal named scope
    assert resolve_alias("global", known) == ""
    with pytest.raises(AmbiguousScope) as exc:
        resolve_alias("infra", known)
    assert set(exc.value.matches) == {"github.com/a/infra", "gitlab.corp/b/infra"}


def test_resolver_uses_remote_url_then_local_fallback(tmp_path):
    repo = tmp_path / "checkout-name-does-not-matter"
    (repo / "pkg").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    resolver = ScopeResolver(tmp_path / "cache.json", hostname="box")
    assert resolver.resolve(repo / "pkg") == f"local:box:{repo.resolve()}"
    subprocess.run(
        ["git", "-C", str(repo), "remote", "add", "origin", "https://github.com/Org/Repo.git"], check=True
    )
    assert resolver.resolve(repo / "pkg") == "github.com/org/repo"
    resolver.save()
    cached = json.loads((tmp_path / "cache.json").read_text())
    assert cached[str(repo.resolve())]["scope"] == "github.com/org/repo"
    plain = tmp_path / "plain"
    plain.mkdir()
    assert resolver.resolve(plain) == f"local:box:{plain.resolve()}"
    assert ScopeResolver().hostname == socket.gethostname()


# -- multi-scope recall ----------------------------------------------------------------


def test_multi_scope_atom_visible_in_each_scope_only(mem):
    mem.remember("blue green deploys use the canary pool", scopes=["repo-a", "repo-b"])
    assert mem.recall("canary pool deploys", scopes=["repo-a"])
    assert mem.recall("canary pool deploys", scopes=["repo-b"])
    assert mem.recall("canary pool deploys", task_id="repo-b")
    assert mem.recall("canary pool deploys", scopes=["repo-c"]) == []
    assert mem.recall("canary pool deploys") == []
    # a query spanning several scopes sees the union
    mem.remember("repo c pins terraform 1.9", scopes=["repo-c"])
    hits = mem.recall("canary pool terraform", scopes=["repo-a", "repo-c"])
    assert {h["content"] for h in hits} >= {
        "blue green deploys use the canary pool",
        "repo c pins terraform 1.9",
    }


def test_keyword_only_hit_respects_scopes(mem):
    mem.remember("ticket ZX-4471 tracks the outage", scopes=["repo-a", "repo-b"])
    assert any("ZX-4471" in h["content"] for h in mem.recall("ZX-4471", scopes=["repo-b"]))
    assert mem.recall("ZX-4471", scopes=["repo-z"]) == []


def test_remember_unions_scopes_and_global_is_a_member(mem):
    a = mem.remember("the build cache lives in s3", task_id="repo-a")
    b = mem.remember("the build cache lives in s3")  # global
    assert a.id == b.id
    assert b.scopes == ["", "repo-a"]
    assert mem.recall("build cache s3")  # visible unscoped now
    assert mem.list(task_id="repo-a")[0]["id"] == a.id
    assert mem.list(task_id="")[0]["id"] == a.id


def test_graph_neighbor_bridge_respects_multi_scope(mem):
    a = mem.remember("service alpha calls the billing api", scopes=["repo-a"])
    hidden = mem.remember("billing api secret rotation runbook", scopes=["repo-b"])
    shared = mem.remember("billing api owners are team payments", scopes=["repo-a", "repo-b"])
    for other in (hidden, shared):
        mem.graph.add_edge(MemoryEdge(source_id=a.id, target_id=other.id, relation=RelationType.related_to))
    hit = next(h for h in mem.recall("service alpha billing", scopes=["repo-a"]) if h["id"] == a.id)
    neighbor_ids = {n["id"] for n in hit["neighbors"]}
    assert shared.id in neighbor_ids
    assert hidden.id not in neighbor_ids


def test_standing_instructions_follow_scopes(mem):
    mem.remember("always run make lint before pushing", category=Category.instruction, scopes=["repo-a"])
    assert "make lint" in mem.context(None, scopes=["repo-a"])
    assert "make lint" not in mem.context(None, scopes=["repo-b"])


# -- scope management ------------------------------------------------------------------


def test_set_scopes_add_remove_replace(mem):
    atom = mem.remember("prod deploys need two approvals", scopes=["repo-a"])
    assert mem.set_scopes(atom.id, add=["repo-b"]).scopes == ["repo-a", "repo-b"]
    assert mem.recall("prod deploys approvals", scopes=["repo-b"])
    assert mem.set_scopes(atom.id, remove=["repo-a"]).scopes == ["repo-b"]
    assert mem.recall("prod deploys approvals", scopes=["repo-a"]) == []
    assert mem.set_scopes(atom.id, scopes=[""]).scopes == [""]
    assert mem.recall("prod deploys approvals")
    with pytest.raises(ValueError):
        mem.set_scopes(atom.id, remove=[""])
    with pytest.raises(ValueError):
        mem.set_scopes(atom.id, scopes=[])
    with pytest.raises(KeyError):
        mem.set_scopes("mem:nope", add=["x"])
    actions = [h["action"] for h in mem.get(atom.id)["history"]]
    assert actions.count("scope") == 3


def test_rename_scope_and_summary(mem):
    mem.remember("uses poetry", scopes=["infra"])
    mem.remember("uses ruff", scopes=["infra", "other"])
    mem.remember("likes tables")
    assert mem.scopes_summary() == {"infra": 2, "": 1, "other": 1}
    assert mem.rename_scope("infra", "github.com/org/infra") == 2
    assert mem.scopes_summary()["github.com/org/infra"] == 2
    assert "infra" not in mem.scopes_summary()
    assert mem.recall("uses poetry", scopes=["github.com/org/infra"])


def test_forget_source_removes_one_scope_then_deletes(mem):
    atom = mem.remember("chunk from notes", source="/n.md", scopes=["a", "b"])
    assert mem.forget_source("/n.md", task_id="a") == 1
    assert mem.graph.get_atom(atom.id).scopes == ["b"]
    assert mem.forget_source("/n.md", task_id="zzz") == 0
    assert mem.forget_source("/n.md", task_id="b") == 1
    assert mem.graph.get_atom(atom.id) is None


def test_forget_source_never_makes_an_atom_global(mem):
    atom = mem.remember("chunk", source="/x.md", scopes=["a"])
    mem.forget_source("/x.md", task_id="a")
    assert mem.graph.get_atom(atom.id) is None
    assert mem.recall("chunk") == []


# -- store parity -------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["sqlite", "kuzu"])
def test_graph_store_scope_membership_filter(tmp_path, backend):
    store = SqliteGraphStore(str(tmp_path / "g.db")) if backend == "sqlite" else KuzuGraphStore(str(tmp_path / "k"))
    rows = {
        "multi": ["a", "b"],
        "glob_a": ["", "a"],
        "glob": [""],
        "solo": ["c"],
    }
    for aid, scopes in rows.items():
        store.add_atom(MemoryAtom(id=aid, category=Category.fact, content=aid, scopes=scopes))
    assert store.get_atom("multi").scopes == ["a", "b"]
    assert store.get_atom("glob_a").scopes == ["", "a"]
    assert {a.id for a in store.list_atoms(task_id="a")} == {"multi", "glob_a"}
    assert {a.id for a in store.list_atoms(task_id="b")} == {"multi"}
    assert {a.id for a in store.list_atoms(task_id="")} == {"glob_a", "glob"}
    assert store.count_atoms(task_id="c") == 1
    assert store.count_atoms() == 4
    store.close()


def test_sqlite_graph_reads_and_filters_legacy_rows(tmp_path):
    store = SqliteGraphStore(str(tmp_path / "g.db"))
    store.add_atom(MemoryAtom(id="x", category=Category.fact, content="x", scopes=["a"]))
    with store._lock:
        store.conn.execute("UPDATE atoms SET task_id='legacy' WHERE id='x'")
        store.conn.commit()
    assert store.get_atom("x").scopes == ["legacy"]
    assert [a.id for a in store.list_atoms(task_id="legacy")] == ["x"]


def test_keyword_and_vector_match_legacy_single_task_rows(tmp_path):
    kw = SqliteKeywordIndex(str(tmp_path / "k.db"))
    with kw._lock:
        kw.conn.execute(
            "INSERT INTO atoms_fts(content, atom_id, task_id, host, category, expires_at) "
            "VALUES ('legacy marker QQ-17', 'old', 'repo-a', '', 'fact', 0.0)"
        )
        kw.conn.commit()
    assert kw.search("QQ-17", ScopedQuery(text="QQ-17", scopes=["repo-a"])) != []
    assert kw.search("QQ-17", ScopedQuery(text="QQ-17", scopes=["repo-b"])) == []

    vs = QdrantVectorStore(client=QdrantClient(":memory:"))
    vec = HashEmbedder().embed(["legacy marker"])[0]
    vs.upsert(MemoryAtom(id="new", category=Category.fact, content="x", scopes=["repo-a"]), vec)
    from qdrant_client.models import PointStruct

    from memstrand.stores.vector_qdrant import COLLECTION, _point_id

    vs.client.upsert(
        COLLECTION,
        points=[
            PointStruct(
                id=_point_id("old"), vector=vec,
                payload={"atom_id": "old", "task_id": "repo-a", "host": "", "category": "fact", "expires_at": 0.0},
            )
        ],
    )
    ids = {aid for aid, _ in vs.search(vec, ScopedQuery(text="q", scopes=["repo-a"]))}
    assert ids == {"new", "old"}
    assert vs.search(vec, ScopedQuery(text="q", scopes=["repo-b"])) == []
    assert vs.get_vector("new") is not None
    assert vs.get_vector("missing") is None


# -- migration -------------------------------------------------------------------------------


def _seed_legacy_store(mem: Memory) -> dict:
    """Write pre-v2 atoms straight into the stores (task-hashed ids)."""
    emb = mem.embedder
    atoms = {
        "a": MemoryAtom(id=_legacy_id("shared fact", "repo-a"), category=Category.fact,
                        content="shared fact", task_id="repo-a", confidence=0.6),
        "b": MemoryAtom(id=_legacy_id("shared fact", "repo-b"), category=Category.fact,
                        content="shared fact", task_id="repo-b", confidence=0.9),
        "g": MemoryAtom(id=_legacy_id("global fact", None), category=Category.fact,
                        content="global fact"),
        "other": MemoryAtom(id=_legacy_id("other fact", "repo-a"), category=Category.fact,
                            content="other fact", task_id="repo-a"),
    }
    for atom in atoms.values():
        mem.vectors.upsert(atom, emb.embed([atom.content])[0])
        mem.graph.add_atom(atom)
        mem.keyword.add(atom)
    mem.graph.add_edge(MemoryEdge(source_id=atoms["other"].id, target_id=atoms["a"].id,
                                  relation=RelationType.related_to))
    mem.sessions.log_provenance(atoms["a"].id, "remember", "explicit_statement")
    return atoms


def test_migration_rekeys_and_merges_legacy_atoms(tmp_path):
    mem = _make_mem(tmp_path)
    atoms = _seed_legacy_store(mem)
    mem.sessions.set_meta("scoping_version", "1")  # pretend pre-v2

    report = mem._migrate_scoping_v2()
    assert report["migrated"] == 3  # a + b merge, `other` moves, global unchanged

    shared_id = mem._atom_id("shared fact")
    merged = mem.graph.get_atom(shared_id)
    assert merged.scopes == ["repo-a", "repo-b"]
    assert merged.confidence == 0.9
    for old in ("a", "b", "other"):
        assert mem.graph.get_atom(atoms[old].id) is None
    assert mem.graph.get_atom(atoms["g"].id) is not None  # global ids never change
    other_id = mem._atom_id("other fact")
    assert any(e.source_id == other_id and e.target_id == shared_id for e in mem.graph.all_edges())
    history = [h["action"] for h in mem.get(shared_id)["history"]]
    assert "remember" in history and "migrate" in history
    assert list((mem.home / "backups").glob("pre-scoping-v2-*.jsonl"))
    assert mem.recall("shared fact", scopes=["repo-b"])[0]["id"] == shared_id

    # idempotent: marker set, a second run does nothing
    assert mem.sessions.get_meta("scoping_version") == "2"
    assert mem._migrate_scoping_v2() == {"migrated": 0}


def test_relearning_after_migration_does_not_duplicate(tmp_path):
    mem = _make_mem(tmp_path)
    _seed_legacy_store(mem)
    mem.sessions.set_meta("scoping_version", "1")
    mem._migrate_scoping_v2()
    before = mem.count()
    mem.remember("shared fact", task_id="repo-c")
    assert mem.count() == before
    assert mem.graph.get_atom(mem._atom_id("shared fact")).scopes == ["repo-a", "repo-b", "repo-c"]


def test_fresh_store_marks_migrated_without_backup(tmp_path):
    mem = _make_mem(tmp_path)
    assert mem.sessions.get_meta("scoping_version") == "2"
    assert not (mem.home / "backups").exists()


def test_import_of_v02_export_is_rekeyed(tmp_path):
    src = _make_mem(tmp_path / "src")
    _seed_legacy_store(src)
    text = to_jsonl(src).replace('"version": "0.3"', '"version": "0.2"')
    dst = _make_mem(tmp_path / "dst")
    report = dst.import_jsonl(text)
    assert report["rekeyed"] == 3
    assert dst.graph.get_atom(dst._atom_id("shared fact")).scopes == ["repo-a", "repo-b"]


def test_import_merges_scopes_into_existing_atom(tmp_path):
    src = _make_mem(tmp_path / "src")
    src.remember("deploy window is tuesday", task_id="repo-a")
    dst = _make_mem(tmp_path / "dst")
    dst.remember("deploy window is tuesday", task_id="repo-b")
    report = dst.import_jsonl(src.export("jsonl"))
    assert "rekeyed" not in report
    assert dst.graph.get_atom(dst._atom_id("deploy window is tuesday")).scopes == ["repo-a", "repo-b"]


# -- distillation ----------------------------------------------------------------------------


def _distill_with(monkeypatch, mem, session_id, task_id, candidates, verdict=None):
    mem.log_turn(session_id, "user", "some turn", task_id=task_id)
    monkeypatch.setattr(distill_mod, "extract_atoms", lambda *a, **k: [dict(c) for c in candidates])
    monkeypatch.setattr(distill_mod, "_enforce_global_trust_boundary", lambda c, t, m: c)
    monkeypatch.setattr(
        distill_mod, "_judge",
        lambda m, c, similar: verdict(similar) if verdict else
        {"action": "add", "target_id": None, "relation": "related_to", "reason": ""},
    )
    return distill_mod.distill_sessions(mem, [session_id])


def _cand(content, scope="task", category=Category.fact):
    return {"category": category, "content": content, "provenance": Provenance.explicit_statement,
            "confidence": 0.8, "scope": scope}


def test_distill_widens_scope_of_fact_seen_in_another_repo(tmp_path, monkeypatch):
    mem = _make_mem(tmp_path)
    r1 = _distill_with(monkeypatch, mem, "s1", "repo-a", [_cand("we tag releases with calver")])
    assert r1["atoms"] == 1
    r2 = _distill_with(monkeypatch, mem, "s2", "repo-b", [_cand("we tag releases with calver")])
    assert r2["scope_widened"] == 1 and r2["atoms"] == 0
    assert mem.graph.get_atom(mem._atom_id("we tag releases with calver")).scopes == ["repo-a", "repo-b"]
    r3 = _distill_with(monkeypatch, mem, "s3", "repo-b", [_cand("we tag releases with calver")])
    assert r3["skipped_dedup"] == 1


def test_distill_correction_in_one_repo_splits_instead_of_superseding(tmp_path, monkeypatch):
    mem = _make_mem(tmp_path)
    old = mem.remember("deploys run via make deploy", scopes=["repo-a", "repo-b"])
    report = _distill_with(
        monkeypatch, mem, "s1", "repo-b", [_cand("deploys run via the deploy workflow")],
        verdict=lambda similar: {"action": "update", "target_id": old.id, "relation": "related_to", "reason": "changed"},
    )
    assert report["atoms"] == 1 and report["errors"] == []
    old_after = mem.graph.get_atom(old.id)
    assert old_after.scopes == ["repo-a"]  # still current in repo-a
    assert old.id not in mem.graph.superseded_ids([old.id])
    edges = [e for e in mem.graph.all_edges() if e.target_id == old.id]
    assert [e.relation for e in edges] == [RelationType.contradicts]
    assert "still current in repo-a" in edges[0].reason
    assert mem.recall("deploys run via make deploy", scopes=["repo-a"])[0]["id"] == old.id


def test_distill_correction_covering_all_scopes_supersedes(tmp_path, monkeypatch):
    mem = _make_mem(tmp_path)
    old = mem.remember("the api runs on port 8080", scopes=["repo-a"])
    _distill_with(
        monkeypatch, mem, "s1", "repo-a", [_cand("the api runs on port 9090")],
        verdict=lambda similar: {"action": "update", "target_id": old.id, "relation": "related_to", "reason": ""},
    )
    assert old.id in mem.graph.superseded_ids([old.id])


def test_task_scoped_correction_never_retires_a_global_fact(tmp_path, monkeypatch):
    mem = _make_mem(tmp_path)
    old = mem.remember("timezone for reports is UTC")
    _distill_with(
        monkeypatch, mem, "s1", "repo-a", [_cand("timezone for reports is PST")],
        verdict=lambda similar: {"action": "update", "target_id": old.id, "relation": "related_to", "reason": ""},
    )
    assert old.id not in mem.graph.superseded_ids([old.id])
    assert mem.graph.get_atom(old.id).scopes == [""]
    assert mem.recall("timezone for reports")  # global fact still visible unscoped


# -- REST + RemoteMemory parity -----------------------------------------------------------------


def test_rest_scopes_roundtrip_via_remote_memory(tmp_path):
    from dataclasses import replace

    from starlette.testclient import TestClient

    from memstrand.client import RemoteMemory
    from memstrand.config import get_settings
    from memstrand.server import create_app

    mem = _make_mem(tmp_path)
    settings = replace(get_settings(), home=tmp_path, server_host="127.0.0.1", server_port=7777)
    app = create_app(mem, settings)
    with TestClient(app, base_url="http://127.0.0.1:7777") as http:
        remote = RemoteMemory("http://127.0.0.1:7777", client=http)
        atom = remote.remember("runbooks live in confluence space OPS", scopes=["repo-a", "repo-b"])
        assert atom["scopes"] == ["repo-a", "repo-b"]
        assert remote.recall("runbooks confluence", scopes=["repo-b"])[0]["scopes"] == ["repo-a", "repo-b"]
        assert remote.recall("runbooks confluence", scopes=["repo-z"]) == []
        assert "confluence" in remote.context("runbooks", scopes=["repo-a"])
        assert remote.get(atom["id"], scoped=True, scopes=["repo-z"]) is None
        assert remote.get(atom["id"], scoped=True, scopes=["repo-a"])["id"] == atom["id"]
        assert remote.set_scopes(atom["id"], add=[""])["scopes"] == ["", "repo-a", "repo-b"]
        assert remote.scopes_summary() == {"": 1, "repo-a": 1, "repo-b": 1}
        assert remote.rename_scope("repo-a", "github.com/org/a") == 1
        assert "github.com/org/a" in remote.scopes_summary()
        resp = http.post(f"/api/atoms/{atom['id']}/scopes", json={"set": []})
        assert resp.status_code == 400  # an explicitly empty set is refused (use [""] for global)
        resp = http.post(f"/api/atoms/{atom['id']}/scopes", json={"remove": ["", "github.com/org/a", "repo-b"]})
        assert resp.status_code == 400
        assert http.post("/api/atoms/mem:nope/scopes", json={"add": ["x"]}).status_code == 404
        assert http.post("/api/recall", json={"query": "x", "scopes": "notalist-but-ok"}).status_code == 200
        assert http.post("/api/recall", json={"query": "x", "scopes": [1]}).status_code == 400


# -- CLI ---------------------------------------------------------------------------------------


@pytest.fixture()
def cli_mem(monkeypatch, tmp_path):
    from memstrand import cli

    mem = _make_mem(tmp_path)
    monkeypatch.setattr(mem, "close", lambda: None)
    monkeypatch.setattr(cli, "get_memory", lambda *a, **k: mem)
    return mem


def test_cli_repo_flag_short_names_paths_and_scope_commands(cli_mem, capsys, tmp_path):
    from memstrand import cli

    repo = tmp_path / "work" / "whatever-folder"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "remote", "add", "origin", "git@github.com:acme/widgets.git"], check=True
    )

    assert cli.main(["remember", "widgets uses pnpm", "--repo", str(repo)]) == 0
    atom_id = capsys.readouterr().out.strip()
    assert cli_mem.graph.get_atom(atom_id).scopes == ["github.com/acme/widgets"]

    # short name resolves against known scopes
    assert cli.main(["recall", "pnpm", "--repo", "widgets"]) == 0
    out = capsys.readouterr().out
    assert "widgets uses pnpm" in out and "scope=widgets" in out

    assert cli.main(["scope", atom_id, "--add", "notes", "--global"]) == 0
    assert capsys.readouterr().out.split() == ["global", "github.com/acme/widgets", "notes"]
    assert cli.main(["scope", atom_id, "--remove", "global"]) == 0
    capsys.readouterr()

    assert cli.main(["list", "--repo", "widgets"]) == 0
    assert "(widgets, notes)" in capsys.readouterr().out

    assert cli.main(["scopes"]) == 0
    out = capsys.readouterr().out
    assert "github.com/acme/widgets" in out and "notes" in out

    assert cli.main(["scopes", "rename", "notes", "kb"]) == 0
    assert "renamed 1" in capsys.readouterr().out
    assert cli.main(["scope", atom_id]) == 0
    assert capsys.readouterr().out.split() == ["github.com/acme/widgets", "kb"]


def test_cli_ambiguous_short_name_is_a_clean_error(cli_mem, capsys):
    from memstrand import cli

    cli_mem.remember("a", scopes=["github.com/one/infra"])
    cli_mem.remember("b", scopes=["gitlab.corp/two/infra"])
    assert cli.main(["recall", "a", "--repo", "infra"]) == 1
    err = capsys.readouterr().err
    assert "ambiguous" in err and "github.com/one/infra" in err


# -- review fixes --------------------------------------------------------------------------


def test_same_text_from_different_sources_stays_separate(mem):
    """Review blocker: one atom has ONE source, so identical text from two
    files must not merge (the later file would overwrite the first's
    provenance, leak its path into the other scope, and break
    forget_source for the first file)."""
    a = mem.remember("Always use TDD", source="/repoA/notes.md", task_id="repoA")
    b = mem.remember("Always use TDD", source="/repoB/readme.md", task_id="repoB")
    assert a.id != b.id
    assert mem.graph.ids_by_source("/repoA/notes.md") == [a.id]
    hit = mem.recall("TDD", task_id="repoA")[0]
    assert hit["source"] == "/repoA/notes.md"
    assert mem.forget_source("/repoA/notes.md", task_id="repoA") == 1
    assert mem.recall("TDD", task_id="repoA") == []
    assert mem.recall("TDD", task_id="repoB")[0]["source"] == "/repoB/readme.md"


def test_sourceless_global_ids_unchanged_from_pre_v2():
    assert Memory._atom_id("x") == _legacy_id("x", None)
    assert Memory._atom_id("x", source="/f") != _legacy_id("x", None)


def test_widening_merge_never_downgrades_existing_metadata(mem):
    a = mem.remember(
        "Always run the linter before committing",
        category=Category.instruction, provenance=Provenance.explicit_statement,
        confidence=0.9, scopes=["repo-a"],
    )
    b = mem.remember(
        "Always run the linter before committing",
        category=Category.observation, provenance=Provenance.inferred,
        confidence=0.3, scopes=["repo-b"],
    )
    assert b.id == a.id and b.scopes == ["repo-a", "repo-b"]
    stored = mem.graph.get_atom(a.id)
    assert (stored.category, stored.provenance, stored.confidence) == (
        Category.instruction, Provenance.explicit_statement, 0.9,
    )
    assert "linter" in mem.context(None, scopes=["repo-a"])  # still a trusted standing instruction
    # re-remembering within existing scopes is still an in-place update
    c = mem.remember("Always run the linter before committing", confidence=0.5, scopes=["repo-a"])
    assert c.confidence == 0.5 and c.scopes == ["repo-a", "repo-b"]


def test_migration_rekeys_sourced_atoms_and_keeps_them_apart(tmp_path):
    mem = _make_mem(tmp_path)
    for task, src in (("t1", "/a.md"), ("t2", "/b.md")):
        atom = MemoryAtom(id=_legacy_id("same chunk", task), category=Category.fact,
                          content="same chunk", task_id=task, source=src)
        mem.vectors.upsert(atom, mem.embedder.embed([atom.content])[0])
        mem.graph.add_atom(atom)
        mem.keyword.add(atom)
    mem.sessions.set_meta("scoping_version", "1")
    mem._migrate_scoping_v2()
    assert mem.graph.get_atom(mem._atom_id("same chunk", source="/a.md")).scopes == ["t1"]
    assert mem.graph.get_atom(mem._atom_id("same chunk", source="/b.md")).scopes == ["t2"]


def test_copy_history_is_idempotent(tmp_path):
    store = SqliteSessionStore(str(tmp_path / "s.db"))
    store.log_provenance("old", "remember", "explicit_statement", ts=1.0)
    assert store.copy_history("old", "new") == 1
    assert store.copy_history("old", "new") == 0
    assert len(store.history("new")) == 1


def test_scoped_forget_source_over_rest_and_cli(tmp_path, capsys, monkeypatch):
    from dataclasses import replace

    from starlette.testclient import TestClient

    from memstrand import cli
    from memstrand.client import RemoteMemory
    from memstrand.config import get_settings
    from memstrand.server import create_app

    mem = _make_mem(tmp_path)
    atom = mem.remember("chunk", source="/n.md", scopes=["a", "b"])
    settings = replace(get_settings(), home=tmp_path, server_host="127.0.0.1", server_port=7777)
    with TestClient(create_app(mem, settings), base_url="http://127.0.0.1:7777") as http:
        remote = RemoteMemory("http://127.0.0.1:7777", client=http)
        monkeypatch.setattr(cli, "get_memory", lambda *a, **k: remote)
        monkeypatch.setattr(remote, "close", lambda: None)
        assert cli.main(["forget-source", "/n.md", "--task", "a"]) == 0
        assert "removed scope a from 1" in capsys.readouterr().out
        assert mem.graph.get_atom(atom.id).scopes == ["b"]
        assert cli.main(["forget-source", "/n.md"]) == 0
        assert mem.graph.get_atom(atom.id) is None
