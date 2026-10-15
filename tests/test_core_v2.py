"""v0.2 "second brain" upgrade: scope+globals, TTL, hybrid recall, FSRS/
Hebbian side effects, append-only provenance, incremental distillation,
context injection, grounded answering, export/import, consolidation, and
config/embedding/thread-safety plumbing.

No network: the embedder is a deterministic hashing vectorizer (shared
pattern with tests/test_memory_flow.py), and every local-LLM call is
monkeypatched at the httpx layer.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time

import httpx
import pytest
from qdrant_client import QdrantClient

from memstrand import Category, Memory, Provenance
from memstrand.config import get_settings
from memstrand.pipeline.extract import _valid_atom
from memstrand.schema import MemoryEdge, RelationType
from memstrand.stores.graph_sqlite import SqliteGraphStore
from memstrand.stores.sqlite import SqliteSessionStore
from memstrand.stores.vector_qdrant import QdrantVectorStore


class HashEmbedder:
    """Deterministic keyword-overlap embeddings. Shared words => similar."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in t.lower().split():
                idx = int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim
                v[idx] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


@pytest.fixture()
def mem(tmp_path):
    """Default graph backend (sqlite) this time, to exercise it directly
    rather than only through tests/test_graph_stores.py."""
    return Memory(
        embedder=HashEmbedder(),
        vector_store=QdrantVectorStore(client=QdrantClient(":memory:")),
        graph_store=SqliteGraphStore(path=str(tmp_path / "graph.db")),
        session_store=SqliteSessionStore(path=str(tmp_path / "sessions.db")),
    )


# -- scope + globals -----------------------------------------------------------


def test_scope_sees_own_task_plus_globals(mem):
    mem.remember("always answer in markdown", category=Category.instruction, task_id=None)
    mem.remember("the staging gate requires two approvals", category=Category.fact, task_id="work")
    mem.remember("the home lab router is a UDM Pro", category=Category.fact, task_id="homelab")

    work_hits = {h["content"] for h in mem.recall("staging gate approvals markdown", task_id="work")}
    assert "the staging gate requires two approvals" in work_hits
    assert "always answer in markdown" in work_hits
    assert "the home lab router is a UDM Pro" not in work_hits

    unscoped_hits = {h["content"] for h in mem.recall("markdown staging gate router")}
    assert "always answer in markdown" in unscoped_hits
    assert "the staging gate requires two approvals" not in unscoped_hits
    assert "the home lab router is a UDM Pro" not in unscoped_hits


def test_scope_aware_ids_same_text_two_tasks(mem):
    a = mem.remember("the deploy key rotates monthly", task_id="task-a")
    b = mem.remember("the deploy key rotates monthly", task_id="task-b")
    assert a.id != b.id
    assert mem.recall("deploy key rotates", task_id="task-a") != []
    assert all(h["id"] != b.id for h in mem.recall("deploy key rotates", task_id="task-a"))


def test_re_remember_preserves_created_at(mem):
    first = mem.remember("the printer is in the supply closet", confidence=0.6)
    time.sleep(0.01)
    second = mem.remember("the printer is in the supply closet", confidence=0.95)
    assert second.id == first.id
    assert second.created_at == first.created_at
    assert second.updated_at >= first.updated_at
    assert second.confidence == 0.95


# -- TTL ------------------------------------------------------------------------


def test_ttl_expired_atoms_not_recalled_and_removed_by_consolidate(mem):
    atom = mem.remember("this coupon expires almost immediately", ttl_days=1e-7)
    time.sleep(0.1)
    assert mem.recall("coupon expires") == []
    # still physically present until consolidation sweeps it
    assert mem.graph.get_atom(atom.id) is not None

    report = mem.consolidate()
    assert report["expired"] >= 1
    assert mem.graph.get_atom(atom.id) is None


# -- hybrid keyword recall --------------------------------------------------------


def test_hybrid_keyword_catches_exact_code(mem):
    mem.remember("ticket ZX-4471 was resolved by restarting the service", category=Category.fact)
    mem.remember("the deployment pipeline failed twice this week", category=Category.fact)
    hits = mem.recall("ZX-4471", limit=5)
    hit = next(h for h in hits if "ZX-4471" in h["content"])
    assert hit["keyword"] is True


# -- absolute relevance (not per-result-set normalized) --------------------------


def test_lone_unrelated_hit_scores_low(mem):
    """Regression for a real smoke-test bug: relevance used to be min-max
    normalized WITHIN the result set, so a single irrelevant candidate (the
    only globally-visible atom) always normalized to 1.0. An unscoped
    `recall("ZX-4471")` returned an unrelated "I prefer dark mode"
    preference with score 0.936. With the HashEmbedder, a query that shares
    no words with the atom has cosine ~= 0 and no keyword match, so its
    absolute relevance -- and therefore its final blended score -- must be
    low regardless of being the only candidate."""
    mem.remember("I prefer dark mode everywhere", category=Category.preference)
    hits = mem.recall("ZX-4471 totally unrelated ticket code")
    assert len(hits) == 1
    assert hits[0]["score"] < 0.5


def test_min_relevance_kwarg_filters_low_relevance_hits(mem):
    mem.remember("I prefer dark mode everywhere", category=Category.preference)
    assert mem.recall("ZX-4471 totally unrelated ticket code") != []
    assert mem.recall("ZX-4471 totally unrelated ticket code", min_relevance=0.2) == []


def test_min_relevance_env_var_filters_low_relevance_hits(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMSTRAND_MIN_RELEVANCE", "0.2")
    mem = Memory(
        embedder=HashEmbedder(),
        vector_store=QdrantVectorStore(client=QdrantClient(":memory:")),
        graph_store=SqliteGraphStore(path=str(tmp_path / "graph.db")),
        session_store=SqliteSessionStore(path=str(tmp_path / "sessions.db")),
        settings=get_settings(),
    )
    mem.remember("I prefer dark mode everywhere", category=Category.preference)
    assert mem.recall("ZX-4471 totally unrelated ticket code") == []


# -- superseded / update -------------------------------------------------------


def test_update_creates_superseding_atom(mem):
    old = mem.remember("the office wifi password is hunter1", category=Category.fact)
    new = mem.update(old.id, content="the office wifi password is hunter2")
    assert new.id != old.id
    assert new.provenance == Provenance.corrected
    edges = [e for e in mem.graph.all_edges() if e.source_id == new.id and e.target_id == old.id]
    assert edges and edges[0].relation == RelationType.supersedes
    # the old atom stays queryable directly
    assert mem.get(old.id) is not None


def test_superseded_hidden_by_default_and_shown_with_flag(mem):
    old = mem.remember("the office wifi password is hunter1", category=Category.fact)
    new = mem.update(old.id, content="the office wifi password is hunter2")

    hits = mem.recall("office wifi password")
    ids = {h["id"] for h in hits}
    assert old.id not in ids
    assert new.id in ids

    hits_all = mem.recall("office wifi password", include_superseded=True)
    old_hit = next(h for h in hits_all if h["id"] == old.id)
    assert old_hit["superseded"] is True


# -- track flag: FSRS / Hebbian side effects -------------------------------------


def test_track_false_has_no_side_effects_track_true_increments_access(mem):
    atom = mem.remember("a tracked fact about the server", category=Category.fact)
    mem.recall("tracked fact server", track=False)
    mem.recall("tracked fact server", track=False)
    assert mem.graph.get_atom(atom.id).access_count == 0

    mem.recall("tracked fact server", track=True)
    assert mem.graph.get_atom(atom.id).access_count == 1
    mem.recall("tracked fact server", track=True)
    assert mem.graph.get_atom(atom.id).access_count == 2


def test_hebbian_strengthen_on_incoming_edge(mem):
    a = mem.remember("NVDA reports earnings in late November", category=Category.fact)
    b = mem.remember("NVDA is in the 21-ticker dip-buying universe", category=Category.fact)
    mem.graph.add_edge(
        MemoryEdge(source_id=a.id, target_id=b.id, relation=RelationType.informs, reason="test")
    )
    before = next(e.weight for e in mem.graph.all_edges() if e.source_id == a.id and e.target_id == b.id)
    mem.recall("NVDA earnings report", track=True)
    after = next(e.weight for e in mem.graph.all_edges() if e.source_id == a.id and e.target_id == b.id)
    assert after > before


# -- append-only provenance ledger -----------------------------------------------


def test_history_is_append_only(mem):
    old = mem.remember("the deploy runbook lives in the wiki", category=Category.fact)
    mem.update(old.id, content="the deploy runbook lives in Notion now")
    history = mem.sessions.history(old.id)
    actions = [h["action"] for h in history]
    assert actions == ["remember"]  # old atom's own ledger entry is untouched
    detail = mem.get(old.id)
    assert detail["history"] == history


# -- sessions / incremental distillation -----------------------------------------


class _FakeResp:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _fake_llm(url, headers=None, json=None, timeout=None):
    system = json["messages"][0]["content"]
    user_msg = json["messages"][1]["content"]
    if "extract durable memories" in system:
        content = "fallback"
        for line in user_msg.splitlines():
            if line.startswith("USER:"):
                content = line[len("USER:") :].strip()
        payload = [
            {
                "category": "preference",
                "content": content,
                "provenance": "explicit_statement",
                "confidence": 0.9,
            }
        ]
        return _FakeResp(json_dumps(payload))
    if "resolve one candidate memory" in system:
        return _FakeResp(
            json_dumps({"action": "add", "target_id": None, "relation": "related_to", "reason": "new info"})
        )
    raise AssertionError(f"unexpected prompt: {system[:80]}")


def json_dumps(obj):
    return json.dumps(obj)


def test_log_turn_incremental_improve_and_undistilled_tracking(mem, monkeypatch):
    monkeypatch.setattr(httpx, "post", _fake_llm)
    mem.log_turn("s1", "user", "I prefer dark mode everywhere", task_id="prefs")
    assert "s1" in mem.sessions.undistilled_sessions()

    report = mem.improve(["s1"])
    assert report["atoms"] == 1
    assert "s1" not in mem.sessions.undistilled_sessions()

    mem.log_turn("s1", "user", "I also like vim keybindings", task_id="prefs")
    assert "s1" in mem.sessions.undistilled_sessions()
    report2 = mem.improve(["s1"])
    assert report2["atoms"] == 1
    assert "s1" not in mem.sessions.undistilled_sessions()


def test_background_improve_busy_guard(mem):
    mem._improve_lock.acquire()
    try:
        result = mem.improve(None, background=True)
        assert result == {"queued": [], "busy": True}
    finally:
        mem._improve_lock.release()


def test_background_improve_runs_and_releases_lock(mem, monkeypatch):
    monkeypatch.setattr(httpx, "post", _fake_llm)
    mem.log_turn("s2", "user", "I prefer tabs over spaces", task_id="prefs")
    result = mem.improve(["s2"], background=True)
    assert result["queued"] == ["s2"]
    for _ in range(100):
        if not mem._improve_lock.locked():
            break
        time.sleep(0.02)
    assert not mem._improve_lock.locked()
    assert "s2" not in mem.sessions.undistilled_sessions()


# -- extraction scope routing (pure unit test) -----------------------------------


def test_extraction_scope_routing_defaults():
    assert _valid_atom({"category": "preference", "content": "x", "confidence": 0.9})["scope"] == "global"
    assert _valid_atom({"category": "instruction", "content": "x", "confidence": 0.9})["scope"] == "global"
    assert _valid_atom({"category": "fact", "content": "x", "confidence": 0.9})["scope"] == "task"
    assert (
        _valid_atom({"category": "fact", "content": "x", "scope": "global", "confidence": 0.9})["scope"]
        == "global"
    )


# -- context injection / grounded answering --------------------------------------


def test_context_includes_global_preferences_and_relevant_memories(mem):
    mem.remember("always respond concisely", category=Category.instruction, task_id=None)
    mem.remember("the api gateway lives at gw.internal", category=Category.fact, task_id="infra")

    ctx = mem.context("api gateway address", task_id="infra")
    assert "always respond concisely" in ctx
    assert "gw.internal" in ctx


def test_context_empty_when_nothing_to_say(mem):
    assert mem.context(task_id="never-used-task") == ""


def test_grounded_answer_llm_down_returns_message(mem, monkeypatch):
    mem.remember("the vpn endpoint is vpn.example.com", category=Category.fact)

    def _down(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", _down)
    answer = mem.grounded_answer("what is the vpn endpoint")
    # grounded_answer() now surfaces the real LLMUnavailable message rather
    # than a generic "unavailable" string (see test_review_fixes.py for the
    # transient/non-transient distinction this message comes from).
    assert "unreachable" in answer.lower()


def test_grounded_answer_no_memories_says_so(mem):
    assert "don't know" in mem.grounded_answer("anything at all").lower()


# -- export / import roundtrip ---------------------------------------------------


def test_export_import_roundtrip(mem, tmp_path):
    mem.remember("exported fact one", category=Category.fact, task_id="t1")
    a = mem.remember("exported fact two", category=Category.fact, task_id="t1")
    b = mem.remember("exported fact three", category=Category.fact, task_id="t1")
    mem.graph.add_edge(MemoryEdge(source_id=b.id, target_id=a.id, relation=RelationType.related_to))
    mem.log_turn("s-export", "user", "hello", task_id="t1")

    blob = mem.export("jsonl")

    mem2 = Memory(
        embedder=HashEmbedder(),
        vector_store=QdrantVectorStore(client=QdrantClient(":memory:")),
        graph_store=SqliteGraphStore(path=str(tmp_path / "graph2.db")),
        session_store=SqliteSessionStore(path=str(tmp_path / "sessions2.db")),
    )
    report = mem2.import_jsonl(blob)
    assert report["errors"] == []
    assert mem2.count() == mem.count()
    assert len(mem2.graph.all_edges()) == len(mem.graph.all_edges())
    hits = mem2.recall("exported fact", task_id="t1", limit=10)
    assert len(hits) == 3

    # idempotent: importing again doesn't duplicate turns/ledger
    report2 = mem2.import_jsonl(blob)
    assert report2["turns"] == 0
    assert report2["ledger"] == 0


def test_export_markdown_contains_categories(mem):
    mem.remember("a markdown-exportable fact", category=Category.fact)
    md = mem.export("markdown")
    assert "## fact" in md
    assert "a markdown-exportable fact" in md


# -- consolidation: near-duplicate linking + replay ------------------------------


def test_consolidate_links_near_duplicates_and_replays(mem):
    # 12 shared words + 1 extra word => cosine = sqrt(12/13) ~= 0.96 >= the
    # 0.95 near-duplicate threshold, under the HashEmbedder's word-overlap model.
    old = mem.remember(
        "the deploy pipeline for this service uses GitHub Actions and runs nightly",
        category=Category.fact,
    )
    time.sleep(0.01)
    new = mem.remember(
        "the deploy pipeline for this service uses GitHub Actions and runs nightly today",
        category=Category.fact,
    )

    report = mem.consolidate()
    assert report["duplicates_linked"] >= 1
    assert report["replayed"] >= 1
    superseded = mem.graph.superseded_ids([old.id, new.id])
    assert old.id in superseded  # newer (new) supersedes older (old)


# -- config -----------------------------------------------------------------------


def test_settings_env_parsing(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMSTRAND_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MEMSTRAND_LLM_MODEL", "my-model")
    monkeypatch.setenv("MEMSTRAND_EMBED_BACKEND", "openai")
    monkeypatch.setenv("MEMSTRAND_EMBED_MODEL", "my-embed-model")
    monkeypatch.setenv("MEMSTRAND_PORT", "9999")

    settings = get_settings()
    assert settings.home == (tmp_path / "home")
    assert settings.llm_model == "my-model"
    assert settings.embed_backend == "openai"
    assert settings.embed_model == "my-embed-model"
    assert settings.server_port == 9999
    assert settings.db_path == settings.home / "memstrand.db"
    assert settings.kuzu_path == settings.home / "kuzu"
    assert settings.qdrant_path == settings.home / "qdrant"
    # embed_base_url defaults to llm_base_url when unset
    assert settings.embed_base_url == settings.llm_base_url


# -- embedded on-disk Qdrant ------------------------------------------------------


def test_qdrant_embedded_on_disk_path_roundtrip(tmp_path):
    from memstrand.schema import MemoryAtom

    path = str(tmp_path / "qdrant-data")
    store = QdrantVectorStore(path=path, dim=4)
    atom = MemoryAtom(category=Category.fact, content="on-disk vector store works")
    store.upsert(atom, [1.0, 0.0, 0.0, 0.0])
    assert store.exists(atom.id)
    store.close()

    # re-opening the same path with a matching dim succeeds
    store2 = QdrantVectorStore(path=path, dim=4)
    assert store2.exists(atom.id)
    store2.close()


def test_qdrant_dim_mismatch_raises(tmp_path):
    path = str(tmp_path / "qdrant-data")
    store = QdrantVectorStore(path=path, dim=384)
    store.close()
    with pytest.raises(ValueError, match="dim mismatch"):
        QdrantVectorStore(path=path, dim=10)


def test_qdrant_local_mode_emits_no_payload_index_warning():
    """Local/embedded Qdrant (no `url=`) must never attempt payload index
    creation -- that always warns `UserWarning: Payload indexes have no
    effect in the local Qdrant`, which previously printed on every single
    `memstrand` CLI invocation. `:memory:` here is the "no dependency"
    default path, same as every other test fixture in this suite."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        store = QdrantVectorStore(client=QdrantClient(":memory:"))
        store.close()


def test_qdrant_embedded_on_disk_mode_emits_no_payload_index_warning(tmp_path):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        store = QdrantVectorStore(path=str(tmp_path / "qdrant-data"))
        store.close()


# -- thread safety smoke test -----------------------------------------------------


def test_concurrent_remember_and_recall_smoke():
    mem = Memory.in_memory(embedder=HashEmbedder())
    errors = []

    def worker(i: int) -> None:
        try:
            mem.remember(f"concurrent fact number {i}", category=Category.fact, task_id="concurrency")
            mem.recall("concurrent fact", task_id="concurrency", limit=3)
        except Exception as e:  # noqa: BLE001 - captured for the assertion below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert mem.count() == 16
