"""Acceptance tests: the 3-test rubric + the Playwright no-leak fix.

Runs against real kuzu + real sqlite + in-memory Qdrant. The only fake
is the embedder: a deterministic hashing vectorizer (keyword overlap),
because the real fastembed model needs a network download.
"""

import hashlib
import math

import httpx
import pytest
from qdrant_client import QdrantClient

from memstrand import Category, Memory, Provenance
from memstrand.pipeline.extract import LLMUnavailable, extract_atoms
from memstrand.schema import MemoryEdge, RelationType
from memstrand.stores.graph_kuzu import KuzuGraphStore
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
    return Memory(
        embedder=HashEmbedder(),
        vector_store=QdrantVectorStore(client=QdrantClient(":memory:")),
        graph_store=KuzuGraphStore(path=str(tmp_path / "kuzu")),
        session_store=SqliteSessionStore(path=str(tmp_path / "m.db")),
    )


# -- the 3-test rubric -------------------------------------------------------


def test_rubric_cross_session_recall(mem):
    mem.remember(
        "Chiran's mini PC is named cognita-pulse",
        category=Category.fact,
        provenance=Provenance.explicit_statement,
        confidence=0.95,
        session_id="s1",
        task_id="homelab",
    )
    hits = mem.recall("what is the mini PC called?", session_id="s2", task_id="homelab")
    assert hits, "same-task cross-session recall returned nothing"
    assert any("cognita-pulse" in (h["content"] or "") for h in hits)
    assert hits[0]["category"] == "fact"


def test_rubric_no_cross_task_leak(mem):
    mem.remember(
        "Playwright 1.63 installed for agent testing",
        category=Category.fact,
        task_id="agent-testing",
    )
    # different task: must see nothing
    assert mem.recall("playwright", task_id="disk-cleanup") == []
    # UNSCOPED query: must also see nothing (the real Playwright fix)
    assert mem.recall("playwright") == []
    # the owning task CAN see it
    hits = mem.recall("playwright", task_id="agent-testing")
    assert len(hits) == 1


def test_task_scoped_atoms_visible_to_same_task_any_session(mem):
    mem.remember("deploys go through the staging gate", task_id="work")
    assert len(mem.recall("staging gate deploy", task_id="work", session_id="s9")) == 1
    assert mem.recall("staging gate deploy", task_id="other") == []


# -- graph: two strands joined ------------------------------------------------


def test_recall_hydrates_neighbors(mem):
    a = mem.remember("NVDA reports earnings in late November", category=Category.fact)
    b = mem.remember("NVDA is in the 21-ticker dip-buying universe", category=Category.fact)
    mem.graph.add_edge(
        MemoryEdge(source_id=a.id, target_id=b.id, relation=RelationType.informs, reason="test")
    )
    hits = mem.recall("NVDA earnings report")
    by_id = {h["id"]: h for h in hits}
    assert a.id in by_id
    neighbor_ids = [n["id"] for n in by_id[a.id]["neighbors"]]
    assert b.id in neighbor_ids


def test_contradictions_are_additive(mem):
    old = mem.remember("the mini PC runs Ubuntu 22.04", category=Category.fact)
    new = mem.remember("the mini PC runs Ubuntu 24.04", category=Category.fact)
    mem.graph.add_edge(
        MemoryEdge(
            source_id=new.id,
            target_id=old.id,
            relation=RelationType.contradicts,
            reason="OS upgraded",
        )
    )
    hits = mem.recall("mini PC Ubuntu version")
    by_id = {h["id"]: h for h in hits}
    # both sides stay readable...
    assert old.id in by_id and new.id in by_id
    # ...and the contradiction is surfaced, not hidden
    assert any(c["id"] == old.id for c in by_id[new.id]["contradictions"])


def test_forget_cleans_graph_too(mem):
    a = mem.remember("temporary note to forget", category=Category.fact)
    mem.forget(a.id)
    assert mem.recall("temporary note") == []
    assert mem.graph.get_atom(a.id) is None


# -- extraction / distillation with a fake local LLM --------------------------


class _FakeResp:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _fake_llm(url, headers=None, json=None, timeout=None):
    system = json["messages"][0]["content"]
    if "extract durable memories" in system:
        return _FakeResp(
            '[{"category": "preference", "content": "Chiran prefers dark mode", '
            '"provenance": "explicit_statement", "confidence": 0.9}, '
            '{"category": "bogus", "content": "dropped by schema check", '
            '"provenance": "inferred", "confidence": 0.5}]'
        )
    if "resolve one candidate memory" in system:
        return _FakeResp(
            '{"action": "add", "target_id": null, '
            '"relation": "related_to", "reason": "new information"}'
        )
    raise AssertionError(f"unexpected prompt: {system[:80]}")


def test_extract_validates_schema_contract(mem, monkeypatch):
    monkeypatch.setattr(httpx, "post", _fake_llm)
    atoms = extract_atoms(
        [{"role": "user", "content": "I prefer dark mode everywhere"}],
        "http://localhost:8081/v1",
        "local",
    )
    assert len(atoms) == 1  # the bogus-category candidate is dropped
    assert atoms[0]["category"] == Category.preference
    assert atoms[0]["confidence"] == 0.9


def test_extract_llm_down_raises(mem, monkeypatch):
    def _down(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", _down)
    with pytest.raises(LLMUnavailable):
        extract_atoms([{"role": "user", "content": "hi"}], "http://x/v1", "local")


def test_distill_end_to_end_and_idempotent(mem, monkeypatch):
    monkeypatch.setattr(httpx, "post", _fake_llm)
    mem.sessions.append_turn("sess-1", "user", "I prefer dark mode everywhere", task_id="prefs")
    mem.sessions.append_turn("sess-1", "assistant", "Noted.", task_id="prefs")

    report = mem.improve(["sess-1"])
    assert report["atoms"] == 1
    assert report["errors"] == []

    hits = mem.recall("dark mode", task_id="prefs")
    assert len(hits) == 1
    assert hits[0]["provenance"] == "explicit_statement"
    # unscoped query must not see the task-scoped distilled atom
    assert mem.recall("dark mode") == []

    # re-run: content-hash dedup makes it a safe no-op
    report2 = mem.improve(["sess-1"])
    assert report2["atoms"] == 0
    assert report2["skipped_dedup"] == 1


def test_distill_records_llm_failure_without_crashing(mem, monkeypatch):
    def _down(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", _down)
    mem.sessions.append_turn("sess-9", "user", "something memorable", task_id="t")
    report = mem.improve(["sess-9"])
    assert report["atoms"] == 0
    assert len(report["errors"]) == 1
    assert report["errors"][0]["session"] == "sess-9"


# -- in_memory() constructor: the README quickstart path ----------------------


def test_in_memory_constructor_is_zero_dependency():
    mem = Memory.in_memory(embedder=HashEmbedder())
    mem.remember(
        "Playwright installed as an agent-testing dependency",
        category=Category.fact,
        provenance=Provenance.explicit_statement,
        task_id="playwright-test",
    )
    # unscoped recall must not leak the task-scoped atom
    assert all("Playwright" not in h["content"] for h in mem.recall("testing setup"))
    # scoped recall sees it
    assert any(
        "Playwright" in h["content"]
        for h in mem.recall("testing setup", task_id="playwright-test")
    )
