"""Regression tests for the second-brain v0.2 code-review findings.

Each test class/section below maps to one numbered finding from the
review. Where a reviewer repro script existed (`/tmp/test_leak.py`,
`/tmp/test_leak2.py`, `/tmp/test_ingest_scope.py`), it has been turned
into a proper pytest test here.

Same no-network discipline as the rest of the suite: a deterministic
HashEmbedder, in-memory Qdrant, tmp SQLite stores, and every LLM/upstream
call monkeypatched or routed through an `httpx.MockTransport`.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from qdrant_client import QdrantClient
from starlette.testclient import TestClient

from memstrand import cli
from memstrand import client as client_module
from memstrand import ingest as ingest_module
from memstrand.api import Memory
from memstrand.client import RemoteMemory
from memstrand.config import get_settings
from memstrand.pipeline.consolidate import _is_llm_reachable
from memstrand.pipeline.extract import LLMUnavailable, chat_completion
from memstrand.schema import Category, MemoryEdge, RelationType
from memstrand.server import create_app
from memstrand.stores.graph_sqlite import SqliteGraphStore
from memstrand.stores.keyword_sqlite import SqliteKeywordIndex
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
        graph_store=SqliteGraphStore(path=str(tmp_path / "graph.db")),
        session_store=SqliteSessionStore(path=str(tmp_path / "sessions.db")),
        home=tmp_path,
    )


def _settings(tmp_path: Path, **overrides):
    s = replace(get_settings(), home=tmp_path, server_host="127.0.0.1", server_port=7777)
    return replace(s, **overrides) if overrides else s


@contextlib.contextmanager
def _server_client(app):
    base_url = f"http://{app.state.settings.server_host}:{app.state.settings.server_port}"
    with TestClient(app, base_url=base_url) as c:
        yield c


def _make_mem(tmp_path: Path) -> Memory:
    return Memory(
        embedder=HashEmbedder(),
        vector_store=QdrantVectorStore(client=QdrantClient(":memory:")),
        graph_store=SqliteGraphStore(path=str(tmp_path / "graph.db")),
        session_store=SqliteSessionStore(path=str(tmp_path / "sessions.db")),
        home=tmp_path,
    )


# ======================================================================================
# 1. BLOCKER — scope leak via graph-neighbor hydration (recall + get)
# ======================================================================================


def test_recall_neighbors_do_not_leak_task_scoped_atom_to_unscoped_query(mem):
    """From /tmp/test_leak.py: a global atom edge-linked to a task-scoped
    atom must never expose that task-scoped atom's content to an unscoped
    (or differently-scoped) query's neighbor list."""
    g = mem.remember("Global fact about the company roadmap", category=Category.fact)
    t = mem.remember(
        "SECRET internal credential rotation plan alpha",
        category=Category.fact,
        task_id="task-alpha",
    )
    mem.graph.add_edge(
        MemoryEdge(source_id=g.id, target_id=t.id, relation=RelationType.related_to, reason="test")
    )

    results = mem.recall("company roadmap", task_id=None)
    assert any(r["id"] == g.id for r in results)
    leaked = any(t.id == n["id"] for r in results for n in r["neighbors"])
    assert not leaked


def test_recall_neighbors_do_not_bridge_two_tasks_via_shared_global(mem):
    """From /tmp/test_leak2.py: two tasks both linked to the same global
    atom must not let task A see task B's content through the global
    atom's neighbor list."""
    g = mem.remember("Company wide roadmap global fact", category=Category.fact)
    a = mem.remember("Task A secret detail about roadmap", category=Category.fact, task_id="task-A")
    b = mem.remember("Task B secret detail about roadmap", category=Category.fact, task_id="task-B")
    mem.graph.add_edge(
        MemoryEdge(source_id=a.id, target_id=g.id, relation=RelationType.related_to, reason="distill")
    )
    mem.graph.add_edge(
        MemoryEdge(source_id=b.id, target_id=g.id, relation=RelationType.related_to, reason="distill")
    )

    results = mem.recall("roadmap", task_id="task-A")
    leaked_b = any(b.id == n["id"] for r in results for n in r["neighbors"])
    assert not leaked_b
    # task A's own atom is still a legitimate neighbor of the global fact
    seen_ids = {n["id"] for r in results for n in r["neighbors"]} | {r["id"] for r in results}
    assert a.id in seen_ids or any(r["id"] == a.id for r in results)


def test_recall_keeps_superseded_neighbors_visible(mem):
    """Design decision: a superseded neighbor stays visible in recall()'s
    neighbor list — the edge itself ("this was replaced by that") is the
    useful signal, not a new disclosure, since the superseding atom is
    already shown as a top-level hit or neighbor in the same scope."""
    old = mem.remember("the api key is ABC123", category=Category.fact)
    new = mem.update(old.id, content="the api key is XYZ789")
    assert any(
        e.source_id == new.id and e.target_id == old.id and e.relation == RelationType.supersedes
        for e in mem.graph.all_edges()
    )
    results = mem.recall("api key", include_superseded=True)
    neighbor_ids = {n["id"] for r in results for n in r["neighbors"]}
    # the old (superseded) atom remains a visible neighbor of the new one
    assert old.id in neighbor_ids or any(r["id"] == old.id for r in results)


def test_standing_instructions_respect_host_scope(mem):
    """_standing_instructions must run every scope check through _visible,
    including host — a host-scoped instruction must not leak into
    context() for a different host."""
    mem.remember(
        "when on laptop-A always use vim", category=Category.instruction, host="laptop-A"
    )
    ctx_other_host = mem.context(task_id=None, host="laptop-B")
    assert "laptop-A" not in ctx_other_host
    ctx_same_host = mem.context(task_id=None, host="laptop-A")
    assert "laptop-A" in ctx_same_host


# ======================================================================================
# 2. get()/list() scope: owner view by default, scoped=True for untrusted callers
# ======================================================================================


def test_get_scoped_hides_out_of_scope_atom_and_filters_neighbors(mem):
    g = mem.remember("global fact", category=Category.fact)
    a = mem.remember("task A secret", category=Category.fact, task_id="task-A")
    b = mem.remember("task B secret", category=Category.fact, task_id="task-B")
    mem.graph.add_edge(
        MemoryEdge(source_id=g.id, target_id=a.id, relation=RelationType.related_to, reason="t")
    )
    mem.graph.add_edge(
        MemoryEdge(source_id=g.id, target_id=b.id, relation=RelationType.related_to, reason="t")
    )

    # owner view (default, unscoped): sees every neighbor regardless of scope
    owner_view = mem.get(g.id)
    neighbor_ids = {n["id"] for n in owner_view["neighbors"]}
    assert {a.id, b.id} <= neighbor_ids

    # scoped to task-A: the global atom is visible, task-B's neighbor is not
    scoped_view = mem.get(g.id, task_id="task-A", scoped=True)
    scoped_neighbor_ids = {n["id"] for n in scoped_view["neighbors"]}
    assert a.id in scoped_neighbor_ids
    assert b.id not in scoped_neighbor_ids

    # an out-of-scope atom itself: None when scoped, still fetchable unscoped
    assert mem.get(b.id, task_id="task-A", scoped=True) is None
    assert mem.get(b.id, task_id="task-A") is not None


def test_rest_atom_detail_scoped_query_param(tmp_path):
    m = _make_mem(tmp_path)
    app = create_app(m, _settings(tmp_path))
    g = m.remember("global fact", category=Category.fact)
    b = m.remember("task B secret", category=Category.fact, task_id="task-B")
    with _server_client(app) as c:
        # default (owner view): any atom, regardless of scope
        resp = c.get(f"/api/atoms/{b.id}")
        assert resp.status_code == 200

        # scoped=1 + a different task: out-of-scope atom 404s
        resp = c.get(f"/api/atoms/{b.id}", params={"scoped": "1", "task_id": "task-A"})
        assert resp.status_code == 404

        # scoped=1 + the global atom: visible from any task scope
        resp = c.get(f"/api/atoms/{g.id}", params={"scoped": "1", "task_id": "task-A"})
        assert resp.status_code == 200


def test_remote_memory_get_forwards_scope_params():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"id": "mem:x"})

    rm = RemoteMemory(
        "http://daemon.local",
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://daemon.local"),
    )
    rm.get("mem:x", task_id="task-A", host="h1", scoped=True)
    assert "scoped=1" in captured["url"]
    assert "task_id=task-A" in captured["url"]
    assert "host=h1" in captured["url"]


# ======================================================================================
# 3. BLOCKER — ingest.py change-detection state keyed by (task_id, path)
# ======================================================================================


def test_ingest_same_path_different_task_ids_not_silently_skipped(tmp_path):
    """From /tmp/test_ingest_scope.py: re-ingesting the same file path
    under a different task_id must not be treated as 'unchanged'."""
    m = Memory.in_memory(embedder=HashEmbedder())
    note = tmp_path / "note.md"
    note.write_text(
        "# Heading\n\nSome content about the project worth remembering here "
        "for test purposes padding padding padding.\n"
    )

    r1 = ingest_module.ingest_path(m, str(note), task_id="task-A")
    assert r1["files_ingested"] == 1
    assert r1["files_unchanged"] == 0

    r2 = ingest_module.ingest_path(m, str(note), task_id="task-B")
    assert r2["files_ingested"] == 1
    assert r2["files_unchanged"] == 0

    atoms_a = m.list(task_id="task-A")
    atoms_b = m.list(task_id="task-B")
    assert atoms_a and atoms_b
    assert {a["id"] for a in atoms_a}.isdisjoint({b["id"] for b in atoms_b})


def test_ingest_forget_source_is_scope_aware_across_tasks(tmp_path):
    """Re-ingesting a changed file under one task must not wipe another
    task's chunks of the same source path."""
    m = Memory.in_memory(embedder=HashEmbedder())
    note = tmp_path / "note.md"
    note.write_text(
        "# Heading\n\nOriginal content about the project worth remembering here "
        "for test purposes padding padding.\n"
    )
    ingest_module.ingest_path(m, str(note), task_id="task-A")
    ingest_module.ingest_path(m, str(note), task_id="task-B")
    atoms_b_before = m.list(task_id="task-B")
    assert atoms_b_before

    note.write_text(
        "# Heading\n\nUpdated content about the project worth remembering here "
        "for test purposes padding.\n"
    )
    report = ingest_module.ingest_path(m, str(note), task_id="task-A")
    assert report["files_ingested"] == 1

    atoms_b_after = m.list(task_id="task-B")
    assert len(atoms_b_after) == len(atoms_b_before)
    assert {a["id"] for a in atoms_b_after} == {a["id"] for a in atoms_b_before}


# ======================================================================================
# 4. Wikilink edges lost on partial re-ingest
# ======================================================================================


def test_partial_reingest_rebuilds_backlink_from_unscanned_file(tmp_path):
    """When file B links to file A, and only A is later re-ingested
    (e.g. the file watcher reacting to a single changed file), the
    B -> A wikilink edge must survive even though B itself was never
    rescanned in that run."""
    m = Memory.in_memory(embedder=HashEmbedder())
    vault = tmp_path / "vault"
    vault.mkdir()
    a_path = vault / "A.md"
    b_path = vault / "B.md"
    a_path.write_text("# A\n\nThis is note A with enough content to form a real chunk here.\n")
    b_path.write_text("# B\n\nNote B refers back to [[A]] right here in this sentence.\n")

    report = ingest_module.ingest_path(m, vault, task_id="proj")
    assert report["links"] == 1

    b_first = ingest_module._first_chunk_id(m, str(b_path.resolve()))
    a_first_before = ingest_module._first_chunk_id(m, str(a_path.resolve()))
    assert b_first is not None and a_first_before is not None
    assert any(
        e.source_id == b_first and e.target_id == a_first_before for e in m.graph.all_edges()
    )

    # Change A's content and re-ingest ONLY A (a single file, not the vault) —
    # this is the partial re-ingest path (e.g. the file watcher).
    a_path.write_text("# A\n\nUpdated note A content, still long enough to be a chunk here.\n")
    partial_report = ingest_module.ingest_path(m, str(a_path), task_id="proj")
    assert partial_report["files_ingested"] == 1

    a_first_after = ingest_module._first_chunk_id(m, str(a_path.resolve()))
    assert a_first_after is not None
    # the backlink from B must have been rebuilt against A's NEW first chunk
    assert any(
        e.source_id == b_first and e.target_id == a_first_after for e in m.graph.all_edges()
    )


# ======================================================================================
# 5. SQLite busy_timeout pragma on every store
# ======================================================================================


def test_sqlite_stores_set_busy_timeout(tmp_path):
    g = SqliteGraphStore(path=str(tmp_path / "g.db"))
    s = SqliteSessionStore(path=str(tmp_path / "s.db"))
    k = SqliteKeywordIndex(str(tmp_path / "k.db"))
    try:
        for store in (g, s, k):
            value = store.conn.execute("PRAGMA busy_timeout").fetchone()[0]
            assert value == 5000
    finally:
        g.close()
        s.close()
        k.close()


# ======================================================================================
# 6. distill.py: judge target_id must be one of the shown `similar` candidates
# ======================================================================================


class _FakeResp:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def test_distill_ignores_judge_target_outside_similar_candidates(mem, monkeypatch):
    """Prompt-injection defense: even if the judge LLM names an id as
    `target_id`, it must be ignored unless that id was actually among the
    `similar` candidates shown to it — otherwise an adversarial session
    transcript could coax the judge into linking into another scope.

    To actually exercise `_judge()` (not just its `if not existing: add`
    shortcut), there must be a genuine same-scope near-duplicate so
    `similar` is non-empty; the malicious target is a DIFFERENT atom in a
    different scope entirely, which must never end up linked."""
    existing = mem.remember(
        "vpn endpoint configuration details for staging environment",
        category=Category.fact,
        task_id="task-isolated",
    )
    malicious = mem.remember(
        "totally unrelated secret from another scope entirely",
        category=Category.fact,
        task_id="other-task",
    )

    def _fake_llm(url, headers=None, json=None, timeout=None):
        # the `json` kwarg (request body) shadows the `json` module in this
        # scope, hence `__import__("json")` below for encoding responses.
        system = json["messages"][0]["content"]
        if "extract durable memories" in system:
            payload = [
                {
                    "category": "fact",
                    "content": "vpn endpoint configuration details for staging environment review",
                    "provenance": "explicit_statement",
                    "confidence": 0.9,
                    "scope": "task",
                }
            ]
            return _FakeResp(__import__("json").dumps(payload))
        if "resolve one candidate memory" in system:
            # Adversarial: claims to update/supersede an id that was never
            # shown to it as a `similar` candidate (only `existing` was).
            return _FakeResp(
                __import__("json").dumps(
                    {
                        "action": "update",
                        "target_id": malicious.id,
                        "relation": "related_to",
                        "reason": "injected",
                    }
                )
            )
        raise AssertionError(f"unexpected prompt: {system[:80]}")

    monkeypatch.setattr(httpx, "post", _fake_llm)
    mem.log_turn(
        "sx",
        "user",
        "please note the vpn endpoint configuration details for staging environment review",
        task_id="task-isolated",
    )
    report = mem.improve(["sx"])

    assert report["atoms"] == 1
    # the judge's "update" verdict was neutralized (its target_id was never
    # in `similar`), so no supersedes/related_to edge gets created from it
    assert report["edges"] == 0
    assert not any(e.target_id == malicious.id for e in mem.graph.all_edges())
    assert existing.id  # sanity: the in-scope near-duplicate exists and was usable as `similar`


# ======================================================================================
# 7. extract.py chat_completion: transient vs non-transient LLMUnavailable
# ======================================================================================


def _make_post(status: int, payload: dict):
    def _post(url, headers=None, json=None, timeout=None):
        return httpx.Response(status, json=payload, request=httpx.Request("POST", url))

    return _post


def test_chat_completion_5xx_is_transient(monkeypatch):
    monkeypatch.setattr(httpx, "post", _make_post(503, {"error": {"message": "overloaded"}}))
    with pytest.raises(LLMUnavailable) as exc_info:
        chat_completion("http://local:1234/v1", "some-model", [{"role": "user", "content": "hi"}])
    assert exc_info.value.transient is True
    assert "overloaded" in str(exc_info.value)


def test_chat_completion_4xx_is_not_transient_and_carries_real_message(monkeypatch):
    monkeypatch.setattr(
        httpx, "post", _make_post(404, {"error": {"message": "model 'x' not found"}})
    )
    with pytest.raises(LLMUnavailable) as exc_info:
        chat_completion("http://local:1234/v1", "x", [{"role": "user", "content": "hi"}])
    assert exc_info.value.transient is False
    assert "model 'x' not found" in str(exc_info.value)
    assert "404" in str(exc_info.value)


def test_chat_completion_connect_error_is_transient_and_unreachable(monkeypatch):
    def _down(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", _down)
    with pytest.raises(LLMUnavailable) as exc_info:
        chat_completion("http://local:1234/v1", "x", [{"role": "user", "content": "hi"}])
    assert exc_info.value.transient is True
    assert "unreachable" in str(exc_info.value)


def test_grounded_answer_surfaces_real_4xx_message_not_generic_unavailable(mem, monkeypatch):
    mem.remember("the vpn endpoint is vpn.example.com", category=Category.fact)
    monkeypatch.setattr(
        httpx, "post", _make_post(404, {"error": {"message": "model 'bad-model' not found"}})
    )
    answer = mem.grounded_answer("what is the vpn endpoint")
    assert "model 'bad-model' not found" in answer
    assert "unavailable" not in answer.lower()


def test_is_llm_reachable_true_when_endpoint_rejects_request(mem, monkeypatch):
    """A non-transient (4xx) LLMUnavailable means the endpoint answered —
    distillation should still be attempted, not skipped as 'down'."""
    monkeypatch.setattr(httpx, "post", _make_post(400, {"error": {"message": "bad request"}}))
    assert _is_llm_reachable(mem) is True


def test_is_llm_reachable_false_when_endpoint_unreachable(mem, monkeypatch):
    def _down(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", _down)
    assert _is_llm_reachable(mem) is False


# ======================================================================================
# 8. server.py: Referer validated when Origin is absent; no stray CORS headers
# ======================================================================================


def test_referer_checked_when_origin_absent(tmp_path):
    m = _make_mem(tmp_path)
    app = create_app(m, _settings(tmp_path))
    with _server_client(app) as c:
        resp = c.post(
            "/api/remember", json={"content": "x"}, headers={"Referer": "http://evil.example/page"}
        )
        assert resp.status_code == 403

        resp = c.post(
            "/api/remember", json={"content": "y"}, headers={"Referer": "http://127.0.0.1:7777/ui"}
        )
        assert resp.status_code == 200

        # neither Origin nor Referer: still allowed (CLI/non-browser clients)
        resp = c.post("/api/remember", json={"content": "z"})
        assert resp.status_code == 200


def test_no_access_control_allow_origin_header_ever_emitted(tmp_path):
    m = _make_mem(tmp_path)
    app = create_app(m, _settings(tmp_path))
    with _server_client(app) as c:
        resp = c.get("/", headers={"Origin": "http://evil.example"})
        assert "access-control-allow-origin" not in {k.lower() for k in resp.headers}

        resp = c.get("/api/export", headers={"Origin": "http://evil.example"})
        assert "access-control-allow-origin" not in {k.lower() for k in resp.headers}


# ======================================================================================
# 9. web/index.html: bearer-token support
# ======================================================================================


def test_web_ui_has_bearer_token_support():
    html_path = (
        Path(__file__).resolve().parents[1] / "src" / "memstrand" / "web" / "index.html"
    )
    text = html_path.read_text(encoding="utf-8")
    assert "memstrand.token" in text
    assert "localStorage" in text
    assert "Authorization" in text
    assert "Bearer" in text
    assert "401" in text


# ======================================================================================
# 10. cli.py: `memstrand serve --no-scheduler` must actually disable the scheduler
# ======================================================================================


def test_cli_serve_no_scheduler_passes_scheduler_false(monkeypatch):
    captured = {}

    def _fake_serve(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(client_module, "daemon_running", lambda *a, **k: False)
    monkeypatch.setattr("memstrand.server.serve", _fake_serve)
    rc = cli.main(["serve", "--no-scheduler"])
    assert rc == 0
    assert captured["scheduler"] is False


def test_cli_serve_without_flag_passes_scheduler_true(monkeypatch):
    captured = {}

    def _fake_serve(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(client_module, "daemon_running", lambda *a, **k: False)
    monkeypatch.setattr("memstrand.server.serve", _fake_serve)
    rc = cli.main(["serve"])
    assert rc == 0
    assert captured["scheduler"] is True


# ======================================================================================
# 9. functional-review findings (2026-10-16): token-aware daemon probe;
#    consolidate() never raises on httpx-level failures
# ======================================================================================


def _raise_invalid_url(*a, **k):
    raise httpx.InvalidURL("Invalid port: ':1]'")


def test_chat_completion_invalid_url_is_transient_llm_unavailable(monkeypatch):
    """httpx errors raised before any response exists (InvalidURL from a
    malformed proxy env, ProxyError, ...) must surface as transient
    LLMUnavailable — not escape raw and crash consolidate()."""
    monkeypatch.setattr(httpx, "post", _raise_invalid_url)
    with pytest.raises(LLMUnavailable) as exc_info:
        chat_completion("http://local:1234/v1", "x", [{"role": "user", "content": "hi"}])
    assert exc_info.value.transient is True


def test_is_llm_reachable_false_when_probe_blows_up_before_request(mem, monkeypatch):
    monkeypatch.setattr(httpx, "post", _raise_invalid_url)
    assert _is_llm_reachable(mem) is False


def test_consolidate_run_never_raises_when_llm_probe_blows_up(mem, monkeypatch):
    """consolidate() documents "Never raises" — an httpx-level failure in
    the LLM reachability ping must degrade to a skipped distillation,
    not abort expire/decay/dedup/replay."""
    monkeypatch.setattr(httpx, "post", _raise_invalid_url)
    report = mem.consolidate()
    assert report["distilled"] == {"skipped": "local LLM unreachable"}


def test_daemon_running_sends_configured_token(monkeypatch):
    """A token-protected daemon 401s unauthenticated /api/health probes.
    daemon_running() must send MEMSTRAND_TOKEN, or get_memory() mistakes
    "protected" for "down" and falls back to a local Memory that crashes
    on the embedded Qdrant on-disk lock while the daemon is healthy."""
    seen = {}

    def _get(url, timeout=None, headers=None):
        seen["headers"] = headers
        assert url.endswith("/api/health")
        if (headers or {}).get("Authorization") == "Bearer s3cr3t":
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(401, json={"error": "missing or invalid bearer token"})

    monkeypatch.setattr(httpx, "get", _get)
    monkeypatch.setenv("MEMSTRAND_TOKEN", "s3cr3t")
    assert client_module.daemon_running("http://127.0.0.1:7777") is True
    assert seen["headers"] == {"Authorization": "Bearer s3cr3t"}


def test_daemon_running_without_token_sends_no_auth_header(monkeypatch):
    seen = {}

    def _get(url, timeout=None, headers=None):
        seen["headers"] = headers
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(httpx, "get", _get)
    monkeypatch.delenv("MEMSTRAND_TOKEN", raising=False)
    assert client_module.daemon_running("http://127.0.0.1:7777") is True
    assert seen["headers"] is None
