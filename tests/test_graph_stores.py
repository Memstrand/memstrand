"""GraphStore protocol conformance: every backend (sqlite, kuzu) must
behave identically. Parametrized so a bug in one backend can't hide
behind the other passing.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

import pytest

from memstrand.schema import Category, MemoryAtom, MemoryEdge, Provenance, RelationType
from memstrand.stores.graph_kuzu import KuzuGraphStore
from memstrand.stores.graph_sqlite import SqliteGraphStore


@pytest.fixture(params=["sqlite", "kuzu"])
def store(request, tmp_path):
    if request.param == "sqlite":
        s = SqliteGraphStore(path=str(tmp_path / "sqlite" / "memstrand.db"))
    else:
        pytest.importorskip("kuzu")
        s = KuzuGraphStore(path=str(tmp_path / "kuzu"))
    yield s
    s.close()


def _atom(**kw) -> MemoryAtom:
    defaults = {"category": Category.fact, "content": "content"}
    defaults.update(kw)
    return MemoryAtom(**defaults)


# -- atoms: upsert, full round-trip, batch reads -----------------------------


def test_atom_full_field_roundtrip(store):
    now = datetime.now(UTC)
    atom = MemoryAtom(
        category=Category.decision,
        content="pick postgres",
        provenance=Provenance.validated,
        confidence=0.42,
        session_id="s1",
        task_id="t1",
        host="h1",
        agent="claude-code",
        created_at=now,
        updated_at=now,
        expires_at=now + timedelta(days=1),
        access_count=3,
        fsrs_difficulty=0.6,
        fsrs_stability_days=12.5,
        fsrs_next_review=now + timedelta(days=2),
        source="/abs/path.md",
        source_ref="notes#heading [2]",
        last_accessed_at=now,
    )
    store.add_atom(atom)
    got = store.get_atom(atom.id)
    assert got is not None
    assert got.model_dump() == atom.model_dump()


def test_add_atom_upserts_by_id(store):
    atom = _atom(content="v1", confidence=0.5)
    store.add_atom(atom)
    atom.content = "v2"
    atom.confidence = 0.9
    store.add_atom(atom)
    assert store.count_atoms() == 1
    assert store.get_atom(atom.id).content == "v2"
    assert store.get_atom(atom.id).confidence == pytest.approx(0.9)


def test_get_atom_missing_returns_none(store):
    assert store.get_atom("mem:does-not-exist") is None


def test_get_atoms_batch(store):
    a, b = _atom(content="a"), _atom(content="b")
    store.add_atom(a)
    store.add_atom(b)
    got = store.get_atoms([a.id, b.id, "mem:missing"])
    assert set(got) == {a.id, b.id}
    assert got[a.id].content == "a"


def test_count_atoms(store):
    assert store.count_atoms() == 0
    store.add_atom(_atom())
    store.add_atom(_atom())
    assert store.count_atoms() == 2


# -- list_atoms: filters, None vs "" semantics, ordering, pagination --------


def test_list_atoms_filters_ordering_pagination(store):
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    a1 = MemoryAtom(category=Category.fact, content="a1", task_id="t1", host="h1", created_at=t0)
    a2 = MemoryAtom(
        category=Category.fact, content="a2", task_id="t1", created_at=t0 + timedelta(seconds=1)
    )
    a3 = MemoryAtom(
        category=Category.event, content="a3", created_at=t0 + timedelta(seconds=2)
    )
    for a in (a1, a2, a3):
        store.add_atom(a)

    # no filter: newest created_at first
    assert [x.id for x in store.list_atoms()] == [a3.id, a2.id, a1.id]
    # value filter
    assert [x.id for x in store.list_atoms(task_id="t1")] == [a2.id, a1.id]
    assert [x.id for x in store.list_atoms(host="h1")] == [a1.id]
    assert [x.id for x in store.list_atoms(category=Category.event.value)] == [a3.id]
    # "" means only global/unset values
    assert [x.id for x in store.list_atoms(task_id="")] == [a3.id]
    assert [x.id for x in store.list_atoms(host="")] == [a3.id, a2.id]
    # category is never unset on a real atom
    assert store.list_atoms(category="") == []
    # pagination
    assert [x.id for x in store.list_atoms(limit=1, offset=0)] == [a3.id]
    assert [x.id for x in store.list_atoms(limit=1, offset=1)] == [a2.id]
    assert store.list_atoms(limit=1, offset=3) == []


# -- edges: round-trip, dedupe, missing-endpoint no-op -----------------------


def test_edge_full_field_roundtrip(store):
    a, b = _atom(content="a"), _atom(content="b")
    store.add_atom(a)
    store.add_atom(b)
    created = datetime.now(UTC)
    edge = MemoryEdge(
        source_id=a.id, target_id=b.id, relation=RelationType.supports,
        confidence=0.33, reason="because", weight=2.5, created_at=created,
    )
    store.add_edge(edge)
    edges = store.all_edges()
    assert len(edges) == 1
    assert edges[0].model_dump() == edge.model_dump()


def test_add_edge_noop_when_endpoint_missing(store):
    b = _atom(content="b")
    store.add_atom(b)
    store.add_edge(MemoryEdge(source_id="mem:ghost", target_id=b.id, relation=RelationType.informs))
    assert store.all_edges() == []


def test_add_edge_dedupe_updates_confidence_and_reason_not_weight(store):
    a, b = _atom(content="a"), _atom(content="b")
    store.add_atom(a)
    store.add_atom(b)
    store.add_edge(
        MemoryEdge(
            source_id=a.id, target_id=b.id, relation=RelationType.related_to,
            confidence=0.3, reason="r1", weight=1.0,
        )
    )
    store.add_edge(
        MemoryEdge(
            source_id=a.id, target_id=b.id, relation=RelationType.related_to,
            confidence=0.9, reason="r2", weight=9.0,
        )
    )
    edges = store.all_edges()
    assert len(edges) == 1
    assert edges[0].confidence == pytest.approx(0.9)
    assert edges[0].reason == "r2"
    assert edges[0].weight == pytest.approx(1.0)  # weight is Hebbian state, not re-seeded


# -- neighbors: both directions, depth, dedupe -------------------------------


def test_neighbors_both_directions_no_self(store):
    a, b, c = _atom(content="a"), _atom(content="b"), _atom(content="c")
    for x in (a, b, c):
        store.add_atom(x)
    store.add_edge(MemoryEdge(source_id=a.id, target_id=b.id, relation=RelationType.informs))
    store.add_edge(MemoryEdge(source_id=c.id, target_id=a.id, relation=RelationType.supports))

    hits = store.neighbors(a.id)
    ids = {n.id for n, _ in hits}
    assert ids == {b.id, c.id}
    assert a.id not in ids
    # the edge keeps its true direction regardless of which side we queried from
    by_id = {n.id: e for n, e in hits}
    assert by_id[b.id].source_id == a.id and by_id[b.id].target_id == b.id
    assert by_id[c.id].source_id == c.id and by_id[c.id].target_id == a.id


def test_neighbors_depth_two_no_duplicates(store):
    a, b, c = _atom(content="a"), _atom(content="b"), _atom(content="c")
    for x in (a, b, c):
        store.add_atom(x)
    store.add_edge(MemoryEdge(source_id=a.id, target_id=b.id, relation=RelationType.informs))
    store.add_edge(MemoryEdge(source_id=c.id, target_id=a.id, relation=RelationType.supports))
    store.add_edge(MemoryEdge(source_id=b.id, target_id=c.id, relation=RelationType.extends))

    hits = store.neighbors(b.id, depth=2)
    ids = [n.id for n, _ in hits]
    assert b.id not in ids
    assert len(ids) == len(set(ids))  # no duplicates even though a<->c<->b all connect
    assert set(ids) == {a.id, c.id}


def test_neighbors_relation_filter(store):
    a, b, c = _atom(content="a"), _atom(content="b"), _atom(content="c")
    for x in (a, b, c):
        store.add_atom(x)
    store.add_edge(MemoryEdge(source_id=a.id, target_id=b.id, relation=RelationType.informs))
    store.add_edge(MemoryEdge(source_id=a.id, target_id=c.id, relation=RelationType.contradicts))

    assert [n.id for n, _ in store.neighbors(a.id, relation="contradicts")] == [c.id]
    assert store.contradicting(a.id)[0][0].id == c.id


# -- supersedes / contradicts semantics --------------------------------------


def test_superseded_ids(store):
    old, new, other = _atom(content="old"), _atom(content="new"), _atom(content="other")
    for x in (old, new, other):
        store.add_atom(x)
    store.add_edge(MemoryEdge(source_id=new.id, target_id=old.id, relation=RelationType.supersedes))
    assert store.superseded_ids([old.id, new.id, other.id]) == {old.id}
    assert store.superseded_ids([]) == set()


# -- Hebbian strengthen / decay -----------------------------------------------


def test_strengthen_edge_matches_either_direction(store):
    a, b = _atom(content="a"), _atom(content="b")
    store.add_atom(a)
    store.add_atom(b)
    store.add_edge(MemoryEdge(source_id=a.id, target_id=b.id, relation=RelationType.informs, weight=1.0))
    # call with args reversed from the edge's actual stored direction
    store.strengthen_edge(b.id, a.id, amount=0.1)
    weight = store.all_edges()[0].weight
    assert weight == pytest.approx(1.0 + 0.1 * (1 - 1.0 / 10))


def test_decay_edges_prunes_weak_edges_never_atoms(store):
    a, b = _atom(content="a"), _atom(content="b")
    store.add_atom(a)
    store.add_atom(b)
    store.add_edge(MemoryEdge(source_id=a.id, target_id=b.id, relation=RelationType.informs, weight=1.0))

    pruned = store.decay_edges(factor=0.01, prune_below=0.5)
    assert pruned == 1
    assert store.all_edges() == []
    assert store.count_atoms() == 2
    assert store.get_atom(a.id) is not None
    assert store.get_atom(b.id) is not None


# -- access tracking, FSRS, expiry, source lookup ----------------------------


def test_record_access_increments_and_stamps(store):
    a = _atom()
    store.add_atom(a)
    t1 = datetime.now(UTC)
    store.record_access([a.id], t1)
    t2 = t1 + timedelta(seconds=5)
    store.record_access([a.id], t2)
    got = store.get_atom(a.id)
    assert got.access_count == 2
    assert got.last_accessed_at == t2


def test_update_fsrs(store):
    a = _atom()
    store.add_atom(a)
    next_review = datetime.now(UTC) + timedelta(days=3)
    store.update_fsrs(a.id, difficulty=0.33, stability_days=7.5, next_review=next_review)
    got = store.get_atom(a.id)
    assert got.fsrs_difficulty == pytest.approx(0.33)
    assert got.fsrs_stability_days == pytest.approx(7.5)
    assert got.fsrs_next_review == next_review


def test_update_fsrs_next_review_none(store):
    a = _atom()
    store.add_atom(a)
    store.update_fsrs(a.id, difficulty=0.5, stability_days=30.0, next_review=None)
    assert store.get_atom(a.id).fsrs_next_review is None


def test_expired_ids(store):
    now = datetime.now(UTC)
    expired = _atom(content="exp", expires_at=now - timedelta(days=1))
    active = _atom(content="act", expires_at=now + timedelta(days=1))
    permanent = _atom(content="perm")
    for x in (expired, active, permanent):
        store.add_atom(x)
    assert store.expired_ids(now) == [expired.id]


def test_ids_by_source(store):
    a = _atom(content="a", source="/tmp/x.md")
    b = _atom(content="b", source="/tmp/x.md")
    c = _atom(content="c")
    for x in (a, b, c):
        store.add_atom(x)
    assert set(store.ids_by_source("/tmp/x.md")) == {a.id, b.id}
    assert store.ids_by_source("nope") == []


# -- delete ------------------------------------------------------------------


def test_delete_atom_removes_atom_and_edges(store):
    a, b = _atom(content="a"), _atom(content="b")
    store.add_atom(a)
    store.add_atom(b)
    store.add_edge(MemoryEdge(source_id=a.id, target_id=b.id, relation=RelationType.informs))
    store.delete_atom(a.id)
    assert store.get_atom(a.id) is None
    assert store.all_edges() == []
    assert store.get_atom(b.id) is not None  # only the targeted atom is gone


# -- Kuzu-specific: migrating a v0.1 database --------------------------------


def test_kuzu_migrates_v01_schema(tmp_path):
    """A v0.1 Kuzu DB predates agent/updated_at/access_count/fsrs_*/source/
    source_ref/last_accessed_at and LINK.created_at. Opening it with the
    current KuzuGraphStore must migrate in place (ALTER TABLE ADD) and
    read the old row back with sane defaults, not crash."""
    kuzu = pytest.importorskip("kuzu")
    path = str(tmp_path / "kuzu_legacy")
    db = kuzu.Database(path)
    conn = kuzu.Connection(db)
    # the exact v0.1 schema (git show HEAD:src/memstrand/stores/graph_kuzu.py)
    conn.execute(
        """CREATE NODE TABLE IF NOT EXISTS Atom(
            id STRING, category STRING, content STRING,
            provenance STRING, confidence DOUBLE,
            session_id STRING, task_id STRING, host STRING,
            created_at STRING, expires_at STRING,
            PRIMARY KEY (id))"""
    )
    conn.execute(
        """CREATE REL TABLE IF NOT EXISTS LINK(
            FROM Atom TO Atom,
            relation STRING, confidence DOUBLE,
            reason STRING, weight DOUBLE)"""
    )
    conn.execute(
        """MERGE (a:Atom {id: $id})
           SET a.category=$cat, a.content=$content,
               a.provenance=$prov, a.confidence=$conf,
               a.session_id=$sess, a.task_id=$task, a.host=$host,
               a.created_at=$created, a.expires_at=$expires""",
        {
            "id": "mem:legacy1", "cat": "fact", "content": "legacy row",
            "prov": "inferred", "conf": 0.5,
            "sess": "", "task": "", "host": "",
            "created": "2024-01-01T00:00:00+00:00", "expires": "",
        },
    )
    del conn, db  # release the v0.1 connection before reopening

    store = KuzuGraphStore(path=path)
    try:
        got = store.get_atom("mem:legacy1")
        assert got is not None
        assert got.content == "legacy row"
        assert got.category == Category.fact
        # migrated columns fall back to sane defaults
        assert got.access_count == 0
        assert got.fsrs_difficulty == pytest.approx(0.5)
        assert got.fsrs_stability_days == pytest.approx(30.0)
        assert got.fsrs_next_review is None
        assert got.agent is None
        assert got.source is None
        assert got.last_accessed_at is None
        assert got.updated_at == got.created_at  # no v0.1 updated_at: fall back to created_at

        # the store is fully usable post-migration
        store.record_access(["mem:legacy1"], datetime.now(UTC))
        assert store.get_atom("mem:legacy1").access_count == 1
    finally:
        store.close()


def test_kuzu_requires_optional_extra_error_message(monkeypatch):
    """Importing the module must not require kuzu; instantiating without
    it installed must raise a clear, actionable ImportError."""
    import builtins

    import memstrand.stores.graph_kuzu as graph_kuzu_module

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "kuzu":
            raise ImportError("no module named kuzu")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="memstrand\\[kuzu\\]"):
        graph_kuzu_module.KuzuGraphStore(path="/tmp/unused")


# -- SQLite-specific: filesystem + concurrency -------------------------------


def test_sqlite_creates_parent_directories(tmp_path):
    path = tmp_path / "a" / "b" / "memstrand.db"
    s = SqliteGraphStore(path=str(path))
    try:
        assert path.exists()
    finally:
        s.close()


def test_sqlite_uses_wal_journal_mode(tmp_path):
    s = SqliteGraphStore(path=str(tmp_path / "memstrand.db"))
    try:
        mode = s.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
    finally:
        s.close()


def test_sqlite_concurrent_add_atom_thread_safety(tmp_path):
    """A multi-threaded HTTP daemon will call this store from several
    threads at once: concurrent add_atom must not corrupt state or raise."""
    s = SqliteGraphStore(path=str(tmp_path / "memstrand.db"))
    errors: list[Exception] = []

    def worker(n: int) -> None:
        try:
            for i in range(25):
                s.add_atom(_atom(content=f"thread{n}-{i}"))
        except Exception as e:  # noqa: BLE001 - want to surface any failure
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert s.count_atoms() == 8 * 25
    s.close()
