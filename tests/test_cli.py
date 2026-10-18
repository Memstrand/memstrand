"""Tests for memstrand.cli and memstrand.client.

CLI tests run `cli.main([...])` against a local, HashEmbedder-backed
`Memory` (same construction as tests/test_core_v2.py) with
`cli.get_memory` monkeypatched so no daemon, no fastembed, no network is
ever touched. RemoteMemory tests run against a fake server via
`httpx.MockTransport` — also no network.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math

import httpx
import pytest
from qdrant_client import QdrantClient

from memstrand import cli, client
from memstrand.api import Memory
from memstrand.stores.graph_sqlite import SqliteGraphStore
from memstrand.stores.sqlite import SqliteSessionStore
from memstrand.stores.vector_qdrant import QdrantVectorStore


class HashEmbedder:
    """Deterministic keyword-overlap embeddings (same test double used
    across the suite) — shared words => similar vectors, no model load."""

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
def mem_factory(tmp_path):
    def _make(subdir="mem"):
        home = tmp_path / subdir
        home.mkdir(exist_ok=True)
        return Memory(
            home=home,
            embedder=HashEmbedder(),
            vector_store=QdrantVectorStore(client=QdrantClient(":memory:")),
            graph_store=SqliteGraphStore(path=str(home / "graph.db")),
            session_store=SqliteSessionStore(path=str(home / "sessions.db")),
        )

    return _make


@pytest.fixture()
def mem(mem_factory):
    return mem_factory()


@pytest.fixture()
def patch_memory(monkeypatch, mem):
    """`cli.py` calls `client.get_memory()` once per command and closes it
    at exit; tests call `main()` repeatedly against the same in-process
    `Memory`, so `.close()` is neutered here and the fixture's own
    `tmp_path` teardown handles cleanup."""
    monkeypatch.setattr(mem, "close", lambda: None)
    monkeypatch.setattr(cli, "get_memory", lambda *a, **k: mem)
    return mem


def _swap_memory(monkeypatch, new_mem):
    monkeypatch.setattr(new_mem, "close", lambda: None)
    monkeypatch.setattr(cli, "get_memory", lambda *a, **k: new_mem)


# -- remember / recall ----------------------------------------------------------------


def test_remember_prints_id(capsys, patch_memory):
    rc = cli.main(["remember", "the office wifi password is hunter2", "--category", "fact"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("mem:")


def test_remember_reads_stdin_when_dash(capsys, patch_memory, monkeypatch):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("piped content for remember"))
    rc = cli.main(["remember", "-"])
    assert rc == 0
    atom_id = capsys.readouterr().out.strip()
    assert atom_id.startswith("mem:")


def test_recall_human_and_json_output(capsys, patch_memory):
    cli.main(["remember", "the office wifi password is hunter2", "--category", "fact"])
    capsys.readouterr()

    rc = cli.main(["recall", "office wifi password"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[fact]" in out
    assert "the office wifi password is hunter2" in out
    assert "id=mem:" in out
    assert "scope=global" in out

    rc = cli.main(["recall", "office wifi password", "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data[0]["content"] == "the office wifi password is hunter2"


def test_recall_no_hits_prints_placeholder(capsys, patch_memory):
    rc = cli.main(["recall", "nothing matches this at all"])
    assert rc == 0
    assert "no memories found" in capsys.readouterr().out


# -- list / show / edit / forget -------------------------------------------------------


def test_list_show_edit_forget_roundtrip(capsys, patch_memory):
    cli.main(["remember", "the deploy key rotates monthly", "--task", "infra"])
    capsys.readouterr()

    rc = cli.main(["list", "--task", "infra"])
    assert rc == 0
    assert "the deploy key rotates monthly" in capsys.readouterr().out

    rc = cli.main(["list", "--task", "infra", "--json"])
    assert rc == 0
    items = json.loads(capsys.readouterr().out)
    atom_id = items[0]["id"]

    rc = cli.main(["show", atom_id])
    assert rc == 0
    assert "the deploy key rotates monthly" in capsys.readouterr().out

    rc = cli.main(["edit", atom_id, "--content", "the deploy key rotates weekly"])
    assert rc == 0
    new_id = capsys.readouterr().out.strip()
    assert new_id != atom_id

    rc = cli.main(["forget", new_id, "--yes"])
    assert rc == 0
    assert "forgot" in capsys.readouterr().out


def test_list_global_flag_excludes_scoped(capsys, patch_memory):
    cli.main(["remember", "a global instruction", "--category", "instruction"])
    capsys.readouterr()
    cli.main(["remember", "a scoped fact", "--task", "work"])
    capsys.readouterr()

    rc = cli.main(["list", "--global", "--json"])
    assert rc == 0
    items = json.loads(capsys.readouterr().out)
    contents = {i["content"] for i in items}
    assert "a global instruction" in contents
    assert "a scoped fact" not in contents


def test_forget_missing_id_exits_1(capsys, patch_memory):
    rc = cli.main(["forget", "mem:does-not-exist", "--yes"])
    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err


# -- export / import / backup / restore -------------------------------------------------


def test_export_import_roundtrip(capsys, patch_memory, mem_factory, tmp_path, monkeypatch):
    cli.main(["remember", "exported fact one", "--task", "t1"])
    capsys.readouterr()
    cli.main(["remember", "exported fact two", "--task", "t1"])
    capsys.readouterr()

    out_file = tmp_path / "out.jsonl"
    rc = cli.main(["export", "--format", "jsonl", "-o", str(out_file)])
    assert rc == 0
    assert out_file.exists()

    target = mem_factory("import-target")
    _swap_memory(monkeypatch, target)

    rc = cli.main(["import", str(out_file)])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["errors"] == []
    assert target.count() == 2


def test_export_markdown_to_stdout(capsys, patch_memory):
    cli.main(["remember", "a markdown-exportable fact"])
    capsys.readouterr()

    rc = cli.main(["export", "--format", "markdown"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "## fact" in out
    assert "a markdown-exportable fact" in out


def test_backup_restore_gz_roundtrip(capsys, patch_memory, mem_factory, tmp_path, monkeypatch):
    cli.main(["remember", "backed up fact", "--task", "t1"])
    capsys.readouterr()

    backup_path = tmp_path / "backup.jsonl.gz"
    rc = cli.main(["backup", "-o", str(backup_path)])
    assert rc == 0
    assert capsys.readouterr().out.strip() == str(backup_path)
    assert backup_path.exists()
    with gzip.open(backup_path, "rt") as f:
        assert '"backed up fact"' in f.read()

    target = mem_factory("restore-target")
    _swap_memory(monkeypatch, target)

    rc = cli.main(["restore", str(backup_path), "--yes"])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["errors"] == []
    assert target.count() == 1


def test_restore_missing_file_errors(capsys, patch_memory, tmp_path):
    rc = cli.main(["restore", str(tmp_path / "nope.jsonl"), "--yes"])
    assert rc == 1
    assert "error:" in capsys.readouterr().err


# -- stats / config ---------------------------------------------------------------------


def test_stats_json_and_human(capsys, patch_memory):
    cli.main(["remember", "a fact for stats"])
    capsys.readouterr()

    rc = cli.main(["stats", "--json"])
    assert rc == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["atoms"] >= 1

    rc = cli.main(["stats"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "atoms:" in out


def test_config_masks_secrets_and_shows_daemon_status(capsys, patch_memory, monkeypatch):
    monkeypatch.setattr(cli, "probe_daemon", lambda *a, **k: "down")
    rc = cli.main(["config"])
    assert rc == 0
    out = capsys.readouterr().out
    daemon_line = next(line for line in out.splitlines() if line.startswith("daemon "))
    assert daemon_line.split()[-1] == "down"
    # the real llm_api_key default ("local") must never appear unmasked
    key_line = next(line for line in out.splitlines() if line.startswith("llm_api_key"))
    assert "local" not in key_line.split(None, 1)[1]


# -- error path: no traceback ------------------------------------------------------------


def test_show_missing_id_exits_1_without_traceback(capsys, patch_memory):
    rc = cli.main(["show", "mem:does-not-exist"])
    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err
    assert "Traceback" not in captured.err


# -- memory_session lifecycle -------------------------------------------------------------


def test_memory_session_closes_memory(monkeypatch):
    class FakeMem:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    fake = FakeMem()
    monkeypatch.setattr(cli, "get_memory", lambda *a, **k: fake)
    with cli._memory_session() as m:
        assert m is fake
    assert fake.closed is True


# ======================================================================================
# RemoteMemory: fake server via httpx.MockTransport
# ======================================================================================


def _remote(handler, token=None):
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport, base_url="http://daemon.local")
    return client.RemoteMemory("http://daemon.local", token=token, client=http_client)


def test_remote_remember_request_shape_and_token_header():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "mem:abc", "content": "hi"})

    rm = _remote(handler, token="secret-token")
    atom = rm.remember("hi", task_id="t1")

    assert atom == {"id": "mem:abc", "content": "hi"}
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/remember"
    assert captured["auth"] == "Bearer secret-token"
    assert captured["body"]["content"] == "hi"
    assert captured["body"]["task_id"] == "t1"
    assert "session_id" not in captured["body"]  # None kwargs are dropped, not sent as null


def test_remote_no_token_sends_no_auth_header():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"atoms": 0})

    rm = _remote(handler)
    rm.stats()
    assert captured["auth"] is None


def test_remote_error_mapping_raises_with_server_message():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "no such atom: mem:nope"})

    rm = _remote(handler)
    with pytest.raises(client.RemoteError, match="no such atom"):
        rm.update("mem:nope", content="x")


def test_remote_unreachable_raises_remote_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    rm = _remote(handler)
    with pytest.raises(client.RemoteError, match="cannot reach"):
        rm.stats()


def test_remote_get_not_found_returns_none():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "no such atom"})

    rm = _remote(handler)
    assert rm.get("mem:nope") is None


def test_remote_forget_not_found_returns_false():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "no such atom"})

    rm = _remote(handler)
    assert rm.forget("mem:nope") is False


def test_remote_forget_found_returns_true():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"forgot": "mem:abc"})

    rm = _remote(handler)
    assert rm.forget("mem:abc") is True


def test_remote_list_task_id_absent_vs_empty_vs_set():
    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(dict(request.url.params))
        return httpx.Response(200, json={"atoms": [], "total": 0})

    rm = _remote(handler)
    rm.list()
    rm.list(task_id="")
    rm.list(task_id="work")

    assert "task_id" not in captured[0]
    assert captured[1]["task_id"] == ""
    assert captured[2]["task_id"] == "work"


def test_remote_export_returns_raw_text():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["format"] == "jsonl"
        return httpx.Response(200, text="line1\nline2\n")

    rm = _remote(handler)
    assert rm.export("jsonl") == "line1\nline2\n"


def test_remote_import_sends_raw_body():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(200, json={"atoms": 1, "errors": []})

    rm = _remote(handler)
    report = rm.import_jsonl("line\n")
    assert report == {"atoms": 1, "errors": []}
    assert captured["body"] == b"line\n"


def test_remote_recall_returns_results_list():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"id": "mem:1", "content": "x"}]})

    rm = _remote(handler)
    hits = rm.recall("x")
    assert hits == [{"id": "mem:1", "content": "x"}]


def test_remote_count_uses_stats():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/stats"
        return httpx.Response(200, json={"atoms": 42})

    rm = _remote(handler)
    assert rm.count() == 42


# -- daemon_running / get_memory ----------------------------------------------------------


def test_daemon_running_true_and_false(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda url, timeout=None, headers=None: httpx.Response(200))
    assert client.daemon_running("http://x") is True

    def _fail(url, timeout=None, headers=None):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", _fail)
    assert client.daemon_running("http://x") is False


def test_get_memory_prefers_remote_and_passes_token(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda url, timeout=None, headers=None: httpx.Response(200))
    monkeypatch.setenv("MEMSTRAND_TOKEN", "tkn")
    rm = client.get_memory()
    assert isinstance(rm, client.RemoteMemory)
    assert rm.token == "tkn"
    rm.close()


def test_get_memory_falls_back_local_without_constructing_real_memory(monkeypatch):
    # Guards against accidentally loading fastembed/qdrant on this path:
    # Memory() is replaced with a sentinel instead of actually constructing one.
    def _fail(url, timeout=None, headers=None):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", _fail)
    sentinel = object()
    # `open_local_memory()` imports `Memory` lazily from `.api` on every
    # call (see client.py), so patching the real `memstrand.api.Memory`
    # -- not a module-level `client.Memory` attribute, which no longer
    # exists -- is what actually takes effect here.
    monkeypatch.setattr("memstrand.api.Memory", lambda **kwargs: sentinel)
    assert client.get_memory() is sentinel


def test_get_memory_raises_daemon_rejected_without_constructing_real_memory(monkeypatch):
    # A "rejected" probe (daemon up, wrong/missing token) must never fall
    # through to a local Memory() -- that would crash on Qdrant's lock.
    monkeypatch.setattr(httpx, "get", lambda url, timeout=None, headers=None: httpx.Response(401))
    monkeypatch.setattr(
        "memstrand.api.Memory",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not construct Memory")),
    )
    with pytest.raises(client.DaemonRejected, match="rejected this client"):
        client.get_memory()


def test_normalize_atom_handles_dict_and_memory_atom(mem):
    atom = mem.remember("normalize me")
    as_dict = client.normalize_atom(atom)
    assert as_dict["id"] == atom.id
    assert as_dict["content"] == "normalize me"
    # already a dict -> passed through unchanged
    assert client.normalize_atom(as_dict) is as_dict
