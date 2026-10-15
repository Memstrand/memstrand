"""Tests for memstrand.ingest.

The real Memory API is being rewritten concurrently, so these tests run
against a small in-test fake that implements the contract ingest.py codes
against: mem.home, mem.remember(...), mem.forget_source(source),
mem.graph.add_edge/ids_by_source/get_atoms.
"""

from __future__ import annotations

import builtins
import itertools
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from memstrand.ingest import (
    chunk_markdown,
    chunk_text,
    extract_wikilinks,
    ingest_path,
    watch_path,
)
from memstrand.schema import Category, MemoryEdge, Provenance

# -- fake Memory ---------------------------------------------------------------


class FakeGraph:
    def __init__(self):
        self.atoms: dict[str, SimpleNamespace] = {}
        self.edges: list[MemoryEdge] = []

    def get_atoms(self, ids):
        return {i: self.atoms[i] for i in ids if i in self.atoms}

    def ids_by_source(self, source):
        return [aid for aid, atom in self.atoms.items() if atom.source == source]

    def add_edge(self, edge: MemoryEdge) -> None:
        if edge.source_id not in self.atoms or edge.target_id not in self.atoms:
            return
        key = (edge.source_id, edge.target_id, edge.relation)
        if any((e.source_id, e.target_id, e.relation) == key for e in self.edges):
            return
        self.edges.append(edge)


class FakeMemory:
    def __init__(self, home: Path):
        self.home = home
        self.graph = FakeGraph()
        self.remember_calls = 0
        self.forget_calls: list[str] = []
        self._counter = itertools.count()

    def remember(
        self,
        content,
        category=Category.fact,
        provenance=Provenance.explicit_statement,
        confidence=0.8,
        session_id=None,
        task_id=None,
        host=None,
        ttl_days=None,
        source=None,
        source_ref=None,
        agent=None,
    ):
        self.remember_calls += 1
        atom_id = f"mem:{next(self._counter)}"
        atom = SimpleNamespace(
            id=atom_id,
            category=category,
            content=content,
            provenance=provenance,
            confidence=confidence,
            session_id=session_id,
            task_id=task_id,
            host=host,
            source=source,
            source_ref=source_ref,
            agent=agent,
        )
        self.graph.atoms[atom_id] = atom
        return atom

    def forget_source(self, source, task_id=None, host=None) -> int:
        # ingest.py now always passes task_id explicitly (scope-aware
        # deletion, see test_review_fixes.py); this fake doesn't need to
        # filter by scope itself to exercise ingest.py's own logic.
        self.forget_calls.append(source)
        ids = self.graph.ids_by_source(source)
        for i in ids:
            del self.graph.atoms[i]
        return len(ids)


# -- chunk_markdown --------------------------------------------------------------


def test_chunk_markdown_tracks_heading_path():
    text = (
        "# Title\n\n"
        "Intro paragraph that is long enough to not be merged away in this test.\n\n"
        "## Section A\n\n"
        "Section A content that is also long enough to stand on its own as a chunk.\n\n"
        "## Section B\n\n"
        "Section B content that is also long enough to stand on its own as a chunk.\n"
    )
    chunks = chunk_markdown(text)
    paths = [p for p, _ in chunks]
    assert paths[0] == "Title"
    assert "Title > Section A" in paths
    assert "Title > Section B" in paths
    assert all(t.strip() for _, t in chunks)


def test_chunk_markdown_frontmatter_tags_prepended():
    text = (
        "---\n"
        "tags: [alpha, beta]\n"
        "---\n\n"
        "# Note\n\n"
        "Some body text long enough to remain its own chunk without merging here.\n"
    )
    chunks = chunk_markdown(text)
    assert chunks
    assert all(c.startswith("Tags: alpha, beta") for _, c in chunks)
    # frontmatter delimiters must not leak into chunk text
    assert all("---" not in c for _, c in chunks)


def test_chunk_markdown_merges_tiny_sections():
    text = (
        "# Title\n\n"
        "## A\n\nok\n\n"
        "## B\n\nThis is a long enough paragraph to be its own retained chunk for sure.\n"
    )
    chunks = chunk_markdown(text)
    assert not any(t.strip() == "ok" for _, t in chunks)
    assert any("ok" in t for _, t in chunks)


def test_chunk_markdown_splits_long_section_with_overlap():
    para1 = "A" * 700
    para2 = "B" * 700
    para3 = "C" * 700
    text = f"# Title\n\n{para1}\n\n{para2}\n\n{para3}\n"
    chunks = chunk_markdown(text, max_chars=1000, overlap=100)
    assert len(chunks) > 1
    assert all(len(t) <= 1000 + 100 for _, t in chunks)
    # the tail of one chunk should reappear at the head of the next (the overlap)
    tail = chunks[0][1][-100:]
    assert tail in chunks[1][1]


def test_chunk_markdown_never_emits_empty_chunks():
    text = (
        "# Title\n\n\n\n"
        "## Empty\n\n\n\n"
        "## Real\n\nActual real content with enough characters in this paragraph here.\n"
    )
    chunks = chunk_markdown(text)
    assert all(t.strip() for _, t in chunks)
    assert not any(p == "Title > Empty" for p, _ in chunks)


def test_chunk_text_paragraphs_fit_in_one_chunk():
    text = "Para one.\n\nPara two.\n\nPara three."
    chunks = chunk_text(text, max_chars=1000)
    assert chunks == ["Para one.\n\nPara two.\n\nPara three."]


def test_chunk_text_no_empty_chunks_on_blank_input():
    assert chunk_text("\n\n\n") == []


# -- extract_wikilinks -----------------------------------------------------------


def test_extract_wikilinks():
    text = "See [[Note One]] and [[Note Two|alias]] and [[Note Three#Heading]]."
    assert extract_wikilinks(text) == {"Note One", "Note Two", "Note Three"}


def test_extract_wikilinks_none():
    assert extract_wikilinks("no links here") == set()


# -- ingest_path: vault ingestion -------------------------------------------------


def test_ingest_vault_basic(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Note A.md").write_text(
        "# Note A\n\nThis is the first note with a link to [[Note B]] right here.\n"
    )
    (vault / "Note B.md").write_text(
        "# Note B\n\nThis is the second note, referenced from Note A via wikilink.\n"
    )
    mem = FakeMemory(home=tmp_path / "home")

    report = ingest_path(mem, vault)

    assert report["files_seen"] == 2
    assert report["files_ingested"] == 2
    assert report["chunks"] >= 2
    assert report["links"] == 1
    assert mem.remember_calls == report["chunks"]
    assert not report["errors"]


def test_ingest_rerun_is_noop(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\nSome content that is long enough to be a full chunk.\n")
    mem = FakeMemory(home=tmp_path / "home")

    ingest_path(mem, vault)
    calls_before = mem.remember_calls

    report2 = ingest_path(mem, vault)

    assert report2["files_ingested"] == 0
    assert report2["files_unchanged"] == 1
    assert mem.remember_calls == calls_before


def test_ingest_modified_file_is_forgotten_and_reingested(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "a.md"
    note.write_text("# A\n\nOriginal content that is long enough to be a full chunk here.\n")
    mem = FakeMemory(home=tmp_path / "home")
    ingest_path(mem, vault)

    note.write_text("# A\n\nUpdated content that is long enough to be a full chunk here too.\n")
    report = ingest_path(mem, vault)

    assert report["files_ingested"] == 1
    assert str(note.resolve()) in mem.forget_calls


def test_ingest_deleted_file_is_removed(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "a.md"
    note.write_text("# A\n\nContent that is long enough to be a full chunk in this case.\n")
    mem = FakeMemory(home=tmp_path / "home")
    ingest_path(mem, vault)

    abs_note = str(note.resolve())
    note.unlink()
    report = ingest_path(mem, vault)

    assert abs_note in report["removed"]
    assert mem.graph.ids_by_source(abs_note) == []


def test_ingest_skips_hidden_and_excluded_dirs(tmp_path):
    vault = tmp_path / "vault"
    (vault / ".obsidian").mkdir(parents=True)
    (vault / ".obsidian" / "config.md").write_text("# hidden\n\nshould not be ingested here.\n")
    (vault / "node_modules").mkdir()
    (vault / "node_modules" / "x.md").write_text("# hidden2\n\nalso should not be ingested.\n")
    (vault / "visible.md").write_text("# Visible\n\nThis one should be ingested properly here.\n")
    mem = FakeMemory(home=tmp_path / "home")

    report = ingest_path(mem, vault)

    assert report["files_seen"] == 1
    assert report["files_ingested"] == 1


def test_ingest_pdf_skipped_without_pypdf(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "doc.pdf").write_bytes(b"%PDF-1.4 fake content")
    mem = FakeMemory(home=tmp_path / "home")

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("no pypdf")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    report = ingest_path(mem, vault)

    assert report["skipped"] == [
        {"path": str((vault / "doc.pdf").resolve()), "reason": "install memstrand[pdf]"}
    ]
    assert report["files_ingested"] == 0


def test_ingest_bad_file_does_not_abort_run(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "good.md").write_text("# Good\n\nThis file should still be ingested just fine.\n")
    (vault / "bad.md").write_text("# Bad\n\nThis file will fail to read in this simulated test.\n")
    mem = FakeMemory(home=tmp_path / "home")

    original_read_bytes = Path.read_bytes

    def flaky_read_bytes(self):
        if self.name == "bad.md":
            raise OSError("simulated failure")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", flaky_read_bytes)

    report = ingest_path(mem, vault)

    assert report["files_ingested"] == 1
    assert len(report["errors"]) == 1
    assert report["errors"][0]["path"].endswith("bad.md")


def test_ingest_force_reingests_unchanged_file(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\nSome content that is long enough to be a full chunk.\n")
    mem = FakeMemory(home=tmp_path / "home")

    ingest_path(mem, vault)
    report = ingest_path(mem, vault, force=True)

    assert report["files_ingested"] == 1
    assert report["files_unchanged"] == 0


# -- watch_path -------------------------------------------------------------------


def test_watch_path_stops_on_event(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\nSome content long enough for a chunk in this test.\n")
    mem = FakeMemory(home=tmp_path / "home")
    stop_event = threading.Event()
    reports = []

    def on_report(r):
        reports.append(r)

    def run():
        watch_path(mem, vault, interval=0.05, stop_event=stop_event, on_report=on_report)

    t = threading.Thread(target=run)
    t.start()
    time.sleep(0.2)
    stop_event.set()
    t.join(timeout=2)

    assert not t.is_alive()
    assert reports
