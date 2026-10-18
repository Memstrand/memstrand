"""Round-3 ingest.py hardening tests.

Turns the reviewer's filesystem-trust PoCs into pytest tests:

- `/tmp/review_sec2/poc_hardlink.py`, `poc_hardlink2.py` -- a hardlink
  inside the ingest root to a file outside it bypasses the symlink-only
  containment check (it's `S_ISREG`, no symlink involved at all).
- `/tmp/review_sec2/poc_toctou.py`, `poc_toctou2.py` -- validation
  (lstat/`S_ISREG`/containment) and the later read are separate path
  lookups; a path swapped between them (regular file <-> symlink to a
  victim file) can win the race.
- `/tmp/review_core/ingest_test_b2.py` -- a file ingested under a wide
  root, re-ingested (on change) under a narrower root so its stored
  `root` field changes, then deleted: a rescan via the original wide root
  never removed it because orphan-detection matched on the stored `root`
  string rather than path containment.

Also covers: `watch_path` surviving the watched root disappearing
mid-run, and binary garbage with a `.md`/`.txt` extension being skipped
instead of ingested as replacement-character soup.

Uses the same minimal `FakeMemory`/`FakeGraph` contract as test_ingest.py
(mem.home, mem.remember, mem.forget_source, mem.graph.*) rather than the
real `Memory` class -- these tests are about `ingest.py`'s own filesystem
handling, independent of the storage backend, and a fake keeps the TOCTOU
stress-test loop (hundreds of rounds) fast and deterministic.
"""

from __future__ import annotations

import itertools
import os
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from memstrand import ingest as ingest_module
from memstrand.ingest import ingest_path, watch_path
from memstrand.schema import Category, MemoryEdge, Provenance

# -- fake Memory (same contract as test_ingest.py) ----------------------------


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
        self.forget_calls.append(source)
        ids = self.graph.ids_by_source(source)
        for i in ids:
            del self.graph.atoms[i]
        return len(ids)

    def _graph_add_edge(self, edge: MemoryEdge) -> None:
        self.graph.add_edge(edge)

    def all_contents(self) -> list[str]:
        return [a.content for a in self.graph.atoms.values()]


def _supports_hardlinks(tmp_path: Path) -> bool:
    a = tmp_path / "_hardlink_probe_a"
    b = tmp_path / "_hardlink_probe_b"
    a.write_text("x")
    try:
        os.link(a, b)
    except OSError:
        return False
    return True


# ======================================================================================
# 1. HIGH: hardlink bypass of the symlink-only containment check
# ======================================================================================


def test_ingest_skips_hardlink_to_file_outside_root(tmp_path):
    if not _supports_hardlinks(tmp_path):
        pytest.skip("filesystem under tmp_path does not support hardlinks")

    secret = tmp_path / "secret.md"
    secret.write_text("# Creds\n\naws_secret_access_key = AKIAFAKEEXAMPLE1234\n")

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "good.md").write_text("# Good\n\nA perfectly ordinary note with real content here.\n")
    hardlinked = vault / "note.md"
    os.link(secret, hardlinked)

    mem = FakeMemory(home=tmp_path / "home")
    report = ingest_path(mem, vault)

    assert any(
        s["reason"] == "hardlinked file (st_nlink>1)" for s in report["skipped"]
    ), report["skipped"]
    # the hardlinked file never got ingested; the ordinary one still did
    assert report["files_ingested"] == 1
    assert not any("AKIAFAKEEXAMPLE1234" in c for c in mem.all_contents())


def test_ingest_hardlink_skip_survives_recall_style_lookup(tmp_path):
    """From poc_hardlink2.py: confirm the secret never shows up via the
    normal content-lookup path either (not just the ingest report)."""
    if not _supports_hardlinks(tmp_path):
        pytest.skip("filesystem under tmp_path does not support hardlinks")

    secret_dir = tmp_path / "outside"
    secret_dir.mkdir()
    secret = secret_dir / "creds.txt"
    secret.write_text("AKIAFAKE secret credential material that must never be exposed.\n")

    vault = tmp_path / "vault2"
    vault.mkdir()
    os.link(secret, vault / "note.txt")

    mem = FakeMemory(home=tmp_path / "home2")
    ingest_path(mem, vault)

    assert mem.remember_calls == 0
    assert mem.graph.atoms == {}


def test_ingest_plain_regular_file_nlink_one_still_ingested(tmp_path):
    """Regression: an ordinary single-link file must still ingest fine --
    the hardlink check must not over-trigger on every regular file."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\nOrdinary content long enough to form a real chunk here.\n")
    mem = FakeMemory(home=tmp_path / "home")

    report = ingest_path(mem, vault)

    assert report["files_ingested"] == 1
    assert not report["skipped"]


# ======================================================================================
# 2. HIGH: TOCTOU -- validation and the later read are separate path lookups
# ======================================================================================


def test_ingest_toctou_symlink_swap_never_leaks_secret(tmp_path):
    """From poc_toctou.py / poc_toctou2.py: a background thread repeatedly
    swaps the target file between plain regular content and a symlink
    pointing at a secret file outside the ingest root, racing concurrent
    `ingest_path(..., force=True)` calls. Across many rounds, the secret
    must never make it into a `remember()` call."""
    secret_dir = tmp_path / "secret_target"
    secret_dir.mkdir()
    secret = secret_dir / "creds.txt"
    secret.write_text("MEMSTRAND-TEST-SECRET-MARKER-7f3a top secret credential content padding padding.\n" * 5)

    vault = tmp_path / "vault_toctou"
    vault.mkdir()
    target_path = vault / "race.md"
    target_path.write_text("legit content\n" * 100)

    mem = FakeMemory(home=tmp_path / "home")

    stop = threading.Event()
    tmp_link = str(target_path) + ".tmplink"
    tmp_reg = str(target_path) + ".tmpreg"

    def swapper():
        while not stop.is_set():
            try:
                if os.path.lexists(tmp_link):
                    os.remove(tmp_link)
                os.symlink(secret, tmp_link)
                os.replace(tmp_link, target_path)  # atomic rename -> now a symlink
                with open(tmp_reg, "w") as f:
                    f.write("legit content\n" * 100)
                os.replace(tmp_reg, target_path)  # swap back to a (new) regular file
            except OSError:
                pass

    t = threading.Thread(target=swapper, daemon=True)
    t.start()

    rounds = 250
    skip_reasons: dict[str, int] = {}
    try:
        for _ in range(rounds):
            report = ingest_path(mem, vault, force=True)
            for s in report["skipped"]:
                skip_reasons[s["reason"]] = skip_reasons.get(s["reason"], 0) + 1
    finally:
        stop.set()
        t.join(timeout=2)

    assert not any("MEMSTRAND-TEST-SECRET-MARKER-7f3a" in c for c in mem.all_contents()), (
        f"secret leaked via TOCTOU race; skip reasons seen: {skip_reasons}"
    )


# ======================================================================================
# 3. Orphan detection must match by path containment, not stored `root` string
# ======================================================================================


def test_ingest_orphan_removed_on_wide_rescan_after_narrow_reingest(tmp_path):
    """From ingest_test_b2.py: ingest a file under a wide root, change it
    and re-ingest under a narrower root (its stored `root` field is
    rewritten to the narrow one), delete the file, then rescan via the
    ORIGINAL wide root -- the file must be recognized as gone and forgotten,
    not left as a permanent orphan because the stored `root` no longer
    string-matches the wide root."""
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    f = root / "sub" / "x.md"
    f.write_text("v1 content about otters, long enough to be a real chunk here for sure.\n")

    mem = FakeMemory(home=tmp_path / "home")

    ingest_path(mem, root)  # wide root: establishes stored root = wide
    f.write_text("v2 content about otters, changed, long enough to be a real chunk too.\n")
    ingest_path(mem, root / "sub")  # narrow root, file changed -> re-ingested, root -> narrow

    assert any("otters" in c for c in mem.all_contents())

    abs_f = str(f.resolve())
    os.remove(f)
    rep_wide = ingest_path(mem, root)  # rescan via the ORIGINAL (wide) root

    assert abs_f in rep_wide["removed"]
    assert not any("otters" in c for c in mem.all_contents())
    assert mem.graph.ids_by_source(abs_f) == []


def test_ingest_orphan_removed_when_reclassified_unsafe(tmp_path):
    """Extension of the same fix: a previously-ingested file that is now
    unsafe (replaced by a hardlink to something else) must also be
    forgotten on rescan, even though a file still physically exists at
    that name (so a naive `os.path.exists` check alone would keep it)."""
    if not _supports_hardlinks(tmp_path):
        pytest.skip("filesystem under tmp_path does not support hardlinks")

    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "note.md"
    note.write_text("# Note\n\nOriginal legitimate content long enough to be a chunk here.\n")
    mem = FakeMemory(home=tmp_path / "home")

    ingest_path(mem, vault)
    assert any("Original legitimate" in c for c in mem.all_contents())

    other = tmp_path / "other.md"
    other.write_text("unrelated content living outside the vault entirely for this test.\n")
    note.unlink()
    os.link(other, note)

    report = ingest_path(mem, vault)

    assert any(s["reason"] == "hardlinked file (st_nlink>1)" for s in report["skipped"])
    assert not any("Original legitimate" in c for c in mem.all_contents())


# ======================================================================================
# 4. watch_path must survive the watched root disappearing mid-run
# ======================================================================================


def test_watch_path_survives_missing_root_and_keeps_polling(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\nSome content long enough for a chunk in this test case.\n")
    mem = FakeMemory(home=tmp_path / "home")

    stop_event = threading.Event()
    reports: list[dict] = []

    def on_report(r):
        reports.append(r)

    def run():
        watch_path(mem, vault, interval=0.05, stop_event=stop_event, on_report=on_report)

    t = threading.Thread(target=run)
    t.start()
    try:
        time.sleep(0.2)  # let the first successful ingest happen
        shutil.rmtree(vault)
        time.sleep(0.3)  # let several poll cycles hit the now-missing root
        assert t.is_alive(), "watch_path must not crash/exit when the root disappears"
    finally:
        stop_event.set()
        t.join(timeout=2)

    assert not t.is_alive()
    error_reports = [r for r in reports if "error" in r]
    assert error_reports, f"expected at least one error report, got: {reports}"
    assert all(str(vault) == r["root"] for r in error_reports)


# ======================================================================================
# 5. Binary garbage with a .md/.txt extension must be skipped, not mangled
# ======================================================================================


def test_ingest_skips_binary_and_nul_content_but_keeps_utf8_bom(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "binary.md").write_bytes(b"\xff\xfe\x00\x01garbage binary content here" * 10)
    (vault / "nul.md").write_bytes(b"some text\x00with a nul byte in the middle of it padding.")
    (vault / "good.md").write_text("# Good\n\nThis is a perfectly fine utf-8 markdown file here.\n")
    (vault / "bom.md").write_bytes(
        "﻿# BOM\n\nThis file starts with a UTF-8 BOM and should still ingest fine.\n".encode()
    )

    mem = FakeMemory(home=tmp_path / "home")
    report = ingest_path(mem, vault)

    reasons = {s["path"]: s["reason"] for s in report["skipped"]}
    assert reasons[str((vault / "binary.md").resolve())] == "binary or non-UTF-8 content"
    assert reasons[str((vault / "nul.md").resolve())] == "binary or non-UTF-8 content"
    assert report["files_ingested"] == 2  # good.md + bom.md
    assert not any("﻿" in c for c in mem.all_contents())
    assert any("This file starts with a UTF-8 BOM" in c for c in mem.all_contents())


# ======================================================================================
# 6. Existing behavior must stay green under the new fd-based open path
# ======================================================================================


def test_ingest_fifo_still_skipped_not_read(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    fifo_path = vault / "pipe.md"
    try:
        os.mkfifo(fifo_path)
    except (AttributeError, OSError):
        pytest.skip("platform does not support os.mkfifo")

    mem = FakeMemory(home=tmp_path / "home")
    report = ingest_path(mem, vault)

    assert any(s["reason"] == "not a regular file" for s in report["skipped"])
    assert report["files_ingested"] == 0


def test_ingest_symlink_inside_root_still_uses_real_path_as_source(tmp_path):
    vault = tmp_path / "vault"
    sub = vault / "sub"
    sub.mkdir(parents=True)
    real = sub / "real.md"
    real.write_text("# Doc\n\nalpha beta gamma content for a symlink-inside-root test.\n")
    (vault / "alias.md").symlink_to(real)

    mem = FakeMemory(home=tmp_path / "home")
    report = ingest_path(mem, vault)

    assert not any(s.get("reason") == "symlink outside root" for s in report["skipped"])
    sources = {a.source for a in mem.graph.atoms.values() if a.source}
    assert str(real.resolve()) in sources


def test_ingest_size_cap_still_enforced_via_fd_read(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    big = vault / "big.md"
    big.write_text("# Big\n\n" + ("x" * (ingest_module.MAX_FILE_BYTES + 100)))

    mem = FakeMemory(home=tmp_path / "home")
    report = ingest_path(mem, vault)

    assert any(s["reason"] == "file too large (> 5MB)" for s in report["skipped"])
    assert report["files_ingested"] == 0
