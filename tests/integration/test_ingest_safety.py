"""Scenario #9: `memstrand ingest` against a vault containing a symlink
that points outside the vault root and a FIFO, run as a real CLI
subprocess against a real daemon. Must complete promptly (never block on
the FIFO), skip both hazards with the exact documented `reason`s, never
read (let alone store/export) the symlink's outside-root target content,
and leave the daemon responsive throughout."""

from __future__ import annotations

import json
import os
import threading
import time

import httpx
import pytest

from .conftest import daemon_client_env, run_cli

pytestmark = pytest.mark.integration


def test_ingest_skips_symlink_outside_root_and_fifo_safely(start_daemon, stub, tmp_path):
    daemon = start_daemon(stub=stub)
    env = daemon_client_env(daemon)

    outside = tmp_path / "outside.md"
    outside.write_text("# top secret\n\nthe nuclear launch codes are 00000000")

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("# a real note\n\nsomething worth remembering about the project")
    (vault / "link.md").symlink_to(outside)
    fifo_path = vault / "a.fifo"
    os.mkfifo(fifo_path)

    # Poll /api/health in the background throughout the ingest subprocess
    # call -- "daemon still responsive during and after" from the daemon's
    # own point of view, not just "the CLI subprocess eventually returned".
    health_failures: list[str] = []
    stop = threading.Event()

    def _poll():
        while not stop.is_set():
            try:
                r = httpx.get(f"{daemon.url}/api/health", timeout=2.0)
                if r.status_code != 200:
                    health_failures.append(f"status {r.status_code}")
            except httpx.HTTPError as e:
                health_failures.append(str(e))
            time.sleep(0.2)

    poll_thread = threading.Thread(target=_poll, daemon=True)
    poll_thread.start()
    try:
        result = run_cli(["ingest", str(vault)], env, timeout=10.0)
    finally:
        stop.set()
        poll_thread.join(timeout=2.0)

    assert result.returncode == 0, result.stdout + result.stderr
    assert health_failures == [], health_failures

    report = json.loads(result.stdout)
    skip_reasons = {s["reason"] for s in report["skipped"]}
    assert "symlink outside root" in skip_reasons
    assert "not a regular file" in skip_reasons
    assert report["files_ingested"] >= 1

    # The outside-root content must never have been read into memory.
    result = run_cli(["recall", "nuclear launch codes"], env)
    assert result.returncode == 0, result.stderr
    assert "00000000" not in result.stdout

    result = run_cli(["export"], env)
    assert result.returncode == 0, result.stderr
    assert "00000000" not in result.stdout

    # The daemon is still responsive after everything.
    r = httpx.get(f"{daemon.url}/api/health", timeout=5.0)
    assert r.status_code == 200
