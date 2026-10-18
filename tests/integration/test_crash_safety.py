"""Scenario #10: `kill -9` the daemon after some writes, then start a new
daemon on the SAME home. Must come up cleanly (no stale on-disk Qdrant
lock left behind by the ungraceful shutdown) with the previously-written
data intact. Uses `list`/`stats` rather than `recall` to keep this test
focused purely on the on-disk stores' crash safety, not the embedder."""

from __future__ import annotations

import json

import pytest

from .conftest import daemon_client_env, run_cli

pytestmark = pytest.mark.integration


def test_daemon_recovers_from_sigkill_with_data_intact(start_daemon, stub):
    daemon = start_daemon(stub=stub)
    env = daemon_client_env(daemon)

    result = run_cli(["remember", "the deploy key lives in 1Password"], env)
    assert result.returncode == 0, result.stderr
    atom_id = result.stdout.strip()

    result = run_cli(["remember", "always double-check prod migrations", "--category", "instruction"], env)
    assert result.returncode == 0, result.stderr

    result = run_cli(["stats", "--json"], env)
    assert result.returncode == 0, result.stderr
    before = json.loads(result.stdout)
    assert before["atoms"] >= 2

    daemon.kill9()

    # A fresh daemon on the exact same home must come up fine -- no stale
    # Qdrant on-disk lock surviving the ungraceful shutdown.
    daemon2 = start_daemon(home=daemon.home, stub=stub)
    env2 = daemon_client_env(daemon2)

    result = run_cli(["stats", "--json"], env2)
    assert result.returncode == 0, result.stderr
    after = json.loads(result.stdout)
    assert after["atoms"] == before["atoms"]

    result = run_cli(["show", atom_id], env2)
    assert result.returncode == 0, result.stderr
    assert "1Password" in result.stdout

    result = run_cli(["list"], env2)
    assert result.returncode == 0, result.stderr
    assert "double-check prod migrations" in result.stdout
