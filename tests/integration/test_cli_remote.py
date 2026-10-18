"""Scenario #2: the full CLI command surface against a real daemon over
the wire, via real `memstrand <cmd>` subprocesses -- remember, recall,
list, show, edit, context, stats, export, import into a second fresh
daemon, backup/restore, forget. Every command must exit 0 and print real
content, not just ids, so a regression that returns well-formed-but-empty
responses doesn't slip through silently."""

from __future__ import annotations

import json

import pytest

from .conftest import daemon_client_env, run_cli

pytestmark = pytest.mark.integration


def test_cli_remote_round_trip(start_daemon, stub, tmp_path):
    daemon_a = start_daemon(stub=stub)
    env_a = daemon_client_env(daemon_a)

    # remember
    result = run_cli(["remember", "the office wifi password is hunter2", "--task", "ops"], env_a)
    assert result.returncode == 0, result.stderr
    atom_id = result.stdout.strip()
    assert atom_id.startswith("mem:")

    # a standing instruction, for the `context` check below
    result = run_cli(
        ["remember", "always answer in a formal tone", "--category", "instruction"], env_a
    )
    assert result.returncode == 0, result.stderr

    # recall
    result = run_cli(["recall", "wifi password", "--task", "ops"], env_a)
    assert result.returncode == 0, result.stderr
    assert "hunter2" in result.stdout

    # list
    result = run_cli(["list", "--task", "ops"], env_a)
    assert result.returncode == 0, result.stderr
    assert atom_id in result.stdout
    assert "hunter2" in result.stdout

    # show
    result = run_cli(["show", atom_id], env_a)
    assert result.returncode == 0, result.stderr
    assert "hunter2" in result.stdout
    assert f"id:         {atom_id}" in result.stdout

    # edit (content change supersedes -> a new id)
    result = run_cli(["edit", atom_id, "--content", "the office wifi password is swordfish"], env_a)
    assert result.returncode == 0, result.stderr
    new_id = result.stdout.strip()
    assert new_id.startswith("mem:")
    assert new_id != atom_id

    result = run_cli(["show", new_id], env_a)
    assert result.returncode == 0, result.stderr
    assert "swordfish" in result.stdout

    # context: the standing instruction must show up
    result = run_cli(["context"], env_a)
    assert result.returncode == 0, result.stderr
    assert "formal tone" in result.stdout

    # stats
    result = run_cli(["stats"], env_a)
    assert result.returncode == 0, result.stderr
    assert "atoms:" in result.stdout

    result = run_cli(["stats", "--json"], env_a)
    assert result.returncode == 0, result.stderr
    stats = json.loads(result.stdout)
    assert stats["atoms"] >= 2

    # export
    export_path = tmp_path / "export.jsonl"
    result = run_cli(["export", "-o", str(export_path)], env_a)
    assert result.returncode == 0, result.stderr
    exported = export_path.read_text()
    assert "swordfish" in exported
    assert "formal tone" in exported

    # import into a SECOND, fresh daemon
    daemon_b = start_daemon(stub=stub)
    env_b = daemon_client_env(daemon_b)
    result = run_cli(["import", str(export_path)], env_b)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["atoms"] >= 2

    result = run_cli(["recall", "wifi password", "--task", "ops"], env_b)
    assert result.returncode == 0, result.stderr
    assert "swordfish" in result.stdout

    # backup (daemon A) / restore (into daemon B, merge semantics)
    result = run_cli(["backup"], env_a)
    assert result.returncode == 0, result.stderr
    backup_path = result.stdout.strip()
    assert backup_path, "backup command must print the backup file path"

    result = run_cli(["restore", backup_path, "--yes"], env_b)
    assert result.returncode == 0, result.stderr
    restore_report = json.loads(result.stdout)
    assert isinstance(restore_report.get("atoms"), int)

    # forget
    result = run_cli(["forget", new_id, "--yes"], env_a)
    assert result.returncode == 0, result.stderr
    assert new_id in result.stdout

    result = run_cli(["show", new_id], env_a)
    assert result.returncode == 1
    assert "error: " in result.stderr
    assert new_id in result.stderr
