"""Scenario #3: `MEMSTRAND_TOKEN` enforcement over the wire, via a real
CLI subprocess talking to a real token-protected daemon.

- No token at all -> the CLI's `probe_daemon()` sees the daemon is "up but
  rejecting" (`client.py`'s `probe_daemon` tri-state) and must raise a
  clean, actionable error mentioning `MEMSTRAND_TOKEN` -- never fall back
  to a local `Memory()`, which would otherwise crash on Qdrant's raw
  "already accessed by another instance" on-disk lock error. That raw
  string must never reach the user.
- The right token -> everything just works.
- A wrong `MEMSTRAND_PORT` while the daemon is up, same MEMSTRAND_HOME ->
  the CLI can't reach anything on that port, falls back to a *local*
  `Memory()` on the SAME (already locked) home, and must surface the
  wrapped, friendly "locked by another process" message (`client.py`'s
  `_wrap_qdrant_lock_error`) with no raw Qdrant string and no traceback.
"""

from __future__ import annotations

import pytest

from .conftest import daemon_client_env, run_cli
from .stub_openai import free_port

pytestmark = pytest.mark.integration


def test_missing_token_is_rejected_cleanly(start_daemon, stub):
    daemon = start_daemon(stub=stub, token="s3cret-token")

    env = daemon_client_env(daemon, token=None)
    result = run_cli(["remember", "should not be stored"], env)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "error: " in result.stderr
    assert "MEMSTRAND_TOKEN" in result.stderr
    assert "already accessed" not in result.stderr
    assert "Traceback" not in result.stderr


def test_correct_token_works(start_daemon, stub):
    daemon = start_daemon(stub=stub, token="s3cret-token")

    env = daemon_client_env(daemon, token=True)
    result = run_cli(["remember", "the vault combination is 12-34-56"], env)
    assert result.returncode == 0, result.stdout + result.stderr
    atom_id = result.stdout.strip()
    assert atom_id.startswith("mem:")

    result = run_cli(["recall", "vault combination"], env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "12-34-56" in result.stdout


def test_wrong_token_is_rejected_cleanly(start_daemon, stub):
    daemon = start_daemon(stub=stub, token="s3cret-token")

    env = daemon_client_env(daemon, token="definitely-the-wrong-token")
    result = run_cli(["remember", "should not be stored either"], env)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "error: " in result.stderr
    assert "MEMSTRAND_TOKEN" in result.stderr
    assert "already accessed" not in result.stderr
    assert "Traceback" not in result.stderr


def test_wrong_port_with_live_daemon_gives_clean_lock_error(start_daemon, stub):
    daemon = start_daemon(stub=stub, token="s3cret-token")

    # Remember something first so there's an on-disk store to collide with.
    env = daemon_client_env(daemon, token=True)
    result = run_cli(["remember", "something durable"], env)
    assert result.returncode == 0, result.stderr

    wrong_port = free_port()
    assert wrong_port != daemon.port
    # Same MEMSTRAND_HOME as the live daemon, but a port nothing listens on
    # -- the CLI falls back to a local Memory() on that same, already-locked
    # home and must get the wrapped Qdrant-lock error, not a crash. The
    # fallback Memory() embeds BEFORE it ever touches the (locked) vector
    # store (lazy vector-store construction, see api.py's _ensure_vectors),
    # so it needs a reachable embedding backend too -- point it at the real
    # stub, otherwise it would hang retrying a fastembed model download.
    bad_env = daemon_client_env(
        daemon,
        port=wrong_port,
        token=True,
        MEMSTRAND_EMBED_BACKEND="openai",
        MEMSTRAND_EMBED_BASE_URL=stub.url,
        MEMSTRAND_EMBED_MODEL="stub-embed",
    )
    result = run_cli(["remember", "this should not succeed"], bad_env, timeout=15.0)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "error: " in result.stderr
    assert "locked by another process" in result.stderr
    assert "already accessed" not in result.stderr
    assert "Traceback" not in result.stderr
