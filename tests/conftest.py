"""Shared pytest fixtures: make the whole test suite hermetic.

Without this, a developer's (or CI runner's) ambient shell environment
can leak into the test process — an exported `MEMSTRAND_TOKEN`,
`MEMSTRAND_PORT`, or `HTTPS_PROXY` changes behavior that tests assume is
at its default, and `~/.memstrand` / real `$HOME` state can bleed in or
get written to. `MEMSTRAND_TOKEN=junk pytest` upstream exposed this as
~22 failures with no conftest at all.
"""

from __future__ import annotations

import os

import pytest

# Every proxy env var (upper- and lower-case — httpx/urllib3 honor both)
# that could redirect outbound HTTP calls made during tests.
_PROXY_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch, tmp_path):
    """Strip ambient `MEMSTRAND_*`/proxy env vars and sandbox `HOME` /
    `MEMSTRAND_HOME` before every test.

    Uses `monkeypatch` (not raw `os.environ` mutation), so:
      - every change here is automatically undone after the test, and
      - any test that needs a specific value — most already do, e.g.
        `monkeypatch.setenv("MEMSTRAND_TOKEN", "tkn")` — can simply call
        `monkeypatch.setenv(...)`/`monkeypatch.delenv(...)` in its own
        body. That happens *after* this fixture's setup runs, so the
        test's own call always wins for the duration of that test.
    """
    for key in list(os.environ):
        if key.startswith("MEMSTRAND_"):
            monkeypatch.delenv(key, raising=False)
    for key in _PROXY_VARS:
        monkeypatch.delenv(key, raising=False)

    # Sandbox both HOME (so `~/.memstrand` / `Path.expanduser()` can never
    # resolve to the real home directory) and MEMSTRAND_HOME (the
    # explicit override) at a fresh per-test tmp dir.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MEMSTRAND_HOME", str(tmp_path / "memstrand-home"))
