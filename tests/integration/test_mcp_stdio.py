"""Scenario #6: `python -m memstrand.mcp_server` as a real stdio
subprocess (it has an `if __name__ == "__main__":` guard, unlike cli.py,
so this is the one entry point that can be invoked this way directly).

- With no daemon running and a real stub embedding/LLM backend: initialize
  over stdio, remember/recall round trip.
- With a daemon already up on `MEMSTRAND_PORT`: refuses to start (exits 1
  with a message) rather than opening a second, racing local `Memory` --
  mirrors `tests/test_mcp_wire.py`'s equivalent subprocess tests, reusing
  the shared `start_daemon`/`stub` fixtures instead of a hand-rolled health
  stub.
"""

from __future__ import annotations

import subprocess
import sys

import anyio
import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from .conftest import REPO_ROOT, base_env
from .stub_openai import free_port

pytestmark = pytest.mark.integration


def test_stdio_remember_recall_round_trip_no_daemon(tmp_path, stub):
    home = tmp_path / "home"
    # Pin MEMSTRAND_PORT to a known-free port -- mcp_server.main() probes
    # settings.server_url before opening local stores, and without this
    # it would default to 7777, making this test's "no daemon" assumption
    # depend on nothing else on the machine/CI runner listening there.
    env = base_env(
        home,
        MEMSTRAND_PORT=str(free_port()),
        MEMSTRAND_EMBED_BACKEND="openai",
        MEMSTRAND_EMBED_MODEL="stub-embed",
        MEMSTRAND_EMBED_BASE_URL=stub.url,
        MEMSTRAND_LLM_BASE_URL=stub.url,
        MEMSTRAND_LLM_MODEL="stub-llm",
    )
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "memstrand.mcp_server"], env=env, cwd=str(REPO_ROOT)
    )

    async def _run():
        errlog_path = tmp_path / "stderr.log"
        with errlog_path.open("w") as errlog:
            async with (
                stdio_client(params, errlog=errlog) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                r = await session.call_tool(
                    "remember", {"content": "the spare key is under the mat", "category": "fact"}
                )
                assert r.is_error is False, r.content[0].text if r.content else r
                r = await session.call_tool("recall", {"query": "spare key"})
                assert r.is_error is False
                hits = r.structured_content["result"]
                assert any("under the mat" in h["content"] for h in hits)

    anyio.run(_run)


def test_stdio_refuses_to_start_when_a_daemon_is_already_up(start_daemon, stub, tmp_path):
    daemon = start_daemon(stub=stub)
    env = base_env(
        daemon.home,
        MEMSTRAND_PORT=str(daemon.port),
        MEMSTRAND_HOST="127.0.0.1",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "memstrand.mcp_server"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert proc.stderr.strip() != ""
    assert "stdio" in proc.stderr or "daemon" in proc.stderr.lower()
