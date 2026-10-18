"""Real-process integration tests.

Unlike the rest of `tests/`, these spawn real subprocesses (`memstrand
serve`, the CLI, `python -m memstrand.mcp_server`), bind real sockets (a
stub OpenAI-compatible server), and drive them over the actual wire
protocols (HTTP, stdio, MCP). They exist to catch the class of bug that
only shows up at a real process boundary -- a daemon that only answers
correctly when probed in-process, a CLI subprocess invocation that
silently does nothing, a streaming proxy that swallows an upstream error
-- which the mocked-edge unit tests elsewhere in `tests/` cannot see.

Collection is opt-in: `tests/integration/conftest.py` deselects every
test under here unless `tests/integration` is explicitly named on the
pytest command line, so a bare `pytest -q` (the fast unit-test run) stays
fast. CI's dedicated `integration` job runs `pytest -q tests/integration`
explicitly and picks these up.
"""
