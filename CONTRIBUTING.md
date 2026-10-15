# Contributing to memstrand

Thanks for considering a contribution. memstrand is a small, local-first
project — most changes are welcome, and the bar for a good first PR is low.

## Dev setup

```bash
git clone https://github.com/chiranjeevirajukr/memstrand.git
cd memstrand
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev,all]'
```

`.[dev,all]` pulls in `pytest`/`ruff` plus the optional `kuzu` (graph
backend) and `pdf` (ingestion) extras, so the full test suite can exercise
every backend.

## Running tests and lint

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

Both must pass before a PR is merged. CI runs the same two commands.

## Test philosophy

- **No network.** Every test runs fully offline — no downloading embedding
  models, no calling a real LLM. Tests that need embeddings use a
  deterministic `HashEmbedder` test double (a small keyword-overlap
  vectorizer defined in each test file, e.g.
  `tests/test_memory_flow.py::HashEmbedder`); tests that need an LLM
  monkeypatch `httpx` at the transport layer instead of hitting a real
  endpoint.
- **In-memory stores.** Tests construct `Memory(...)` directly with
  `vector_store=QdrantVectorStore(client=QdrantClient(":memory:"))` and
  SQLite stores pointed at `tmp_path`, so nothing touches `~/.memstrand` or
  persists between runs.
- Prefer adding a focused unit test near the existing tests for that area
  (`tests/test_core_v2.py` for core `Memory` behavior, `tests/test_cli.py`
  for CLI commands, `tests/test_server.py` for REST/MCP, etc.) over one
  giant end-to-end test.

## Project layout

```
src/memstrand/
  schema.py            # MemoryAtom, MemoryEdge, Category, Provenance, RelationType
  config.py             # Settings / get_settings() — single source of truth for env vars
  api.py                # Memory — the core in-process API
  client.py             # RemoteMemory + get_memory() daemon-or-local dispatch
  cli.py                # argparse CLI, one subcommand per `memstrand <command>`
  server.py             # starlette ASGI app: REST /api, MCP /mcp, web UI /, proxy /v1
  mcp_server.py          # MCP tool/prompt/resource definitions
  hooks.py               # Claude Code hook bodies (claude-code, claude-code-context)
  ingest.py              # Markdown/Obsidian/PDF ingestion + --watch
  export.py              # jsonl/markdown export + import
  embeddings.py          # FastEmbedder, OpenAIEmbedder
  ranking.py             # blended_score, absolute_relevance
  pipeline/
    extract.py           # LLM candidate-atom extraction from session turns
    distill.py           # judge pass (add/update/delete/none) + promotion
    consolidate.py        # FSRS update, TTL expiry, edge decay, replay
  stores/
    base.py               # GraphStore / VectorStore / KeywordIndex / SessionStore protocols
    graph_sqlite.py, graph_kuzu.py, keyword_sqlite.py, sqlite.py, vector_qdrant.py
  web/                    # static web UI assets
tests/                    # one file per area, see "Test philosophy" above
examples/                 # quickstart.py (the 3-test rubric) + docs/integrations.md pointer
docs/                     # configuration, architecture, python-api, cli, integrations
```

## Adding a store backend

All four stores are `Protocol`s in `src/memstrand/stores/base.py`
(`GraphStore`, `VectorStore`, `KeywordIndex`, `SessionStore`) — duck-typed,
not abstract base classes, so a new backend just needs to implement the
same method signatures. To add one:

1. Pick the protocol you're implementing (e.g. a new `VectorStore` for a
   different vector DB) and read its docstring in `stores/base.py` — it
   documents the scope/TTL filtering contract every store must honor.
2. Add `stores/<name>_<backend>.py` implementing the protocol.
3. Wire it up behind a `MEMSTRAND_*` env var in `config.py` if it should be
   selectable at runtime (see `MEMSTRAND_GRAPH_BACKEND` for the pattern), or
   accept it as a constructor kwarg on `Memory` (see `embedder=`,
   `vector_store=`, `graph_store=`, `session_store=` in
   `src/memstrand/api.py`).
4. Add a test module that runs the same contract tests against your new
   backend — see `tests/test_graph_stores.py` for the pattern of testing
   one protocol against multiple backends.
5. Document the new option in [docs/configuration.md](docs/configuration.md)
   and, if it needs a new dependency, add it as an optional extra in
   `pyproject.toml` (see the `kuzu`/`pdf` extras) rather than a hard
   dependency.

## PR checklist

- [ ] `.venv/bin/python -m pytest -q` passes
- [ ] `.venv/bin/ruff check .` passes
- [ ] New behavior has a test (see "Test philosophy")
- [ ] Docs updated if you changed a config var, CLI flag, REST/MCP surface,
      or Python API method (`README.md` / `docs/*.md`)
- [ ] `CHANGELOG.md` updated under `[Unreleased]` for anything user-facing

## Good first issues

- Automated migration tooling from 0.1.x's working-directory layout (see
  the "Breaking" section of `CHANGELOG.md`'s `[0.2.0]` entry)
- Multi-user auth beyond a single shared `MEMSTRAND_TOKEN`
- More store backends behind the existing protocols (e.g. another vector
  DB, Postgres-backed graph/session stores)
- Additional ingestion sources (see `src/memstrand/ingest.py`)

If you're not sure whether an idea fits, open a
[discussion](https://github.com/chiranjeevirajukr/memstrand/discussions)
or an issue first.
