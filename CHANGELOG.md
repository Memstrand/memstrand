# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- MCP server migrated to the v2 SDK (`mcp>=2,<3`): `FastMCP` is now
  `MCPServer` (`mcp.server.mcpserver`), and `streamable_http_path` moved
  from the constructor to `streamable_http_app()`. Tool/resource/prompt
  surface is unchanged; tests updated for v2's `CallToolResult` return
  shape and generic `ToolError` message.

### Changed

- Docs reorganized: the long reference material (configuration, architecture,
  Python API, CLI/REST/MCP reference, integrations) moved out of
  `README.md` into `docs/*.md`. `examples/second_brain.md` now points to
  [`docs/integrations.md`](docs/integrations.md), which carries the same
  content. No commands, env vars, or behavior changed — this is a docs-only
  reorganization.

## [0.2.1]

### Changed

- Project moved to the `memstrand` GitHub organization: the repository is
  now `github.com/memstrand/memstrand` and the landing page lives at
  `memstrand.github.io`. All project URLs (README, PyPI page, issue
  templates, contributing docs) updated; old URLs redirect.

## [0.2.0] — "second brain"

A rewrite of the storage and service layer: one daemon, zero required Docker,
and a persistent "second brain" shared across Claude Code, Cursor, Open
WebUI, and scripts.

### Added

- **One daemon, four surfaces.** `memstrand serve` runs a single process on
  `127.0.0.1:7777` exposing MCP (streamable HTTP) at `/mcp`, a JSON REST API
  at `/api`, a web UI at `/`, and an OpenAI-compatible memory-injecting proxy
  at `/v1`. Embedded Qdrant/SQLite are single-writer, so the daemon is what
  lets Claude Code, Cursor, Open WebUI, and the CLI all share one memory
  store safely.
- **No Docker required.** Qdrant now runs embedded, on-disk, under
  `$MEMSTRAND_HOME` by default. `MEMSTRAND_QDRANT_URL` still points at a real
  Qdrant server for anyone who wants one.
- **SQLite graph store is the new default** (`MEMSTRAND_GRAPH_BACKEND=sqlite`).
  Kuzu moved behind the optional `memstrand[kuzu]` extra
  (`MEMSTRAND_GRAPH_BACKEND=kuzu`) since upstream Kuzu was archived after its
  Oct 2025 acquisition and only tolerates one writer process — a poor fit for
  a shared daemon.
- **Hybrid retrieval.** Dense (vector) and keyword (SQLite FTS5 / BM25)
  search run in parallel; relevance is an *absolute* 0..1 signal per
  candidate — `max(clipped cosine similarity, keyword strength)` plus a
  small bonus when both retrievers agree, or `sigmoid(cross-encoder score)`
  if `MEMSTRAND_RERANK_MODEL` is set — not normalized within the result
  set, so a lone weak match can't score a perfect 1.0. `MEMSTRAND_MIN_RELEVANCE`
  (or `recall(min_relevance=...)`) drops hits below a threshold. Relevance
  is then blended with confidence, recency, provenance, and access-count
  signals into one final score.
- **Memory dynamics, scheduled.** Per-atom FSRS strength (difficulty,
  stability, next-review) updates on every successful recall; `consolidate()`
  expires TTL'd atoms, decays/prunes weak Hebbian edges, links near-duplicate
  atoms via `supersedes` instead of deleting them, and replays the
  highest-priority due atoms. The daemon runs it on a schedule
  (`MEMSTRAND_CONSOLIDATE_HOURS`) alongside background distillation of
  pending sessions.
- **`log_turn()` + incremental `improve()`.** Raw session turns are logged
  cheaply (no embedding, no extraction); `improve()` distills only the turns
  since the last run, in the background if requested.
- **Claude Code integration.** `memstrand hook print-config` prints a
  `settings.json` snippet wiring `Stop`/`PreCompact`/`UserPromptSubmit` to
  `memstrand hook claude-code` and `memstrand hook claude-code-context` —
  transcript logging plus background distillation, and context injection
  into every prompt.
- **Context injection primitive.** `mem.context()` (and the `/api/context`
  / MCP `context` surfaces) render standing instructions/preferences plus
  query-relevant memories as a ready-to-inject Markdown block.
- **Ingestion.** `memstrand ingest` pulls Markdown/Obsidian vaults, plain
  text, and (with `memstrand[pdf]`) PDFs into memory, with change detection,
  heading-aware chunking, Obsidian `[[wikilink]]` edges, and `--watch` for
  continuous ingestion.
- **Full CLI management surface.** `list` / `show` / `edit` / `forget` /
  `forget-source` / `export` / `import` / `backup` / `restore` / `stats` /
  `config`, on top of `remember` / `recall`.
- **Web UI** at `/` for browsing, searching, and editing memories.
- **OpenAI-compatible embeddings.** `MEMSTRAND_EMBED_BACKEND=openai` points
  embedding calls at any OpenAI-compatible `/embeddings` endpoint (Ollama,
  llama.cpp, LM Studio) instead of the local fastembed default; embedding
  dimension is auto-detected and guarded against mismatches.
- **Append-only provenance ledger.** Every write/edit/forget/expire is
  logged permanently; edits create a new, superseding atom rather than
  mutating history.
- **Security.** Loopback-only by default; a non-loopback `MEMSTRAND_HOST`
  refuses to start without `MEMSTRAND_TOKEN`; Origin/Host-header checks
  guard the mutating surface against CSRF and DNS rebinding (falls back to
  `Referer` when `Origin` is absent, for browsers that omit it on
  same-origin requests). The web UI stores the token in `localStorage`,
  prompts for it on a 401, and has a settings (⚙) button to set/change it.
- **Scoped browse/delete controls.** `Memory.get(atom_id, task_id=, host=,
  scoped=False)` and `GET /api/atoms/{id}?scoped=1&task_id=..` add an
  opt-in scoped view (returns `None`/filters neighbors outside scope) on
  top of the default owner-facing unscoped view. `forget_source()` now
  takes optional `task_id`/`host` for an exact-scope delete (default:
  every scope) — `ingest.py` uses this so re-ingesting a changed file
  under one task never deletes another task's or the global copy of the
  same source path.
- **Per-(task, folder) ingestion.** Ingest change-tracking is now keyed by
  `(task_id, path)` instead of `path` alone, so the same folder can be
  ingested into multiple tasks independently; wikilink backlinks are
  re-linked correctly across partial re-ingests.

### Changed

- Data now lives under `MEMSTRAND_HOME` (default `~/.memstrand`) instead of
  `./kuzu_data` and `./memstrand.db` in the working directory.
- Memory ids are now scope-aware content hashes (`mem:<sha256 of content +
  task_id + host>`) instead of random ids, so the same fact remembered twice
  in the same scope is a no-op instead of a duplicate.
- Scoped `recall()` now also returns global (task/host-less) memories in
  addition to its own scope — previously a scoped query only saw its own
  scope.
- `Memory.update()` on a changed-content edit now always creates a new atom
  linked by a `supersedes` edge, rather than mutating the atom in place.
- Default `MEMSTRAND_LLM_BASE_URL` moved from `http://localhost:8081/v1` to
  `http://localhost:47811/v1`. Port 8081 is commonly used by other local
  services (e.g. SSM tunnels), so memstrand's distillation/consolidation
  could silently POST `/chat/completions` into an unrelated server; 47811 is
  unassigned and below macOS's ephemeral port range.

### Fixed

- TTL (`expires_at`) was never actually enforced; expired atoms could still
  be recalled.
- Content-only atom ids could collide across different task/host scopes.
- Hebbian edge strengthening ignored the incoming-edge direction.
- The Kuzu graph backend returned neighbor edges with source/target reversed
  for incoming edges.
- The provenance ledger overwrote prior history instead of appending to it.
- `recall(categories=[...])` crashed (fixed in 0.1.3, carried forward here).

### Breaking

- **Data location**: old `./kuzu_data`, `./memstrand.db`, and Docker-hosted
  Qdrant data are **not** migrated automatically. There is no `memstrand
  import`-from-0.1 tool — 0.1.x shipped without an export command, so there
  is nothing to migrate *from*. If you have 0.1.x data you care about,
  re-add it with `memstrand remember` / `mem.remember()`, or re-run
  `memstrand ingest` over your source notes under the new `MEMSTRAND_HOME`.
- **Memory ids changed** (scope-aware content hash instead of random uuid) —
  any ids you saved from 0.1.x no longer resolve.
- **Kuzu is no longer a required dependency**; it's an optional extra
  (`memstrand[kuzu]`). SQLite is the default graph backend.
- **Scoped `recall()` now includes globals** — if you relied on a
  task-scoped query *not* seeing global memories, that behavior changed.
- **Default LLM endpoint changed**: `MEMSTRAND_LLM_BASE_URL` now defaults to
  `http://localhost:47811/v1` instead of `http://localhost:8081/v1`. Set
  `MEMSTRAND_LLM_BASE_URL=http://localhost:8081/v1` to keep the old default.

## [0.1.3]

### Fixed

- `recall(categories=[...])` crashed in the Qdrant filter (`MatchValue` ->
  `MatchAny`).

## [0.1.2]

### Changed

- Version bump; added GitHub project URLs to the PyPI page.
- `__version__` is now single-sourced from installed package metadata.

## [0.1.1]

### Fixed

- Republished: the 0.1.0 wheel predated `Memory.in_memory()`.

## [0.1.0]

### Added

- Initial release: local-first typed memory scaffold — schema, store
  interfaces, `Memory.remember()`/`recall()`, `Memory.in_memory()`
  zero-dependency quickstart, and the Playwright scope-leak regression test.
