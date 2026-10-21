# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0] — unreleased

### Added (backfill)

- `memstrand hook backfill`: distill existing Claude Code transcripts from
  disk — chunked (one extraction call per `--chunk` turns), oldest first,
  skips subagent transcripts and live sessions, resumable, and shares
  progress with the live Stop hook. `--dry-run`, `--project`, `--since`,
  `--limit`.
- A synchronous `Memory.improve()` no longer holds the store-wide lock for
  the whole distillation (minutes of LLM calls) — only the distillation
  lock — so a shared daemon keeps serving recall and context requests.

Scoping v2, phase 1 — see [docs/design/scoping-v2.md](docs/design/scoping-v2.md).

### Changed

- **Memories belong to a set of scopes, not one task.** `MemoryAtom.scopes`
  replaces `task_id` (still accepted everywhere as a one-element set, and
  readable as a legacy property). A query sees memories sharing any of its
  scopes, plus global ones; global is the reserved member `""`.
- **Atom ids no longer include the task** — they hash content, host, and
  source. Remembering the same content in another scope widens the
  existing atom's scope set instead of creating a sibling (distillation
  reports this as `scope_widened`); a widening merge only adds scopes and
  never downgrades the atom's provenance/category/confidence. Identical
  text from two different source files stays two atoms.
- **Repo identity.** The Claude Code hooks' `--task-from cwd` now scopes by
  the repo's normalized git remote URL (`github.com/org/repo`), falling
  back to `local:<hostname>:<path>` — no more bare, collision-prone folder
  names.
- **Scope-aware corrections.** Distillation only `supersedes` an atom
  whose scopes the correction fully covers; otherwise the overlapping
  scopes move to the correction and the two are linked by `contradicts`,
  so a fix learned in one repo can't retire a fact that's still true in
  another (or a global one).
- `forget_source(..., task_id=X)` removes scope X from matching atoms and
  only deletes an atom once its scope set is empty.
- Export format `0.3` (atoms carry `scopes`). Importing a `0.2` export
  re-keys its atoms; importing an atom that already exists unions scopes.

### Added

- **Attribution from tool calls** (scoping v2 phase 2). With the hooks'
  new default `--task-from auto`, each logged turn records the other repos
  it touched (Read/Edit/Write/Grep/Glob paths, `cd` targets and paths in
  Bash, paths in your prompts; git repos only) in a new `turns.scopes`
  column. Distillation offers those repos to the extractor as a candidate
  list and files a fact under the repos it names; names not on the list
  are dropped. `log_turn`/REST `/api/turns`/MCP `log_turn` accept `scopes`.
- **Scope review queue** (scoping v2 phase 3). Memories the extractor marks
  `"ambiguous"`, and claimed-global instructions/preferences the trust
  boundary couldn't confirm from the user's own words, are stored narrowly
  and queued. The `UserPromptSubmit` hook adds a "scope check" block (≤3
  per prompt, once per session, expires after 5 sessions) asking Claude to
  ask the user and record the answer. `memstrand review`, `scope --keep`,
  REST `GET`/`POST /api/reviews`, MCP `scope(keep=True)`. Confirming an
  unverified-global memory upgrades it to `validated` provenance — only on
  a human decision, never from distillation's own scope bookkeeping.
  Queued memory text is sanitized like `context()` output.
- Claude Code hook transcripts are read with `errors="replace"`: one
  non-UTF-8 byte no longer stops a session from being logged.
- **Read-side scopes and agent attribution** (scoping v2 phase 4). The
  context hook adds repos named by path in the prompt to the lookup; the
  `/v1` proxy accepts `x-memstrand-scopes`. Every memory records its writer
  (`claude-code@host`, `cli@host`, `openai-proxy@host`, `mcp@host`; turns
  carry it too and distilled atoms inherit it); `list --agent` /
  `GET /api/atoms?agent=` filter by it, and REST `remember` now honors
  `agent` from the client.
- `memstrand forget-source <source> --task/--repo X` removes just that scope
  (REST: `{"source", "scope"}`).
- `MEMSTRAND_LLM_API_KEY_COMMAND` refreshes never block other callers: they
  get the cached token while one thread refreshes.
- `memstrand scope <id>` (`--set/--add/--remove/--global`), `memstrand
  scopes` and `scopes rename <old> <new>`; `--repo` (key, short name, or a
  path resolved through git) on `remember`/`recall`/`list`/`context`/`answer`.
- REST: `POST /api/atoms/{id}/scopes`, `GET /api/scopes`,
  `POST /api/scopes/rename`, and `scopes` on remember/recall/context/answer.
  MCP: a `scope` tool and `scopes` arguments.
- `memstrand.scopes`: remote-URL normalization (credentials always
  stripped), git-root resolution with a per-repo cache, short-name aliases.

### Migration

- On first open, a pre-0.3 store is re-keyed automatically (task-scoped
  atoms only; global ids are unchanged), merging identical content across
  tasks, re-pointing edges, carrying provenance history forward, and
  reusing stored vectors (no re-embedding). A full backup is written to
  `backups/pre-scoping-v2-<ts>.jsonl` first. Legacy rows that don't move
  keep working as-is.

## [0.2.2] — unreleased

### Added

- **`MEMSTRAND_LLM_API_KEY_COMMAND`.** A shell command whose stdout is the
  LLM bearer key (like Claude Code's `apiKeyHelper`), re-run at most once a
  minute so a long-running daemon keeps working against gateways that mint
  short-lived tokens. Used by distillation, `answer`, the `/v1` proxy, and
  the `openai` embedding backend; falls back to the last good token, then
  `MEMSTRAND_LLM_API_KEY`, if the command fails.

### Fixed

- **Hermetic test suite.** `tests/conftest.py` (previously nonexistent) now
  has an autouse fixture that strips every `MEMSTRAND_*` and proxy
  (`HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY`/`NO_PROXY`, both cases) env var
  and sandboxes `HOME`/`MEMSTRAND_HOME` at a tmp dir before each test.
  Previously `MEMSTRAND_TOKEN=junk pytest` produced 22 failures from
  ambient-environment leakage with no conftest at all to prevent it.
- **`make install` on Python 3.14.** `kuzu==0.11.3` has no cp314 wheels and
  its sdist doesn't build, so `make install` broke outright wherever a bare
  `python3` resolved to 3.14. The Makefile now auto-detects the first
  available `python3.13`/`python3.12`/`python3.11` on `PATH` (`PYTHON ?=
  ...`), prints a clear error if none is found, and still allows `make
  install PYTHON=python3.12` to force a specific interpreter.
- **Dependency floors corrected**: `pydantic>=2.12` and `uvicorn>=0.31.1`
  (both are `mcp>=2`'s own real floors; the previous `pydantic>=2.0`/
  `uvicorn>=0.30` under-declared them), and `fastembed>=0.4` (0.3.x
  predates `fastembed.rerank.cross_encoder`, which `Reranker` imports).
- **CI matrix**: dropped the Python 3.13 "install without `kuzu`" special
  case — kuzu 0.11.3 ships cp313 wheels for macOS (arm64 + x86_64) and
  manylinux (`2_28`, aarch64 + x86_64), so `.[dev,all]` now installs
  cleanly on every matrix cell. The `kuzu` extra now carries a
  `python_version < "3.14"` marker in `pyproject.toml` so `pip install
  memstrand[all]` on 3.14 simply omits kuzu instead of trying (and
  failing) to build its sdist.
- **README/PyPI**: relative image sources (`docs/assets/banner.svg`,
  `docs/assets/demo.svg`) and relative links (`LICENSE`, `docs/*.md`) 404
  on PyPI, which doesn't resolve repo-relative paths. All now point at
  absolute `https://raw.githubusercontent.com/...` (images) or
  `https://github.com/.../blob/master/...` (links) URLs, which also render
  correctly on GitHub.
- `client.daemon_running()` now sends `MEMSTRAND_TOKEN` on its health
  check — previously an unauthenticated probe against a token-protected
  daemon looked identical to "no daemon running", so the CLI/hooks would
  mistakenly fall back to opening the stores directly.
- `chat_completion()` now maps httpx-level transport errors (connection
  refused, timeout, etc.) to a `LLMUnavailable` (transient) error instead
  of letting the raw `httpx` exception propagate — previously a local LLM
  server being down could crash a scheduled `consolidate()`/`improve()`
  pass instead of being retried later.
- `ingest()` now errors on a missing path (previously a silent no-op) and
  skips symlinks that resolve outside the ingestion root and non-regular
  files (FIFOs, device files, sockets) instead of following/reading them.
- `import_jsonl()` now refuses to overwrite an existing atom id with
  different content — re-importing the same export is idempotent, but a
  conflicting atom id is an error rather than a silent clobber.
- The `/v1` streaming proxy now forwards the upstream's error status
  verbatim instead of masking non-2xx responses.
- `recall(limit=0)` (or negative) now returns `[]` instead of erroring or
  silently falling back to a default limit; `remember(ttl_days=0)` now
  expires the atom immediately; `update(atom_id=...)` on a since-superseded
  atom id now resolves to and applies the update on the current version.
- `Memory` now constructs its vector store eagerly, at `Memory.__init__`
  time, instead of lazily on first successful embed. Two consequences,
  both fixed: a locked on-disk Qdrant store (`StoreLocked`) now fails fast
  at daemon/CLI startup instead of surfacing confusingly on first use, and
  `memstrand improve` as the very first store-touching operation on a
  fresh daemon no longer crashes every candidate with `AttributeError:
  'NoneType' object has no attribute 'exists'` (the dedup check in
  `pipeline/distill.py` assumed the vector store already existed).
- `server.serve()` now catches `StoreLocked`/`ConfigError` (and other
  startup failures) around store construction and prints one clean
  `error: ...` line to stderr and exits 1 — no ready banner, no traceback
  — instead of letting a second `memstrand serve` against an
  already-locked `MEMSTRAND_HOME` crash with a raw Qdrant lock string or
  hang. `client.open_local_memory()` translates `StoreLocked` into the
  same friendly "locked by another process" message the CLI already used
  for the Qdrant-lock-string fallback path. The CLI also now prints a
  clean one-line error (and exits 1) for `ConfigError`.
- Fixed two integration tests that were marked `xfail` for the two product
  bugs above (`test_wrong_port_with_live_daemon_gives_clean_lock_error`,
  renamed `test_improve_as_the_very_first_operation_crashes` ->
  `test_improve_as_first_operation_works`) — both now pass for real, and a
  new integration test (`test_second_daemon_on_same_home_fails_fast_with_store_locked_error`)
  covers the second-`serve`-on-the-same-home scenario directly.
- Several integration/unit tests that spawned a real subprocess daemon or
  MCP stdio server without explicitly pinning `MEMSTRAND_PORT` relied on
  the implicit default port `7777` being free, which is flaky in CI/on a
  developer machine with anything else listening there. All such tests
  (`tests/test_mcp_wire.py`, `tests/integration/test_mcp_stdio.py`) now
  request a fresh OS-assigned free port instead.
- `hooks.py`'s `hook.log`/`hook_state.json` and `cli.py`'s `backup`
  destination directory/file are now created through
  `config.ensure_private_dir`/`config.secure_file` (`0700`/`0600`) like
  every other on-disk store under `MEMSTRAND_HOME` — previously these two
  surfaces were missed by the private-files-by-default change below.

### Security

- CLI commands, Claude Code hooks, and stdio `memstrand mcp` now detect a
  running daemon that *rejects* the client's `MEMSTRAND_TOKEN` (missing or
  wrong) and fail fast with a clear "set MEMSTRAND_TOKEN" message, instead
  of silently falling back to opening the stores directly as a second,
  unscoped writer. Hooks log this (and any other failure) to
  `{MEMSTRAND_HOME}/hook.log`.
- Non-loopback (`MEMSTRAND_HOST=0.0.0.0`) + `MEMSTRAND_TOKEN` now works
  end-to-end for MCP, not just REST/the proxy — `/mcp` enforces the same
  bearer-token check on a non-loopback bind.
- Standing instructions (global, task-less promoted memories) now only
  come from trusted provenance (`explicit_statement`, `validated`,
  `corrected`); distillation only promotes **user**-turn content into
  global standing instructions, never assistant turns.
- Everything memstrand writes under `MEMSTRAND_HOME` is now created
  private (directories `0700`, files `0600`).
- The daemon now sends security headers (`X-Frame-Options: DENY`, a
  restrictive `Content-Security-Policy`, etc.) on every response.
- New `MEMSTRAND_ALLOWED_ORIGINS` env var (comma-separated) to trust
  additional browser origins (e.g. the MCP inspector) for the existing
  CSRF Origin check, on top of — not instead of — the loopback/token
  checks.
- **DNS-rebinding hole closed.** The CSRF/same-origin check used to also
  accept an Origin (or Referer-derived origin) whose netloc merely
  *matched the request's own Host header*, even if that origin wasn't in
  `MEMSTRAND_ALLOWED_ORIGINS` — but an attacker who controls DNS (or a
  malicious page the victim's browser resolves) controls both headers,
  so that comparison proved nothing. This same-origin-by-Host fallback is
  now removed entirely: in non-loopback mode, an Origin/Referer is only
  trusted if it's the daemon's own origin or is explicitly listed in
  `MEMSTRAND_ALLOWED_ORIGINS`. Requests with neither header (CLI, MCP
  clients, hooks) are unaffected — the bearer token is still mandatory
  for them. **To use the web UI over LAN** (binding to a non-loopback
  host), add its origin to `MEMSTRAND_ALLOWED_ORIGINS`, e.g.
  `MEMSTRAND_ALLOWED_ORIGINS=http://192.168.1.5:7777`.
- `MEMSTRAND_ALLOWED_ORIGINS` entries are now strictly parsed and
  validated at startup: each entry is normalized (whitespace/trailing `/`
  stripped, scheme+host lowercased) and must be exactly
  `scheme://host[:port]` with scheme `http` or `https` and no
  path/query/fragment. `null`, `*`, bare hosts, and other malformed
  entries are rejected — the daemon refuses to start with a clear error
  naming the bad entry (fail closed) instead of silently ignoring it or
  accepting something unsafe.
- In non-loopback mode, when bound to a specific host/address (not
  `0.0.0.0`/`::`), the `Host` header is now also validated: it must match
  the bind host or the host of an allowed origin, or the request is
  rejected with 403. Wildcard binds (`0.0.0.0`/`::`) have no single "own
  Host" to check against, so this check is skipped for them (the
  Origin/token checks still apply).
- REST `/api/turns` and the MCP `log_turn` tool now validate `role`
  against `user`/`assistant`/`system`/`tool` — REST rejects any other
  value with 400, and `log_turn`'s `role` parameter is typed
  `Literal["user", "assistant", "system", "tool"]` so the MCP SDK itself
  rejects an invalid role before the tool body runs (and the value shows
  up as an enum in the tool's schema). `hooks.py` and the OpenAI-proxy
  turn logging already only ever send `user`/`assistant`.

### Changed

- MCP tool arguments `category`/`provenance` are now validated against the
  `Category`/`Provenance` enums and rejected with an actionable error
  (listing valid values) instead of propagating a raw exception; tool
  errors generally are returned to MCP clients as actionable text.
- The daemon now starts even if the configured embedding backend is
  unreachable; the first operation that actually needs an embedding
  reports a clear `embedding backend unreachable` error instead of
  blocking startup or crashing later.
- `export()` (jsonl) now includes `distill_state`, so a `backup`/`restore`
  round-trip preserves per-session distillation progress instead of
  re-distilling already-processed sessions after a restore.
- CI lint now runs `ruff check .` (previously `ruff check src tests`, which
  skipped `scripts/`). `ruff` is pinned to `>=0.17,<0.18` in the `dev`
  extra so a new ruff minor can't silently fail CI on new default rules.
  The `Makefile`'s `lint` target matches: `ruff check .`.
- Added `src/memstrand/py.typed` (PEP 561 marker) so type checkers treat
  memstrand's inline annotations as authoritative for downstream users;
  confirmed included in the built wheel (`hatchling build --target
  wheel` + `unzip -l`).
- Version bumped to 0.2.2.

## [0.2.1]

### Changed

- Project moved to the `memstrand` GitHub organization: the repository is
  now `github.com/memstrand/memstrand` and the landing page lives at
  `memstrand.github.io`. All project URLs (README, PyPI page, issue
  templates, contributing docs) updated; old URLs redirect.
- MCP server migrated to the v2 SDK (`mcp>=2,<3`): `FastMCP` is now
  `MCPServer` (`mcp.server.mcpserver`), and `streamable_http_path` moved
  from the constructor to `streamable_http_app()`. Tool/resource/prompt
  surface is unchanged.
- Docs reorganized: the long reference material (configuration, architecture,
  Python API, CLI/REST/MCP reference, integrations) moved out of
  `README.md` into `docs/*.md`. `examples/second_brain.md` now points to
  [`docs/integrations.md`](docs/integrations.md), which carries the same
  content. No commands, env vars, or behavior changed — this is a docs-only
  reorganization.

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
