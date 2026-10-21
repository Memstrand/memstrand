# Configuration

Everything is an environment variable, read once per process via `get_settings()`
(`src/memstrand/config.py`). Run `memstrand config` any time to see the
effective values for the current environment.

| Variable | Default | Purpose |
|---|---|---|
| `MEMSTRAND_HOME` | `~/.memstrand` | data directory — SQLite db, embedded Qdrant, ingest/hook state, backups |
| `MEMSTRAND_QDRANT_URL` | *(unset — embedded on-disk Qdrant under `$MEMSTRAND_HOME/qdrant`)* | point at a real Qdrant server instead |
| `MEMSTRAND_GRAPH_BACKEND` | `sqlite` | `sqlite` (default) or `kuzu` (needs `pip install memstrand[kuzu]`) |
| `MEMSTRAND_LLM_BASE_URL` | `http://localhost:47811/v1` (placeholder — point it at your actual server: Ollama `http://localhost:11434/v1`, LM Studio `http://localhost:1234/v1`, llama.cpp `--port 47811`) | OpenAI-compatible endpoint for `improve()` distillation, `grounded_answer()`, and the `/v1` proxy's upstream |
| `MEMSTRAND_LLM_MODEL` | `local` | model name sent to the LLM endpoint |
| `MEMSTRAND_LLM_API_KEY` | `local` | bearer key sent to the LLM endpoint (most local servers ignore it) |
| `MEMSTRAND_LLM_API_KEY_COMMAND` | *(unset)* | shell command whose stdout is the LLM bearer key, used instead of `MEMSTRAND_LLM_API_KEY` — for gateways with short-lived tokens. Re-run at most once a minute, so a long-running daemon picks up refreshed tokens; if it fails, the last good token (else `MEMSTRAND_LLM_API_KEY`) is used. Also applies to `MEMSTRAND_EMBED_BACKEND=openai` |
| `MEMSTRAND_EMBED_BACKEND` | `fastembed` | `fastembed` (local ONNX, downloads `BAAI/bge-small-en-v1.5` on first use) or `openai` (any OpenAI-compatible `/embeddings` endpoint) |
| `MEMSTRAND_EMBED_MODEL` | *(fastembed's built-in default)* | required when `MEMSTRAND_EMBED_BACKEND=openai` |
| `MEMSTRAND_EMBED_BASE_URL` | falls back to `MEMSTRAND_LLM_BASE_URL` | endpoint used when `MEMSTRAND_EMBED_BACKEND=openai` |
| `MEMSTRAND_RERANK_MODEL` | *(unset — off)* | cross-encoder reranker model, e.g. `Xenova/ms-marco-MiniLM-L-6-v2` |
| `MEMSTRAND_MIN_RELEVANCE` | `0.0` | drop `recall()` hits below this absolute relevance score; model-dependent, tune per embedding/reranker |
| `MEMSTRAND_HOST` | `127.0.0.1` | daemon bind host |
| `MEMSTRAND_PORT` | `7777` | daemon bind port |
| `MEMSTRAND_CONSOLIDATE_HOURS` | `24` | hours between scheduled `consolidate()` passes |
| `MEMSTRAND_TASK` | *(unset)* | default task scope for `hook claude-code --task-from env` and the `/v1` proxy when no `x-memstrand-task` header is sent |
| `MEMSTRAND_TOKEN` | *(unset)* | bearer token required on `/api`, `/mcp`, `/v1`; **mandatory** if `MEMSTRAND_HOST` is non-loopback (this now works end-to-end for MCP too, not just REST — see [docs/cli.md](cli.md#security-notes)) |
| `MEMSTRAND_ALLOWED_ORIGINS` | *(unset)* | comma-separated extra browser `Origin`s to trust in addition to the daemon's own host:port, e.g. `http://localhost:6274` for the MCP inspector, or the web UI's origin when binding to a non-loopback host for LAN access — on top of the loopback/token checks, not a replacement for them. Each entry must be exactly `scheme://host[:port]` (`http`/`https`, no path/query/fragment); `null`, `*`, and other malformed entries make the daemon refuse to start with an error naming the bad entry. There is no same-origin-by-Host fallback — an Origin/Referer matching the request's `Host` header is **not** automatically trusted (that was a DNS-rebinding hole); it must be the daemon's own origin or explicitly listed here. |

Embedding dimension is auto-detected from a probe embed call and guarded
against mismatches (e.g. switching embedding models on an existing store).

## Data layout

Everything lives under `MEMSTRAND_HOME` (default `~/.memstrand`):

- `memstrand.db` — SQLite: graph store (atoms/edges/FSRS state), keyword
  (FTS5) index, session cache, and the append-only provenance ledger.
- `qdrant/` — embedded, on-disk Qdrant (unless `MEMSTRAND_QDRANT_URL` points
  at a real server).
- `kuzu/` — only used when `MEMSTRAND_GRAPH_BACKEND=kuzu`.
- `backups/` — timestamped `memstrand backup` output.
- `scope_cache.json` — repo root → scope key cache for the hooks and
  `--repo <path>` (invalidated when the repo's git config changes).
- `backups/pre-scoping-v2-*.jsonl` — the automatic backup written before
  the one-time scoping-v2 migration re-keys a pre-0.3 store.
- `hook.log` — written by the Claude Code hooks when they can't reach/
  authenticate with a running daemon (see [docs/cli.md](cli.md#security-notes)).

Everything memstrand creates under `MEMSTRAND_HOME` is private by
construction: directories are created `0700` and files `0600`, so the
store, backups, and hook log aren't readable by other local users.

See [docs/architecture.md](architecture.md) for how the stores interact, and
[docs/integrations.md](integrations.md) for local-LLM-specific configuration
recipes (Ollama, llama.cpp, LM Studio).
