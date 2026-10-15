# memstrand

[![PyPI](https://img.shields.io/pypi/v/memstrand)](https://pypi.org/project/memstrand/)
[![Python](https://img.shields.io/pypi/pyversions/memstrand)](https://pypi.org/project/memstrand/)
[![License: MIT](https://img.shields.io/github/license/chiranjeevirajukr/memstrand)](LICENSE)

**Local-first typed memory for AI agents. Zero API keys.**

Two strands, like DNA: one strand holds **facts** (vectors), the other holds **relationships** (graph). Every write joins them base-pair by base-pair — a memory is never stored without its connections, and never retrieved without its scope.

0.2 turns this into a *second brain*: one daemon (`memstrand serve`) that Claude Code, Cursor, Open WebUI, and your own scripts all share — over MCP, REST, and an OpenAI-compatible proxy — backed by an embedded, on-disk store. No Docker, no cloud, no API keys.

## 30-second demo

No Docker. No server. No LLM. `pip install` is the entire setup:

```python
from memstrand import Category, Memory, Provenance

mem = Memory.in_memory()

# A fact scoped to one task...
mem.remember(
    "Playwright installed as an agent-testing task dependency",
    category=Category.fact,
    provenance=Provenance.explicit_statement,
    task_id="playwright-test",
)

# ...and a global preference
mem.remember(
    "Chiran prefers a natural, unstylized look in AI portraits",
    category=Category.preference,
    provenance=Provenance.explicit_statement,
    confidence=0.95,
)

# An unrelated session asks about testing setup.
# The Playwright fact does NOT leak across the task boundary:
for hit in mem.recall("testing setup"):
    print(hit["category"], "-", hit["content"])
# preference - Chiran prefers a natural, unstylized look in AI portraits

# Same query WITH the task scope sees it:
for hit in mem.recall("testing setup", task_id="playwright-test"):
    print(hit["category"], "-", hit["content"])
# fact - Playwright installed as an agent-testing task dependency
```

Typed. Scoped. Provenance on every atom. `Memory.in_memory()` runs Qdrant in `:memory:` mode and the graph/session/keyword stores in a temp SQLite file — nothing touches disk and nothing persists past the process. Embeddings are local ONNX (fastembed) by default, downloaded once on first use.

## Second brain in 5 minutes

```bash
pip install memstrand
memstrand serve
```

This starts the daemon on `127.0.0.1:7777` with an embedded, on-disk Qdrant + SQLite under `~/.memstrand` — no Docker, no config. It prints the four endpoints it's serving (MCP, REST, web UI, OpenAI proxy).

**Connect Claude Code** (MCP, for tool-calling: remember/recall/context/answer/...):

```bash
claude mcp add --transport http memstrand http://127.0.0.1:7777/mcp
```

**Add automatic hooks** (transcript logging + distillation + context injection, no tool calls required):

```bash
memstrand hook print-config   # paste the printed "hooks" key into ~/.claude/settings.json
```

**Point Open WebUI at your local LLM through memstrand** (so every chat gets memory injected and logged):

```bash
export MEMSTRAND_LLM_BASE_URL=http://localhost:11434/v1   # Ollama, llama.cpp, LM Studio...
memstrand serve
```

Then set Open WebUI's OpenAI connection Base URL to `http://127.0.0.1:7777/v1`.

**Ingest your notes:**

```bash
memstrand ingest ~/Documents/MyVault --task notes
```

Full recipes (Ollama/llama.cpp/LM Studio model setup, Cursor, launchd/systemd autostart, cron backups, Obsidian `--watch`) are in [`examples/second_brain.md`](examples/second_brain.md).

## Why

Flat vector memory has three failure modes, and every agent hits all three:

1. **No types** — a preference, a decision, and a guess all look the same to cosine similarity.
2. **No scope** — a fact learned while testing Playwright leaks into an unrelated Mac disk-cleanup session. (This actually happened. It's why this exists.)
3. **No provenance** — nobody knows whether a memory was *said*, *inferred*, or *corrected*, or how confident the agent was.

memstrand fixes all three: every memory is **typed** (13 categories), **scoped** (session/task/host), and **carries provenance** (how it was learned, confidence, expiry).

## Configuration

Everything is an environment variable, read once per process via `get_settings()`. Run `memstrand config` to see the effective values.

| Variable | Default | Purpose |
|---|---|---|
| `MEMSTRAND_HOME` | `~/.memstrand` | data directory — SQLite db, embedded Qdrant, ingest/hook state, backups |
| `MEMSTRAND_QDRANT_URL` | *(unset — embedded on-disk Qdrant under `$MEMSTRAND_HOME/qdrant`)* | point at a real Qdrant server instead |
| `MEMSTRAND_GRAPH_BACKEND` | `sqlite` | `sqlite` (default) or `kuzu` (needs `pip install memstrand[kuzu]`) |
| `MEMSTRAND_LLM_BASE_URL` | `http://localhost:47811/v1` (placeholder — point it at your actual server: Ollama `http://localhost:11434/v1`, LM Studio `http://localhost:1234/v1`, llama.cpp `--port 47811`) | OpenAI-compatible endpoint for `improve()` distillation, `grounded_answer()`, and the `/v1` proxy's upstream |
| `MEMSTRAND_LLM_MODEL` | `local` | model name sent to the LLM endpoint |
| `MEMSTRAND_LLM_API_KEY` | `local` | bearer key sent to the LLM endpoint (most local servers ignore it) |
| `MEMSTRAND_EMBED_BACKEND` | `fastembed` | `fastembed` (local ONNX, downloads `BAAI/bge-small-en-v1.5` on first use) or `openai` (any OpenAI-compatible `/embeddings` endpoint) |
| `MEMSTRAND_EMBED_MODEL` | *(fastembed's built-in default)* | required when `MEMSTRAND_EMBED_BACKEND=openai` |
| `MEMSTRAND_EMBED_BASE_URL` | falls back to `MEMSTRAND_LLM_BASE_URL` | endpoint used when `MEMSTRAND_EMBED_BACKEND=openai` |
| `MEMSTRAND_RERANK_MODEL` | *(unset — off)* | cross-encoder reranker model, e.g. `Xenova/ms-marco-MiniLM-L-6-v2` |
| `MEMSTRAND_MIN_RELEVANCE` | `0.0` | drop `recall()` hits below this absolute relevance score; model-dependent, tune per embedding/reranker |
| `MEMSTRAND_HOST` | `127.0.0.1` | daemon bind host |
| `MEMSTRAND_PORT` | `7777` | daemon bind port |
| `MEMSTRAND_CONSOLIDATE_HOURS` | `24` | hours between scheduled `consolidate()` passes |
| `MEMSTRAND_TASK` | *(unset)* | default task scope for `hook claude-code --task-from env` and the `/v1` proxy when no `x-memstrand-task` header is sent |
| `MEMSTRAND_TOKEN` | *(unset)* | bearer token required on `/api`, `/mcp`, `/v1`; **mandatory** if `MEMSTRAND_HOST` is non-loopback |

Embedding dimension is auto-detected from a probe embed call and guarded against mismatches (e.g. switching embedding models on an existing store).

## Architecture

Four swappable stores behind thin protocols (`stores/base.py`), sharing `$MEMSTRAND_HOME` by default:

| Store | Default backend | Role |
|---|---|---|
| Vector | Qdrant (embedded on-disk, or `MEMSTRAND_QDRANT_URL`) | dense semantic search over atom embeddings; scope + TTL filters are baked into every query |
| Graph | SQLite (optional: Kuzu via `memstrand[kuzu]`) | atoms, typed edges, FSRS memory-strength state, access stats |
| Keyword | SQLite FTS5 | BM25 lexical search, fused with vector search |
| Session | SQLite (same file as the graph store) | raw per-session turns (`log_turn`) + an append-only provenance ledger |

**Write path:** `remember()` embeds the content once and writes the atom to the vector, graph, and keyword stores under one scope-aware content-hash id (`mem:<sha256(content + task_id + host)>`), then logs the write to the provenance ledger. Remembering identical content in the same scope twice is a no-op id, not a duplicate. `log_turn()` is just an append — no embedding, no extraction — so a live chat session costs nothing until it's distilled.

**Distillation (`improve()`):** extracts typed candidate atoms from a session's new turns via your local LLM, checks each candidate against similar existing memories with a second "judge" LLM call (add / update / delete / none — Mem0's resolution pattern), and promotes survivors with `supersedes` or `contradicts` edges as needed. Nothing is ever silently deleted — corrections and contradictions are additive edges, and the old atom stays queryable.

**Read path (`recall()`):** scope (session/task/host) is a hard filter applied *before* ranking, not a ranking signal — a scoped query sees its own scope plus globals; an unscoped query sees only globals (the Playwright-bug fix). Dense and keyword search run in parallel; relevance is computed as an *absolute* signal per candidate (not normalized within the result set — a lone weak match no longer scores a perfect 1.0 just for being "the best of one"), and hits below `MEMSTRAND_MIN_RELEVANCE` are dropped. One hop of graph neighbors is hydrated onto every result (neighbors outside the query's scope are dropped — the scope guarantee applies to hydration, not just the initial match); contradictions are surfaced, never hidden.

**Memory dynamics (`consolidate()`):** per-atom FSRS state (difficulty/stability/next-review) updates on every successful recall; a scheduled pass expires TTL'd atoms, decays and prunes weak Hebbian edges (`strengthen_edge`/`decay_edges`), links near-duplicate atoms (cosine ≥ 0.95, same scope + category) via `supersedes` instead of deleting, and replays the highest-priority due atoms to keep them stable. The daemon runs this every `MEMSTRAND_CONSOLIDATE_HOURS`, plus background `improve()` for any pending sessions.

## Retrieval & ranking

`recall()`'s final score blends five signals (`ranking.blended_score`):

- **0.65 relevance** — an *absolute* 0..1 signal, not normalized within the result set (a single weak candidate can't score a perfect 1.0 just for being "the best of one"). With `MEMSTRAND_RERANK_MODEL` set, it's `sigmoid(cross-encoder score)`; otherwise it's `max(clipped cosine similarity, keyword strength = bm25/(bm25+2))`, plus a small bonus when both the dense and keyword retrievers find the same atom (`ranking.absolute_relevance`).
- **0.12 confidence** — the atom's stored confidence
- **0.10 recency** — exponential decay, 90-day half-life, off whichever of `updated_at`/`last_accessed_at` is newer
- **0.08 provenance** — explicit/validated/corrected > observed > imported > inferred
- **0.05 access frequency** — capped at 10 accesses

`MEMSTRAND_RERANK_MODEL` and recall's `limit`/`include_superseded` aside, `MEMSTRAND_MIN_RELEVANCE` (or `recall(min_relevance=...)`) drops hits whose relevance score falls below the threshold before blending — default `0.0` (no filtering), since the right cutoff is model-dependent (cosine distributions differ across embedding models). Superseded atoms are excluded by default (`include_superseded=True` to see them, at half score) and TTL-expired atoms never come back.

## Python API

```python
from memstrand import Memory, Category, Provenance

mem = Memory()                 # persistent: on-disk Qdrant + SQLite under MEMSTRAND_HOME
mem = Memory.in_memory()       # ephemeral: temp dir, in-memory Qdrant — for demos/tests

# write path
mem.remember(content, category=Category.fact, provenance=Provenance.explicit_statement,
             confidence=0.8, session_id=None, task_id=None, host=None, ttl_days=None,
             source=None, source_ref=None, agent=None)              # -> MemoryAtom
mem.log_turn(session_id, role, content, task_id=None)                # raw turn, cheap
mem.update(atom_id, content=None, category=None, confidence=None)    # -> MemoryAtom (supersedes if content changed)
mem.forget(atom_id)                                                  # -> bool
mem.forget_source(source, task_id=..., host=...)                     # -> int; default = every scope, exact-scope match if given

# scoped read path
mem.recall(query, session_id=None, task_id=None, host=None, categories=None,
           limit=10, include_superseded=False, track=True,
           min_relevance=None)       # -> list[dict]; defaults to MEMSTRAND_MIN_RELEVANCE; neighbors are scope-filtered too
mem.get(atom_id, task_id=None, host=None, scoped=False)              # -> dict | None; scoped=False (default) is the owner-facing browse view (all scopes)
mem.list(category=None, task_id=None, host=None, limit=100, offset=0)  # -> list[dict]
mem.count()                                                           # -> int
mem.stats()                                                           # -> dict

# context injection / grounded answering
mem.context(query=None, task_id=None, host=None, limit=8, max_chars=4000)  # -> str (Markdown block)
mem.grounded_answer(question, **recall_kwargs)                       # -> str (needs a local LLM)

# sessions / distillation / consolidation
mem.improve(session_ids=None, background=False)                      # -> dict
mem.consolidate(now=None)                                            # -> dict

# export / import / ingest
mem.export(fmt="jsonl")                                              # -> str ("jsonl" or "markdown")
mem.import_jsonl(text)                                                # -> dict
mem.ingest(path, task_id=None, force=False)                           # -> dict

mem.close()  # or: with Memory() as mem: ...
```

`memstrand.client.get_memory()` returns a `RemoteMemory` (same method surface, talks to `memstrand serve` over REST) if a daemon is reachable, else falls back to a local `Memory()` — the embedded stores are single-writer, so this is how the CLI and hooks avoid corrupting a daemon's state.

## CLI reference

`memstrand <command> --help` for full flags on any of these:

| Command | What it does |
|---|---|
| `serve` | run the daemon (REST + MCP + web UI + `/v1` proxy) |
| `mcp` | run the MCP server over stdio (refuses to start if a daemon already owns the stores) |
| `ui` | open the web UI in a browser |
| `remember <content>` | store one memory (`-` to read content from stdin) |
| `recall <query>` | scoped hybrid recall |
| `list` | list memories (filter by `--category`, `--task`, `--global`) |
| `show <id>` | one memory in detail, with neighbors and history |
| `edit <id>` | edit a memory (a changed `--content` supersedes the old atom) |
| `forget <id>` | delete a memory |
| `forget-source <source>` | delete every memory ingested from one source |
| `log-turn <session> <role> <content>` | append one raw session turn |
| `improve` | distill raw sessions into typed memory |
| `answer <question>` | answer using only stored memories |
| `context [query]` | print the context block memstrand would inject |
| `consolidate` | expire, decay, dedup, replay, and distill |
| `ingest <path>` | ingest a file or folder of notes (`--watch` to keep syncing) |
| `export` / `import` | full jsonl or markdown snapshot |
| `backup` / `restore` | timestamped backup, merge-restore |
| `stats` | atom/edge/session counts, by category and task |
| `config` | effective configuration |
| `hook claude-code` / `hook claude-code-context` / `hook print-config` | Claude Code integration |

Every data-touching command talks to a running daemon if one is up, otherwise opens a local store directly.

## REST / MCP reference

**REST** (`/api`, JSON): `GET /health`, `GET /stats`, `POST /remember`, `POST /recall`, `GET /atoms`, `GET|PATCH|DELETE /atoms/{id}`, `POST /turns`, `POST /improve`, `POST /answer`, `POST /context`, `POST /consolidate`, `POST /ingest`, `GET /export`, `POST /import`, `POST /forget_source`. `GET /atoms/{id}` defaults to the unscoped owner view; pass `?scoped=1&task_id=..&host=..` to apply the same scope rule as `recall()`.

**MCP** (`/mcp`, streamable HTTP, or stdio via `memstrand mcp`) tools: `remember`, `recall`, `context`, `answer`, `update`, `forget`, `log_turn`, `improve`, `ingest`, `stats` — plus a `memory_context` prompt and a `memstrand://context` resource for standing instructions/preferences.

**OpenAI-compatible proxy** (`/v1`): `GET /models`, `POST /chat/completions` (streaming and non-streaming) — injects a `context()` block into the system message and logs both sides of the conversation, then forwards to `MEMSTRAND_LLM_BASE_URL`.

## Security notes

- Binds to `127.0.0.1` by default. A non-loopback `MEMSTRAND_HOST` refuses to start unless `MEMSTRAND_TOKEN` is set.
- When loopback, requests with an untrusted `Host` header are rejected (DNS-rebinding guard).
- `/api`, `/mcp`, and `/v1` require `Authorization: Bearer <MEMSTRAND_TOKEN>` on every request if a token is configured.
- Mutating requests (anything but `GET`/`HEAD`/`OPTIONS`) with a cross-origin `Origin` header are rejected — a CSRF guard for browser-based clients (the web UI, Open WebUI).
- **Scope is an LLM-facing guarantee, not a browse-time one.** The hard task/host scope filter (the Playwright-bug fix) applies to everything that can end up in front of a model or an agent: `recall()`, `context()`, every MCP tool, graph-neighbor hydration, and the `/v1` proxy's injected context. The owner-facing browse surfaces — `memstrand list`/`show`, the web UI's Browse view, and `GET /api/atoms` — intentionally show memories across *all* scopes, since they're for the human operator inspecting their own store, not for an agent. On a shared or multi-user machine, put `MEMSTRAND_TOKEN` on the daemon so these surfaces aren't open to anyone who can reach the port.
- `/api/ingest` reads whatever local filesystem path the caller passes it — it's gated by the same loopback/Origin/token checks as the rest of `/api`, but there's no separate path allowlist. Don't expose the daemon to untrusted callers.
- The web UI supports `MEMSTRAND_TOKEN`: it stores the token in `localStorage` (`memstrand.token`), prompts for it automatically on a 401, and has a gear (⚙) button to set or change it at any time.

## How it compares

| | memstrand | Mem0 | Cognee | raw vector RAG |
|---|---|---|---|---|
| Local-first, no API keys | ✅ | OSS self-host option | ✅ | ✅ |
| Typed memory schema | ✅ 13 categories | ✅ notes | ✅ ontology | ❌ |
| Scope enforcement (hard filter) | ✅ | ❌ | ❌ | ❌ |
| Graph + vector joint writes | ✅ | ✅ | ✅ | ❌ |
| Contradictions as edges, not deletes | ✅ | ❌ | ✅ | ❌ |
| Hybrid (dense + BM25) retrieval, built in | ✅ | — | — | ❌ |
| One daemon shared across tools (MCP + REST + OpenAI proxy) | ✅ | — | — | ❌ |

## Acceptance test

Any memory system should pass these three before you trust it (rubric adapted from a Cognee ad, because it was right):

1. **Remember me in a brand-new chat** without re-pasting anything.
2. **Search by meaning**, not keywords.
3. **Connect facts across sessions** — and *don't* leak facts across unrelated tasks (the Playwright test).

`examples/quickstart.py` runs all three against a persistent store; `tests/test_memory_flow.py` runs the same rubric in CI.

## Acknowledgments

Standing on shoulders, with credit:

- **Memanto** (moorcheh-ai/memanto) — the 13-category typed schema, provenance, and confidence model. Their product is cloud-only; this is the schema reimplemented local-first.
- **Cognee** (topoteretes/cognee) — the pipeline patterns: fast session cache → distillation, joint graph+vector writes, contradictions as additive edges.
- **ZenBrain** (zensation-ai/zenbrain) — per-fact FSRS, deterministic write routing, replay consolidation, Hebbian edges.
- **Mem0** — the ADD/UPDATE/DELETE/NONE extraction resolution pattern.

## Roadmap

- [x] Schema + store interfaces + scaffold
- [x] Session-commit distillation loop (extract → dedup → judge → promote)
- [x] Scoped retrieval (task/host hard filters; scoped queries also see globals)
- [x] Graph neighbor hydration + Hebbian strengthen/decay
- [x] FSRS + replay consolidation pass, scheduled by the daemon
- [x] MCP server tools, over both stdio and streamable HTTP
- [x] One daemon: REST + MCP + web UI + OpenAI-compatible memory-injecting proxy
- [x] Hybrid (dense + BM25) retrieval with absolute, non-normalized relevance scoring + optional cross-encoder reranking
- [x] Markdown/Obsidian/PDF ingestion with change detection and `--watch`
- [x] `memstrand export` / `import` / `backup` / `restore`
- [x] Claude Code hooks (transcript logging, background distillation, context injection)
- [x] Benchmarks: the 3-test rubric, automated (`tests/test_memory_flow.py`)
- [ ] Automated migration tooling from 0.1.x's working-directory layout (today: re-ingest/re-remember — see `CHANGELOG.md`)
- [ ] Multi-user auth beyond a single shared bearer token

## License

MIT — see [LICENSE](LICENSE).
