# CLI reference

`memstrand <command> --help` for full flags on any of these:

| Command | What it does |
|---|---|
| `serve` | run the daemon (REST + MCP + web UI + `/v1` proxy) |
| `mcp` | run the MCP server over stdio (refuses to start if a daemon already owns the stores) |
| `ui` | open the web UI in a browser |
| `remember <content>` | store one memory (`-` to read content from stdin) |
| `recall <query>` | scoped hybrid recall |
| `list` | list memories (filter by `--category`, `--task`, `--repo`, `--global`) |
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
| `scope <id>` | show or change which repos/tasks a memory applies to |
| `scopes` / `scopes rename <old> <new>` | known scopes with counts / re-key one scope |
| `review` | memories waiting for you to decide their scope (also a tab in the web UI) |
| `tags` | most-used tags (`--repo` to narrow) |
| `tag <id>` | show or change a memory's tags (`--set`, `--add`, `--remove`) |
| `retag` | LLM-tag existing memories using their source conversations |
| `reindex` | rebuild vector + keyword indexes from the graph store (no LLM) |
| `session <id>` | print a session's logged conversation (full id, Claude Code UUID, or unique prefix) |
| `search-sessions <query>` | full-text search over logged session turns |
| `stats` | atom/edge/session counts, by category and task |
| `usage` | LLM tokens used by distillation/answers, by day, purpose, session, or model |
| `config` | effective configuration |
| `hook claude-code` / `hook claude-code-context` / `hook print-config` | Claude Code integration |
| `hook backfill [paths…]` | distill existing Claude Code transcripts from disk |

Every data-touching command talks to a running daemon if one is up,
otherwise opens a local store directly.

## Scopes

A memory belongs to a *set* of scopes (see
[design/scoping-v2.md](design/scoping-v2.md)): repo keys like
`github.com/org/repo` (the normalized git remote), `local:<host>:<path>`
for folders without a remote, named tasks like `notes`, and `global`. A
query in scope X sees memories whose set contains X, plus global ones.

`--repo` (repeatable where noted) accepts a full key, a short name (the
last path segment, e.g. `memstrand` — an ambiguous short name is an error
listing the candidates), or a filesystem path, which is resolved through
git. `--task X` still works and means the single scope `X`.

`<role>` for `log-turn` (and the REST `/api/turns` / MCP `log_turn`
surfaces below) must be one of `user`, `assistant`, `system`, `tool`. REST
rejects any other value with a 400; the MCP tool's `role` parameter is
typed so the MCP SDK rejects an invalid value before the tool runs.

## Flags by command

- `serve`: `--host`, `--port`, `--no-scheduler` (disable the background
  consolidation scheduler)
- `remember <content>`: `--category`, `--provenance`, `--confidence`,
  `--task`, `--repo` (repeatable), `--global` (with `--repo`: also global),
  `--host`, `--ttl-days`, `--source`. Remembering content that already
  exists adds the new scopes to it.
- `recall <query>`: `--task`, `--repo` (repeatable), `--host`, `--limit`, `--all` (include
  superseded), `--json`
- `list`: `--category`, `--task`, `--repo` (memories in that scope),
  `--agent` (who wrote it, e.g. `claude-code@myhost`),
  `--global` (only memories in the global scope),
  `--limit`, `--offset`, `--json`
- `show <id>`: `--json`
- `edit <id>`: `--content`, `--category`, `--confidence`
- `forget <id>`: `--yes`
- `log-turn <session> <role> <content>`: `--task`
- `improve`: `--session` (repeatable), `--background`
- `answer <question>`: `--task`, `--repo` (repeatable)
- `context [query]`: `--task`, `--repo` (repeatable), `--limit`
- `ingest <path>`: `--task`, `--force`, `--watch`, `--interval` — a missing
  `path` is an error (not a silent no-op); symlinks that resolve outside
  `path`'s own root and non-regular files (FIFOs, device files, sockets)
  are skipped rather than followed/read
- `export`: `--format {jsonl,markdown}`, `-o/--output`
- `import <file>`: positional file, or `-` for stdin
- `backup`: `-o/--output` (defaults to `{home}/backups/memstrand-<ts>.jsonl`)
- `restore <file>`: `--yes`
- `stats`: `--json`
- `scope <id>`: no flags prints the scope set; `--set` (replace),
  `--add`, `--remove` (all repeatable, same values as `--repo`),
  `--global`, `--keep` (confirm as-is). Any of these resolves a pending
  scope review. Removing every scope is refused — use `forget`.
- `review`: `--limit`, `--json`
- `remember`: `--tag` (repeatable). `recall`: `--tag` (repeatable; any of
  them, exact). `list`: `--tag`, `--show-tags`. `show` prints tags and
  memories sharing 2+ tags.
- `retag`: `--missing-only`, `--session` (repeatable), `--limit N`
  (sessions), `--dry-run`. One LLM call per source session, with the
  session's logged conversation as context (memories with no session are
  tagged from their text, in batches); replaces each memory's tags.
  Prints tokens per session.
- `session <id>`: `--limit`, `--json`. `memstrand show <id>` prints each
  memory's source session.
- `search-sessions <query>`: `--session`, `--limit`, `--json`. Every word
  must match (FTS5). Raw turns are not scope-filtered — like `list`/`show`,
  these are owner-facing; neither is exposed as an MCP tool.
- `scopes`: `--json`; `scopes rename <old> <new>` re-keys every memory in
  `old` (e.g. a folder name to its remote URL)
- `hook claude-code`: `--task-from {auto,cwd,none,env}`, `--no-improve`.
  `cwd` scopes by the cwd's repo key (remote URL, else `local:` path);
  `auto` (default) also records per-turn touched repos for distillation
- `hook claude-code-context`: `--task-from {auto,cwd,none,env}`, `--limit`
- `hook backfill [paths…]`: paths are transcripts, project folders, or the
  projects folder (default `~/.claude/projects`); `--project` (substring,
  repeatable), `--since YYYY-MM-DD`, `--limit N` (oldest first), `--chunk`
  (turns per distillation call, default 40), `--min-turns N` (smaller
  sessions are logged and searchable but never distilled — no LLM cost),
  `--skip-active-minutes`
  (default 10), `--task-from`, `--no-improve`, `--dry-run`, `--timeout`.
  Subagent transcripts are skipped. Resumable: progress is shared with the
  live Stop hook, so re-running continues where it stopped and never
  re-logs turns; `--limit N` picks the next N sessions that still have
  unprocessed turns, so repeated runs work through everything in batches.
  Prints LLM tokens per session and per run.
- `usage`: `--by {day,purpose,session,model}`, `--since YYYY-MM-DD`,
  `--json`. Reads `{MEMSTRAND_HOME}/llm_usage.jsonl`, where every LLM call
  records the token usage the endpoint reported, tagged by purpose
  (`extract`, `trust`, `judge`, `answer`, `ping`) and session.
- `hook print-config`: prints the Claude Code `settings.json` hooks snippet

## REST / MCP reference

**REST** (`/api`, JSON): `GET /health`, `GET /stats`, `POST /remember`,
`POST /recall`, `GET /atoms`, `GET|PATCH|DELETE /atoms/{id}`,
`POST /turns`, `POST /improve`, `POST /answer`, `POST /context`,
`POST /consolidate`, `POST /ingest`, `GET /export`, `POST /import`,
`POST /forget_source`, `POST /atoms/{id}/scopes` (`{"set"|"add"|"remove": [...]}`),
`GET /sessions/{id}/turns` (`?limit=`), `GET /turns/search` (`?q=&limit=&session=`),
`POST /atoms/{id}/tags` (`{"set"|"add"|"remove": [...]}`), `GET /atoms/{id}/related`
(`?min_shared=&limit=`), `GET /tags` (`?scope=&limit=`), `POST /retag`
(`{"atom_ids": [...], "session_id"?}`), `POST /reindex`, `tags` on
`remember`/`recall`, `?tag=` on `GET /atoms`,
`GET /scopes`, `POST /scopes/rename` (`{"old","new"}`), `GET /reviews`
(`?limit=`, read-only), `POST /reviews` (`{"limit","session"}`: each review
returned once per session and counted as shown), and `{"keep": true}` on
`/atoms/{id}/scopes`. `remember`,
`recall`, `context`, and `answer` accept `scopes: [...]` alongside the
single-scope `task_id`. `GET /atoms/{id}` defaults to the unscoped owner
view; pass `?scoped=1&task_id=..&host=..` (or repeated `&scope=..`) to apply the same scope rule as
`recall()`.

**MCP** (`/mcp`, streamable HTTP, or stdio via `memstrand mcp`) tools:
`remember`, `recall`, `context`, `answer`, `update`, `forget`, `log_turn`,
`improve`, `ingest`, `stats`, `scope` (`remember`/`recall`/`context`/`answer`
take an optional `scopes` list) — plus a `memory_context` prompt and a
`memstrand://context` resource for standing instructions/preferences.
`category`/`provenance` tool arguments are validated against the
`Category`/`Provenance` enums; an invalid value is rejected with an
actionable error (listing the valid values) instead of a raw stack trace,
same for any other tool-call failure — MCP clients get an error message
they can act on, not just "internal error".

**OpenAI-compatible proxy** (`/v1`): `GET /models`, `POST
/chat/completions` (streaming and non-streaming) — injects a `context()`
block into the system message and logs both sides of the conversation,
then forwards to `MEMSTRAND_LLM_BASE_URL`. Streaming responses forward the
upstream's error status verbatim (e.g. a 429 or 500 from your local LLM
server reaches the client as that same status, not a generic 502) rather
than masking it.

## Security notes

- Binds to `127.0.0.1` by default. A non-loopback `MEMSTRAND_HOST` refuses
  to start unless `MEMSTRAND_TOKEN` is set. With a token set, this now
  works end-to-end for MCP as well as REST/the proxy — `/mcp` enforces the
  same bearer-token check as `/api` and `/v1` on a non-loopback bind.
- When loopback, requests with an untrusted `Host` header are rejected
  (DNS-rebinding guard). When bound to a specific non-loopback host or
  address (not `0.0.0.0`/`::`), the same `Host` check applies: it must
  match the bind host or the host of an allowed origin. A wildcard bind
  (`0.0.0.0`/`::`) has no single "own Host" to check against, so this
  check is skipped for it (the Origin/token checks still apply).
- `/api`, `/mcp`, and `/v1` require `Authorization: Bearer
  <MEMSTRAND_TOKEN>` on every request if a token is configured.
- Mutating requests (anything but `GET`/`HEAD`/`OPTIONS`) with a
  cross-origin `Origin` (or Referer-derived origin) header are rejected —
  a CSRF guard for browser-based clients (the web UI, Open WebUI). Set
  `MEMSTRAND_ALLOWED_ORIGINS` (comma-separated) to trust additional
  browser origins beyond the daemon's own host:port — e.g. the MCP
  inspector's dev-server origin, or the web UI's origin when accessing a
  non-loopback-bound daemon over the LAN — on top of, not instead of, the
  loopback/token checks. **There is no same-origin-by-Host fallback**: an
  Origin/Referer that merely matches the request's own `Host` header is
  *not* automatically trusted — that was a DNS-rebinding hole (an
  attacker who controls DNS, or a malicious page the victim's browser
  resolves, controls both headers). It must be the daemon's own origin or
  explicitly listed in `MEMSTRAND_ALLOWED_ORIGINS`. Requests with neither
  header (the CLI, MCP clients, hooks) are unaffected — the bearer token
  is still mandatory for them. `MEMSTRAND_ALLOWED_ORIGINS` entries are
  strictly validated at startup (`scheme://host[:port]`, `http`/`https`
  only, no path/query/fragment, not `null`/`*`) — a malformed entry makes
  the daemon refuse to start with an error naming it.
- The daemon sends security headers on every response (`X-Frame-Options:
  DENY`, a restrictive `Content-Security-Policy`, etc.) to harden the web
  UI and REST/MCP endpoints against being framed or loaded cross-origin.
- If the CLI, a Claude Code hook, or stdio `memstrand mcp` finds a daemon
  already running but the daemon rejects the client's `MEMSTRAND_TOKEN`
  (missing or wrong), it fails fast with a clear "set MEMSTRAND_TOKEN"
  message instead of silently falling back to opening the stores itself
  (which would be a second, unscoped writer). Hooks log this — and
  anything else that goes wrong — to `{MEMSTRAND_HOME}/hook.log` rather
  than failing silently inside the Claude Code hook pipeline.
- The daemon starts even if the configured embedding backend is
  unreachable (e.g. Ollama isn't running yet) — it doesn't block startup
  on a dependency. The first operation that actually needs an embedding
  (`remember`, `recall`, ingest, etc.) reports a clear `embedding backend
  unreachable` error instead of hanging or crashing the daemon.
- **Scope is an LLM-facing guarantee, not a browse-time one.** The hard
  task/host scope filter (the Playwright-bug fix) applies to everything
  that can end up in front of a model or an agent: `recall()`, `context()`,
  every MCP tool, graph-neighbor hydration, and the `/v1` proxy's injected
  context. The owner-facing browse surfaces — `memstrand list`/`show`, the
  web UI's Browse view, and `GET /api/atoms` — intentionally show memories
  across *all* scopes, since they're for the human operator inspecting
  their own store, not for an agent. On a shared or multi-user machine, put
  `MEMSTRAND_TOKEN` on the daemon so these surfaces aren't open to anyone
  who can reach the port.
- `/api/ingest` reads whatever local filesystem path the caller passes it —
  it's gated by the same loopback/Origin/token checks as the rest of
  `/api`, but there's no separate path allowlist. Don't expose the daemon
  to untrusted callers.
- The web UI supports `MEMSTRAND_TOKEN`: it stores the token in
  `localStorage` (`memstrand.token`), prompts for it automatically on a
  401, and has a gear (⚙) button to set or change it at any time.

See also: [docs/configuration.md](configuration.md) for the full
`MEMSTRAND_*` variable reference, and [docs/integrations.md](integrations.md)
for client-specific setup (Claude Code, Cursor, Open WebUI).
