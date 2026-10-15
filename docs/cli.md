# CLI reference

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

Every data-touching command talks to a running daemon if one is up,
otherwise opens a local store directly.

## Flags by command

- `serve`: `--host`, `--port`, `--no-scheduler` (disable the background
  consolidation scheduler)
- `remember <content>`: `--category`, `--provenance`, `--confidence`,
  `--task`, `--host`, `--ttl-days`, `--source`
- `recall <query>`: `--task`, `--host`, `--limit`, `--all` (include
  superseded), `--json`
- `list`: `--category`, `--task`, `--global` (only task-less memories),
  `--limit`, `--offset`, `--json`
- `show <id>`: `--json`
- `edit <id>`: `--content`, `--category`, `--confidence`
- `forget <id>`: `--yes`
- `log-turn <session> <role> <content>`: `--task`
- `improve`: `--session` (repeatable), `--background`
- `answer <question>`: `--task`
- `context [query]`: `--task`, `--limit`
- `ingest <path>`: `--task`, `--force`, `--watch`, `--interval`
- `export`: `--format {jsonl,markdown}`, `-o/--output`
- `import <file>`: positional file, or `-` for stdin
- `backup`: `-o/--output` (defaults to `{home}/backups/memstrand-<ts>.jsonl`)
- `restore <file>`: `--yes`
- `stats`: `--json`
- `hook claude-code`: `--task-from {cwd,none,env}`, `--no-improve`
- `hook claude-code-context`: `--task-from {cwd,none,env}`, `--limit`
- `hook print-config`: prints the Claude Code `settings.json` hooks snippet

## REST / MCP reference

**REST** (`/api`, JSON): `GET /health`, `GET /stats`, `POST /remember`,
`POST /recall`, `GET /atoms`, `GET|PATCH|DELETE /atoms/{id}`,
`POST /turns`, `POST /improve`, `POST /answer`, `POST /context`,
`POST /consolidate`, `POST /ingest`, `GET /export`, `POST /import`,
`POST /forget_source`. `GET /atoms/{id}` defaults to the unscoped owner
view; pass `?scoped=1&task_id=..&host=..` to apply the same scope rule as
`recall()`.

**MCP** (`/mcp`, streamable HTTP, or stdio via `memstrand mcp`) tools:
`remember`, `recall`, `context`, `answer`, `update`, `forget`, `log_turn`,
`improve`, `ingest`, `stats` — plus a `memory_context` prompt and a
`memstrand://context` resource for standing instructions/preferences.

**OpenAI-compatible proxy** (`/v1`): `GET /models`, `POST
/chat/completions` (streaming and non-streaming) — injects a `context()`
block into the system message and logs both sides of the conversation,
then forwards to `MEMSTRAND_LLM_BASE_URL`.

## Security notes

- Binds to `127.0.0.1` by default. A non-loopback `MEMSTRAND_HOST` refuses
  to start unless `MEMSTRAND_TOKEN` is set.
- When loopback, requests with an untrusted `Host` header are rejected
  (DNS-rebinding guard).
- `/api`, `/mcp`, and `/v1` require `Authorization: Bearer
  <MEMSTRAND_TOKEN>` on every request if a token is configured.
- Mutating requests (anything but `GET`/`HEAD`/`OPTIONS`) with a
  cross-origin `Origin` header are rejected — a CSRF guard for
  browser-based clients (the web UI, Open WebUI).
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
