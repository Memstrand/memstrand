# Integrations

Detailed recipes for wiring memstrand into a local-LLM workflow. Start here
after the README's Quickstart section. All of these assume the daemon is
running:

```bash
memstrand serve
```

which listens on `127.0.0.1:7777` by default (`MEMSTRAND_HOST` /
`MEMSTRAND_PORT` to change) and exposes:

- MCP (streamable HTTP) at `/mcp`
- REST at `/api`
- the web UI at `/`
- an OpenAI-compatible, memory-injecting chat proxy at `/v1`

Run `memstrand config` any time to see the effective settings (home
directory, backends, endpoints, whether a token is set).

---

## Local LLMs

memstrand needs two things from a local model server: a **chat** endpoint
(for `improve()` distillation and `grounded_answer()`) and an
**embeddings** endpoint (unless you keep the fastembed default). Both are
configured as one OpenAI-compatible base URL plus model names — see
[docs/configuration.md](configuration.md) for the full variable reference.

### Ollama

Pull a chat model and an embedding model:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

Point memstrand at Ollama's OpenAI-compatible API:

```bash
export MEMSTRAND_LLM_BASE_URL=http://localhost:11434/v1
export MEMSTRAND_LLM_MODEL=llama3.1
export MEMSTRAND_EMBED_BACKEND=openai
export MEMSTRAND_EMBED_MODEL=nomic-embed-text
# MEMSTRAND_EMBED_BASE_URL defaults to MEMSTRAND_LLM_BASE_URL, so it's
# optional here since Ollama serves both chat and embeddings on :11434.

memstrand serve
```

### llama.cpp server

```bash
llama-server -m your-chat-model.gguf --port 47811
```

`http://localhost:47811/v1` is memstrand's own default
`MEMSTRAND_LLM_BASE_URL`, so if you're only using llama.cpp for chat you
don't need to set anything. For embeddings, run a second `llama-server`
instance with an embedding model (`--embedding`) on another port and point
`MEMSTRAND_EMBED_BASE_URL` at it:

```bash
llama-server -m your-embedding-model.gguf --port 8082 --embedding

export MEMSTRAND_EMBED_BACKEND=openai
export MEMSTRAND_EMBED_MODEL=your-embedding-model
export MEMSTRAND_EMBED_BASE_URL=http://localhost:8082/v1
```

Or skip all of this and leave `MEMSTRAND_EMBED_BACKEND=fastembed` (the
default) — embeddings then run locally via ONNX, independent of your chat
model.

### LM Studio

Start LM Studio's local server (Developer tab -> Start Server; default
`http://localhost:1234/v1`), load a chat model, and:

```bash
export MEMSTRAND_LLM_BASE_URL=http://localhost:1234/v1
export MEMSTRAND_LLM_MODEL=<the model name shown in LM Studio>
```

LM Studio also serves `/v1/embeddings` for models that support it — load an
embedding model and set `MEMSTRAND_EMBED_BACKEND=openai` /
`MEMSTRAND_EMBED_MODEL` the same way as above, or stick with the fastembed
default.

---

## Claude Code

### MCP (tool access: remember / recall / context / answer / update / forget / log_turn / improve / ingest / stats)

With the daemon running:

```bash
claude mcp add --transport http memstrand http://127.0.0.1:7777/mcp
```

If you set `MEMSTRAND_TOKEN` (required once the daemon isn't bound to
loopback-only), pass it as a header:

```bash
claude mcp add --transport http memstrand http://127.0.0.1:7777/mcp \
  --header "Authorization: Bearer $MEMSTRAND_TOKEN"
```

This also works against a non-loopback daemon (`MEMSTRAND_HOST=0.0.0.0`)
as long as `MEMSTRAND_TOKEN` is set and sent as the header above — see
[docs/cli.md](cli.md#security-notes) for the full security model. If
you're pointing a browser-based MCP client (e.g. the MCP inspector) at
the daemon, add its origin to `MEMSTRAND_ALLOWED_ORIGINS` — see
[docs/configuration.md](configuration.md).

Equivalent `.mcp.json` / `~/.claude/mcp.json` entry:

```json
{
  "mcpServers": {
    "memstrand": {
      "type": "http",
      "url": "http://127.0.0.1:7777/mcp"
    }
  }
}
```

### Hooks (automatic transcript logging + context injection, no tool calls needed)

Print the hooks snippet and paste the `"hooks"` key into
`~/.claude/settings.json`:

```bash
memstrand hook print-config
```

This wires:

- `Stop` and `PreCompact` -> `memstrand hook claude-code` — logs any new
  transcript turns and kicks off a background `improve()` distillation
  pass.
- `UserPromptSubmit` -> `memstrand hook claude-code-context` — injects a
  memory context block (standing instructions/preferences + relevant
  memories) into the prompt. This only queries memory when the daemon is
  already running (a cold local `Memory()` is too slow for a hook that runs
  before every prompt) — so run `memstrand serve` first.

Both hook subcommands scope memories by the cwd's repo
(`--task-from auto`, the default, or `cwd`): its normalized git remote URL (e.g.
`github.com/org/repo`, the same for every clone on every machine), or
`local:<hostname>:<path>` when there's no remote. Use `--task-from none`
for no scoping, or `--task-from env` to read `MEMSTRAND_TASK` instead.
With `auto`, the Stop/PreCompact hook also records which *other* repos each
turn touched — files Claude read/edited/searched, `cd` targets, and paths
you mention in prompts (only paths inside git repos count) — and
distillation can file a fact under those repos. So asking Claude in
`~/.claude` to "read ~/projects/memstrand" files what it learns about
memstrand under memstrand.

When memstrand can't tell where a memory belongs — the extractor marks it
ambiguous, or an "always do X" it couldn't confirm came from you — it's
saved narrowly (never global) and queued. The `UserPromptSubmit` hook then
adds a short "scope check" asking Claude to put the question to you (at
most 3 at a time, each once per session, and it stops asking after 5
sessions) and to record your answer with `memstrand scope <id> …`. You can
also run `memstrand review` yourself.

The context hook also reads paths in your prompt: asking from `~/.claude`
to "read ~/projects/memstrand …" injects memstrand's memories alongside
`.claude`'s and the global ones. Every memory records who wrote it
(`claude-code@<host>`, `cli@<host>`, `openai-proxy@<host>`, `mcp@<host>`);
see it with `memstrand show <id>` or filter with `memstrand list --agent`.
A memory can belong to several repos at once — see `memstrand scope` in
[docs/cli.md](cli.md#scopes).

### Backfilling existing sessions

The hooks only see sessions from the moment they're installed. To distill
sessions already on disk:

```bash
memstrand hook backfill --dry-run              # what would be processed, ~tokens
memstrand hook backfill --limit 3              # pilot: the 3 oldest sessions
memstrand hook backfill --project my-repo      # one project's sessions
memstrand hook backfill                        # everything under ~/.claude/projects
```

Sessions are fed to distillation in chunks (`--chunk`, default 40 turns)
so a long session never becomes one huge extraction call. It's resumable
and shares progress with the live Stop hook, so interrupting and
re-running is safe. Distillation calls your LLM for every chunk — use
`--dry-run` first to see how much text that is.

Distilled memories keep a link to their source session. To get back to
the original conversation — e.g. a memory says "switched to Kaniko" and
you want the discussion behind it:

```bash
memstrand show <memory-id>              # includes "session: claude-code:<uuid>"
memstrand session <uuid-or-prefix>      # the logged conversation (text turns)
memstrand search-sessions "kaniko artifactory"
```

memstrand stores the text of each turn, not tool calls/outputs. Claude
Code deletes its own transcripts after 30 days by default
(`cleanupPeriodDays`); keep a separate transcript archive if you need the
full sessions, and point `memstrand hook backfill` at restored copies to
distill sessions older than that.

If a daemon is running but rejects the hook's `MEMSTRAND_TOKEN` (missing
or wrong), the hook fails fast with a "set MEMSTRAND_TOKEN" message rather
than silently opening a second, unscoped local store — see
[docs/cli.md](cli.md#security-notes). That failure (and anything else that
goes wrong) is logged to `{MEMSTRAND_HOME}/hook.log`, since hook stderr
isn't normally visible in Claude Code's UI.

---

## Cursor

Add an MCP server pointing at the daemon's `/mcp` endpoint. In Cursor's
`~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (per-project):

```json
{
  "mcpServers": {
    "memstrand": {
      "type": "http",
      "url": "http://127.0.0.1:7777/mcp"
    }
  }
}
```

With a token set, add it as a header the same way your MCP client supports
(consult Cursor's current MCP docs for the exact header field name, since
this has changed across versions).

---

## Open WebUI

memstrand's `/v1` endpoint is an OpenAI-compatible chat-completions proxy
that injects a memory context block into every request and logs both sides
of the conversation back into memory. Point Open WebUI's OpenAI connection
at it instead of (or in addition to) your local LLM directly:

1. Admin Settings -> Connections -> OpenAI API
2. **Base URL**: `http://127.0.0.1:7777/v1`
3. **API key**: anything, unless `MEMSTRAND_TOKEN` is set — then use that
   token as the API key (Open WebUI sends it as `Authorization: Bearer
   <key>`, which is exactly what memstrand checks).

The proxy forwards everything else straight to `MEMSTRAND_LLM_BASE_URL`
(your real Ollama/llama.cpp/LM Studio server), so `GET /v1/models`
reflects whatever that upstream reports.

Per-request headers, if you need to override behavior from a client:

- `x-memstrand-disable: 1` — skip memory injection and logging for this
  request.
- `x-memstrand-task: <id>` — scope this conversation to a task (overrides
  `MEMSTRAND_TASK`).
- `x-memstrand-scopes: <a>,<b>` — additional scopes whose memories are
  injected (comma-separated scope keys).
- `x-memstrand-session: <id>` — override the session id used for logging
  (defaults to `proxy-<today's date>`).

---

## Running the daemon at login

### macOS (launchd)

`~/Library/LaunchAgents/dev.memstrand.serve.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>dev.memstrand.serve</string>
  <key>ProgramArguments</key>
  <array>
    <string>/path/to/.venv/bin/memstrand</string>
    <string>serve</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/tmp/memstrand.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/memstrand.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>MEMSTRAND_LLM_BASE_URL</key>
    <string>http://localhost:11434/v1</string>
  </dict>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/dev.memstrand.serve.plist
```

### Linux (systemd --user)

`~/.config/systemd/user/memstrand.service`:

```ini
[Unit]
Description=memstrand daemon

[Service]
ExecStart=/path/to/.venv/bin/memstrand serve
Restart=on-failure
Environment=MEMSTRAND_LLM_BASE_URL=http://localhost:11434/v1

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable --now memstrand.service
```

---

## Backups via cron

```bash
crontab -e
```

```cron
# daily backup at 3am, written to $MEMSTRAND_HOME/backups/
0 3 * * * /path/to/.venv/bin/memstrand backup
```

Restore (merges into the current store; never replaces):

```bash
memstrand restore ~/.memstrand/backups/memstrand-20261014-030000.jsonl
```

---

## Obsidian vault watch

Ingest a vault once:

```bash
memstrand ingest ~/Documents/MyVault --task notes
```

Then keep it in sync continuously (polls for changed/new/deleted files):

```bash
memstrand ingest ~/Documents/MyVault --task notes --watch --interval 10
```

`--watch` talks to the daemon if one is running (so ingestion goes through
the shared store) and falls back to a local, private `Memory()` otherwise.
Only changed files are re-ingested (sha256 change detection); `[[wikilink]]`
targets become `related_to` graph edges between notes. A missing vault path
is an error; symlinks inside the vault that point outside its root, and
non-regular files (FIFOs, device files, sockets — not uncommon in a synced
folder), are skipped rather than followed/read.

Change tracking is per `(task, path)`, so the same vault can be ingested
into more than one task independently:

```bash
memstrand ingest ~/Documents/MyVault --task notes
memstrand ingest ~/Documents/MyVault --task research
```

Each `--task` gets its own change-detection state and its own scoped
copies of the notes; re-ingesting one task's changes never touches the
other task's atoms (or the global ones from a taskless ingest).

See also: [docs/configuration.md](configuration.md) for every
`MEMSTRAND_*` variable, and [docs/cli.md](cli.md) for the full REST/MCP/CLI
reference.
