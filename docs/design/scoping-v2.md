# Design: scoping v2 — repo identity, multi-scope memories, smarter attribution

Status: **Phases 1–4 implemented** (scopes core, attribution, review queue, read side + agent) · Target: 0.3.0 · Breaking: storage format (migrated automatically)

> Refinements made during Phase 1 are marked **(impl)**: they replace the
> original draft text where the two differ.

## Problem

Today every memory carries exactly one `task_id` (or none = global), and the
Claude Code hooks derive it from the **folder name** of the session's cwd
(`hooks._resolve_task_id`). That breaks down in four ways:

1. **Folder names are local.** `~/work/infra` and `~/src/infra` collide; the
   same repo cloned under two names splits. Once memstrand is a central store
   shared by several machines and agents, a bare folder name is meaningless.
2. **cwd is the wrong signal.** A session started in `~/.claude` that reads
   `~/projects/memstrand` by path files everything it learns under `.claude`.
3. **One scope per fact is too few.** A fact about how `pa-devops-agent`
   deploys `memstrand` is relevant in both repos, and only one can hold it.
4. **The global/task call is silent.** The extractor decides, and when it's
   unsure it guesses. Users have no way to see or fix the call except
   `forget` + `remember`.

## Goals

- A repo has one stable identity everywhere: its normalized git remote URL,
  falling back to a machine-qualified folder path.
- A memory can belong to **any number** of scopes, or be global.
- Scopes are attributed from what the session actually touched, not just cwd.
- Ambiguous scope calls are deferred to the user, asked at a natural moment.
- Every memory records which agent/machine wrote it.
- Existing stores migrate automatically; nothing is lost.

Non-goals (v1): per-user scoping / multi-tenant auth (README roadmap),
monorepo sub-path scopes, cross-machine path resolution on the server.

---

## 1. Scope identity

A **scope key** is a string. Three kinds:

| Kind | Example | When |
|---|---|---|
| git | `github.com/memstrand/memstrand` | path is inside a git repo with a remote |
| local | `local:M-TXWV62R02T:/Users/ckunaparaju/.claude` | no git repo, or a repo with no remote |
| named | `playwright-test`, `notes` | explicit `--task` / `MEMSTRAND_TASK` / `x-memstrand-task` (today's behavior, unchanged) |

Global is the reserved member `""` of the scope set (see §2).

### Normalizing a git remote

`resolve_scope(path) -> str` in a new `memstrand/scopes.py`:

1. Find the git root (existing `_git_root` walk).
2. `git -C <root> remote get-url origin`; if no `origin`, the first of
   `git remote`. (Handles worktrees and submodules, where `.git` is a file.)
3. Normalize:
   - `git@host:org/repo(.git)` and `ssh://git@host[:port]/org/repo` → `host/org/repo`
   - `https://[user[:token]@]host/org/repo(.git)` → `host/org/repo`
     (credentials are **always** stripped — never persisted)
   - drop trailing `/` and `.git`; lowercase the host
   - path case preserved by default; per-host case-folding rule, on for
     `github.com` (case-insensitive there), off elsewhere (GitLab and others
     are case-sensitive, so folding could merge distinct repos)
4. No root, or no remote → `local:<hostname>:<absolute path of root or cwd>`.

Results are cached per git root in `{home}/scope_cache.json`, keyed on the
root plus the mtime of its `.git/config`, so the hooks don't fork `git` on
every prompt.

Forks: `origin` of a fork is the fork, so a fork's memories stay separate from
upstream. An `upstream` override (`MEMSTRAND_SCOPE_REMOTE=upstream`) is a
cheap follow-up if wanted.

### Short names (aliases)

Nobody types `github.com/memstrand/memstrand`. **(impl)** There's no alias
table: short names (the last path segment, e.g. `memstrand`, `.claude`) are
computed from the scopes the store already knows (`Memory.scopes_summary`),
so they can never drift out of sync. Everywhere a user passes a
scope (`--repo`, REST, MCP), a short name resolves to its full key, or
fails with a "which one?" list if it's ambiguous. Context blocks, the web
UI, and `list` output show short names.

---

## 2. Multi-scope memories

### Model

`MemoryAtom.task_id: str | None` → `MemoryAtom.scopes: list[str]`.

**(impl)** Global is a *member* of the set, `GLOBAL_SCOPE = ""` (the value
every store already used for "no task"), not the empty set:

- `[""]` = global (visible everywhere). An atom always has ≥1 scope.
- `["", "notes"]` = global, *and* contributed by `notes` (matters for removal).
- `["github.com/memstrand/memstrand", "gitlab.corp/devops/pa-devops-agent"]` =
  visible in a session whose scope set intersects it.

Why: merging becomes plain set union with no "global absorbs" special
case, and removing a scope can never silently make an atom global. An atom
whose set becomes empty is **deleted**. This is what keeps ingest's
guarantee: re-ingesting one task never touches another task's (or the
global) claim on the same content.

`MemoryAtom.task_id` survives as a read-only legacy property (None if
global, else the first scope), and `task_id=` is still accepted as a
constructor/API keyword meaning a one-element set.

The query side generalizes the same way: `ScopedQuery.task_id` →
`ScopedQuery.scopes: list[str]` (the session's active scopes, §3.3).

**The one visibility rule** (`Memory._visible`, and its mirrors in every
store):

> an atom is visible to a query if `"" ∈ atom.scopes` **or**
> `atom.scopes ∩ query.scopes ≠ ∅` — and it's unexpired, and the host rule
> (unchanged) passes. (`schema.scopes_visible`)

`host` stays a separate scope dimension with today's semantics.

### Storage changes

**(impl)** One encoding everywhere, no new tables or columns: the scope set
is stored as `|a|b|` (`schema.encode_scopes`; global member sorts first, so
`["", "a"]` → `||a|` and global-only → `||`). Membership is a substring
test for `|key|`, which is exact because `|` is rejected in keys. Legacy
plain values (`"x"` / `""`) decode as one-element sets, and every filter
also matches them, so an un-migrated row is never invisible.

| Store | Today | v2 (impl) |
|---|---|---|
| Qdrant payload | `task_id: ""` or `"x"`, `MatchAny([q, ""])` | `scopes: ["", "a"]` list payload (keyword index); filter `should[scopes ∈ q+[""], task_id ∈ q+[""]]` — the second arm matches pre-v2 points until rewritten |
| SQLite graph | `atoms.task_id` column | same column, holds the encoded set (name kept for on-disk compatibility); `list_atoms(task_id=k)` = membership via `instr` |
| Kuzu graph | `task_id STRING` | same, via `contains()` — **kept**, not deprecated (cheap, and its parity tests run) |
| FTS5 keyword | `task_id UNINDEXED` | same column, encoded set; scope clause ORs one `instr` per query scope plus the global member |
| Session turns | `turns.task_id` | unchanged in Phase 1; `turns.scopes` arrives with Phase 2 attribution (§3.1) |

### Atom IDs: content-only, scopes merge

`_atom_id(content, task_id, host)` → `_atom_id(content, host, source)`.

**(impl, from review)** `source` is part of the key: an atom has ONE
source, so identical text ingested from two different files stays two atoms
— otherwise the later file would overwrite the earlier one's provenance,
leaking its path into the other scope and breaking `forget_source` for the
first file. Sourceless atoms (distilled facts, CLI/MCP `remember`) keep the
exact pre-v2 global key, so they merge across repos as intended.

**(impl, from review)** A merge that *widens* an atom into new scopes only
adds those scopes: existing provenance/category/session/agent/expiry are
kept and confidence is the max, so a weaker write from one repo can't
downgrade what another repo established (e.g. strip a standing
instruction's trusted provenance). Re-remembering within scopes the atom
already has is an in-place update, as before.

Re-remembering the same content under another scope **widens** the existing
atom's scopes (set union) instead of creating a sibling. That makes facts
that keep coming up in several repos turn multi-scope on their own.
**(impl)** Union is literal: remembering a task-scoped fact again as global
adds the global member. Global is only ever requested explicitly — a
`remember` with no scope, `scope --global`, or a distilled candidate that
passed the user-corroboration trust boundary (§3.2). Distillation of a
fact already stored in other scopes widens it (`scope_widened` in the
report) instead of skipping it as a duplicate.

### Supersede / contradict across scopes

A correction learned in repo B must not rewrite repo A's truth.
**(impl)** A `supersedes` edge hides its target in *every* scope, so distill
only writes one when the correction covers all of the old atom's scopes (or
is itself global). Otherwise it **splits**: the overlapping scopes are
removed from the old atom (it stays current in its other scopes) and the
two are linked with a `contradicts` edge whose reason records
"superseded in B; still current in A". A task-scoped correction never
retires a global fact — it can't cover "everywhere" — so that case is
linked as a contradiction and both surface together.

---

## 3. Attribution: which scopes did this session touch?

### 3.1 Code finds candidates (deterministic)

The Stop/PreCompact hook today keeps only `{"type": "text"}` blocks
(`hooks._extract_text`) and throws away tool calls — where the repo signal
is. v2 also walks `tool_use` blocks and collects paths from:

- `Read` / `Edit` / `Write` / `NotebookEdit`: `file_path`
- `Grep` / `Glob`: `path`
- `Bash`: `cd <dir>` targets and absolute or `~` paths in `command`
  (best-effort, shlex tokenization)
- user prompt text: absolute or `~` paths

Only paths that exist **and fall inside a git repo** are kept **(impl)** —
`ls /tmp` mustn't make `/tmp` a candidate. Each resolves via
`ScopeResolver`. Tool calls usually sit in assistant entries with no text,
so their paths carry forward onto the next logged turn (leftovers go onto
the last one). Each logged turn records the scopes it touched other than
the session's own (`turns.scopes`, added in place to existing DBs).
The session gets:

- **primary scope** = the cwd's scope (today's signal, now URL-based)
- **candidate scopes** = primary ∪ every scope touched in the session

### 3.2 The model picks from the candidates

The extraction prompt receives the candidates as a numbered list with short
names, plus per-turn annotations like `[touched: memstrand]`. Each item's
`"scope"` becomes one of:

- `"global"`
- `["memstrand", "pa-devops-agent"]` — any subset of the candidates
- `"ambiguous"` — with the candidates it was torn between (§4)

Code maps names back to keys and **drops any name not in the candidate
list**, so the model can't invent or misspell a scope. **(impl)** A bare
string name is accepted as a one-element list (models do this). `"task"`,
an omitted scope, or a list that filters down to nothing all mean the
session's own scope — per-fact turn provenance isn't tracked, so "the
turns the fact came from" isn't knowable. When the session touched no
other repo, extraction is called exactly as before (no candidate list).
Labels are short names, falling back to full keys where two collide.
The prompt also states that facts about a specific repo or folder are
never global, even the user's own folder (live testing showed
"hook scripts live in ~/.claude/hooks" otherwise came back global).

The existing global trust boundary (`distill._enforce_global_trust_boundary`)
is unchanged: global still requires corroboration by the user-turns-only
pass. **(impl)** A failed check still downgrades to the session's own scope
(with provenance capped at `inferred`). Live testing showed genuine user
preferences failing corroboration on paraphrase ("always want answers in
bullet points" → "prefers answers formatted as bullet points"), so Phase 3
queues downgraded candidates for review instead of silently scoping them.

### 3.3 The read side uses the same signal

`hook claude-code-context` (UserPromptSubmit) queries with
`scopes = primary ∪ scopes of paths in the prompt` (**(impl)** `auto` mode
only; git repos only, same harvesting as §3.1). So "read
~/projects/memstrand and …" from `~/.claude` injects memstrand's memories
alongside `.claude`'s and the global ones. The `/v1` proxy and MCP accept
an explicit scope list.

---

## 4. Ambiguous scope → ask the user

Distillation runs in the background after Claude finishes; nothing can ask
the user at that moment. So it's a queue:

1. **Model is unsure** → it returns `"ambiguous"` with its candidate scopes.
2. **Stored safely**: the atom is saved with those candidate scopes (never
   global) and `scope_status = "needs_review"`. Hiding a memory from another
   repo costs little; leaking it globally is the bug memstrand exists to
   prevent.
3. **Asked at a natural point**: on the next prompt, the context hook emits,
   *separately from and before* the "Memory (memstrand)" block (whose
   disclaimer marks its contents as non-instructions):
   ```
   ## memstrand: scope check (2)
   Ask the user once whether each of these applies everywhere or only to the listed repos,
   then run `memstrand scope <id> --global` or `memstrand scope <id> --set <repo> [...]`:
   - mem:…  "Always deploy to staging before prod"   (repos: pa-devops-agent)
   - mem:…  "Terraform plans are summarized in tables" (repos: pa-devops-agent, memstrand)
   ```
   At most 3 items per prompt, and each item is shown at most once per
   session. An item nobody resolves within N sessions (default 5) is promoted
   to `scope_status = "inferred"` and stops asking.
4. **Resolved** by the new `scope` command (§6). That sets
   `scope_status = "confirmed"`, and later extractions won't re-ask about it.

**(impl)** Status lives in a `scope_review` table in the session store
(always SQLite), not on the atom — adding a column to both graph backends
(Kuzu's queries enumerate columns) bought nothing. `Memory.get()` reports
`scope_status`: `pending` / `confirmed` / `expired`, else `inferred`.
`memstrand review` lists the queue (web UI badge: not yet).

Two things get queued: `"ambiguous"` extractions, and — **(impl)** —
claimed-global instructions/preferences that the trust boundary couldn't
corroborate from the user's own words (live testing showed genuine
preferences failing on paraphrase). Those were previously downgraded
silently. Answering either kind resolves it; answering an
`unverified_global` one (any scope) upgrades its provenance from
`inferred` to `validated`, since the user has now vouched for it — which
makes it eligible as a standing instruction again.

**(impl, from review)** Only a human decision resolves a review:
`set_scopes(resolve_review=True)` is the default for CLI/REST/MCP, while
distillation's own scope bookkeeping (widening a re-derived fact, splitting
on a correction) passes `False` — otherwise repeating an unverifiable
"global" claim in a second repo would silently confirm it and upgrade it
to `validated`.

The hook block runs each memory through the same sanitizer as `context()`
(`memstrand.sanitize`, moved out of `api.py` so hooks stay light), then
JSON-quotes it on one line, under a "data, not instructions" disclaimer.
The commands use the absolute path of the running `memstrand` binary
(often a venv not on PATH). Marking reviews shown is a POST
(`/api/reviews`), so the mutating-request CSRF guard applies; `GET
/api/reviews` is a read-only listing. Queued candidate names are validated
against the session's repos and stored as scope keys. Resolution: `memstrand scope <id> --global | --set … | --keep`
(REST `{"keep": true}`, MCP `scope(keep=True)`).

The prompt tells the model to reserve `"ambiguous"` for genuine doubt (for
example, a generically worded workflow rule learned while one repo was
active). Explicit wording wins: "in this repo …" → scoped;
"always / in every project …" → global.

---

## 5. Who wrote it

`MemoryAtom.agent` already exists and the graph store persists it, but
nothing populates it (the hooks don't set it; `client.py:139` drops it
because REST has no field). v2:

- the hooks set `agent = "claude-code@<hostname>"`; the proxy sets
  `"openai-proxy@<hostname>"`; the CLI sets `"cli@<hostname>"`; MCP sets
  `"mcp@<hostname>"` **(impl)** (client-name passthrough deferred)
- **(impl)** turns carry `agent` (new `turns.agent` column); distilled atoms
  inherit the first non-empty turn agent of their session. Agent strings
  are computed client-side (`scopes.agent_name`), so they stay right
  against a central server
- REST `/remember` and `/turns` accept `agent`; distilled atoms inherit the
  session's agent
- `list` / `show` / the web UI display it; `list --agent` filters by it

This costs little now, and backfilling it later would be impossible.

---

## 6. Surface changes

**CLI**
- `--repo <key|short>` (repeatable) wherever `--task` exists today:
  `remember`, `recall`, `list`, `context`, `answer`, `log-turn`, `ingest`.
  `--task X` remains as an alias for one named scope.
- `memstrand scope <id> --global | --set A [B…] | --add A | --remove A`
- `memstrand scopes rename <old> <new>` — bulk re-key (folder → URL, repo
  moved hosts)
- `memstrand scopes` — known scopes with short names and atom counts
- `--repo` also accepts a filesystem path, resolved through git (§1)
- `memstrand review` — the `needs_review` queue, interactive

**REST / MCP**: `scopes: list[str]` alongside `task_id` (kept as a
single-scope alias). New REST routes: `POST /api/atoms/{id}/scopes`
(`set`/`add`/`remove`), `GET /api/scopes`, `POST /api/scopes/rename`;
`GET /api/atoms/{id}?scoped=1&scope=a&scope=b`. MCP gains a `scope` tool.
`x-memstrand-scopes` header (comma-separated) for the proxy.

**Hooks**: `--task-from cwd` becomes `--scope-from {auto,cwd,none,env}`.
`auto` (the new default) = §3; `cwd` = primary scope only.

**Export**: `FORMAT_VERSION` 0.2 → 0.3, with `scopes`, `scope_status` and
`agent`. Import accepts 0.2 (`task_id` → `scopes=[task_id]`).

---

## 7. Migration

Runs once, automatically, on first open of a pre-v2 store, guarded by a
`scoping_version` row in a new session-store `meta` table **(impl)**. A
backup is written first (`backups/pre-scoping-v2-<ts>.jsonl`) whenever
anything needs to move.

1. `scopes = [task_id]` (or `[""]` when global) — **(impl)** no rewrite
   needed: legacy values decode as one-element sets on read.
   (`scope_status` arrives with Phase 3.)
2. Re-key IDs to content-only. **(impl)** Legacy *global, sourceless*
   atoms hashed an empty task, which equals the new key, so only
   task-scoped or sourced atoms move. Atoms with identical content (and host) in
   different tasks merge: scopes unioned, the highest confidence kept,
   access/FSRS stats taken from the most-recently-accessed atom, and edges
   rewritten to the surviving ID. The provenance log records every merge.
3. Qdrant: moved points are re-upserted under the new id with the stored
   vector (`VectorStore.get_vector`). No re-embedding. The full old→new id
   map is built before any edge is re-pointed, so an edge between two
   atoms that both move is preserved. Provenance history is copied forward
   (`SessionStore.copy_history`; the ledger stays append-only).
4. FTS5 rows for moved atoms are rewritten; others keep legacy values,
   which the scope clause still matches.
5. Old folder-name keys stay as named scopes. `memstrand scopes rename`
   converts them. *(Deferred: `memstrand scopes --suggest`, proposing
   renames by resolving known folder names under `$HOME` to remote URLs.)*
6. **(impl)** Importing a pre-0.3 export runs the same re-key step after the
   file is loaded; importing an atom that already exists unions its scopes.

Since the author's real store is still empty, landing this **before** the
launchd/hooks rollout means nothing needs migrating in practice. The
migration exists for other users.

---

## 8. Delivery plan

Each phase is a self-contained PR with tests; the suite stays green between
phases.

1. **Scopes core** — `scopes.py` (resolve/normalize/alias), the
   `MemoryAtom`/`ScopedQuery` schema change, the visibility rule, all four
   stores, content-only IDs + union/split rules, `scope`/`scopes` CLI,
   migration, export 0.3. *(largest)*
2. **Hook attribution** — tool-call path harvesting, per-turn scopes,
   candidate list into the extraction prompt, name → key validation,
   `--scope-from auto`.
3. **Review queue** — `scope_status`, `"ambiguous"` handling, the context-hook
   scope-check block, `memstrand review`, a web UI badge.
4. **Read-side scopes + agent** — prompt-path scopes in the context hook,
   proxy/MCP scope lists, populating and exposing `agent`.

Test focus: the visibility rule held identical across all stores (one
parametrized test per store, including global, multi-scope, disjoint, and
graph-neighbor-bridge cases); normalization table tests (ssh, https+token,
ports, case); the merge/split invariants; migration round-trips on a
fixture 0.2 store; attribution from a recorded transcript fixture with
cross-repo reads.

## Decisions (resolved open questions)

1. **Kuzu** — kept and ported (encoded column + `contains()`); its parity
   tests cover multi-scope.
2. **Naming** — "task" stays as a permanent alias (`--task`,
   `MEMSTRAND_TASK`, `task_id=`); docs say "scope".
3. **Forks** — `origin` (fork-separate) by default; no upstream merge in v1.
4. **Case-folding** — only `github.com` folds path case.
5. **Central server** — `local:` keys are allowed for now; revisit when
   multi-user scoping lands.
