"""memstrand CLI.

Every data-touching command goes through `client.get_memory()`, which
prefers the daemon (`memstrand serve`) and only opens a local `Memory`
when nothing is listening — see client.py for why that order matters.
Expected errors (daemon down, no such memory, bad export line, ...) are
printed as `error: ...` and exit 1; anything unexpected still raises, so
a real bug doesn't get silently swallowed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import tempfile
import webbrowser
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .client import get_memory, normalize_atom, probe_daemon
from .config import ensure_private_dir, get_settings, secure_file
from .schema import Category, Provenance

# RemoteError subclasses RuntimeError; KeyError covers Memory.update()'s "no
# such atom"; OSError/FileNotFoundError cover bad paths for export/import/ingest.
_EXPECTED_ERRORS = (RuntimeError, KeyError, ValueError, OSError)


class _CliError(RuntimeError):
    """An expected, user-facing CLI error: printed without a traceback."""


@contextmanager
def _memory_session():
    mem = get_memory()
    try:
        yield mem
    finally:
        mem.close()


def _mask(value: str | None) -> str:
    """Show just enough to confirm a secret is set, never the value itself."""
    if not value:
        return "(none)"
    return (value[:3] + "…") if len(value) > 3 else "***"


def _read_content_arg(content: str) -> str:
    return sys.stdin.read() if content == "-" else content


def _edit_in_editor(current: str) -> str:
    editor = os.environ.get("EDITOR", "vi")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp.write(current)
        tmp_path = tmp.name
    try:
        subprocess.run([editor, tmp_path], check=True)
        return Path(tmp_path).read_text()
    finally:
        os.unlink(tmp_path)


def _read_stdin_json() -> dict:
    try:
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except Exception:  # noqa: BLE001 - hook stdin must never crash the CLI
        return {}


# -- serve / mcp / ui ---------------------------------------------------------------


def _cmd_serve(args) -> int:
    from . import server

    server.serve(host=args.host, port=args.port, scheduler=not args.no_scheduler)
    return 0


def _cmd_mcp(args) -> int:
    from . import mcp_server

    mcp_server.main()
    return 0


def _cmd_ui(args) -> int:
    settings = get_settings()
    status = probe_daemon(settings.server_url)
    if status == "down":
        print(
            f"warning: no memstrand daemon detected at {settings.server_url} "
            "(run `memstrand serve` first)",
            file=sys.stderr,
        )
    elif status == "rejected":
        print(
            f"warning: a memstrand daemon at {settings.server_url} rejected the health "
            "probe (wrong/missing MEMSTRAND_TOKEN) — the UI may prompt you to authenticate",
            file=sys.stderr,
        )
    webbrowser.open(settings.server_url)
    return 0


# -- remember / recall ---------------------------------------------------------------


def _cmd_remember(args) -> int:
    content = _read_content_arg(args.content)
    with _memory_session() as mem:
        atom = mem.remember(
            content,
            category=Category(args.category),
            provenance=Provenance(args.provenance),
            confidence=args.confidence,
            task_id=args.task,
            host=args.host,
            ttl_days=args.ttl_days,
            source=args.source,
        )
        print(normalize_atom(atom)["id"])
    return 0


def _cmd_recall(args) -> int:
    with _memory_session() as mem:
        categories = [Category(c) for c in args.categories] if args.categories else None
        hits = mem.recall(
            args.query,
            task_id=args.task,
            host=args.host,
            categories=categories,
            limit=args.limit,
            include_superseded=args.include_superseded,
        )
        if args.json:
            print(json.dumps(hits, indent=2))
            return 0
        if not hits:
            print("(no memories found)")
            return 0
        for h in hits:
            flag = "  (superseded)" if h.get("superseded") else ""
            print(f"{h['score']:.3f}  [{h['category']}] {h['content']}{flag}")
            scope = h.get("task_id") or "global"
            bits = [f"id={h['id']}", f"scope={scope}"]
            if h.get("source_ref"):
                bits.append(f"source={h['source_ref']}")
            print(f"          {'  '.join(bits)}")
            for c in h.get("contradictions") or []:
                print(f"          ⚠ contradicts: {c['content']}")
    return 0


# -- list / show / edit / forget -------------------------------------------------------


def _cmd_list(args) -> int:
    with _memory_session() as mem:
        task_id = "" if args.global_only else args.task
        items = mem.list(
            category=Category(args.category) if args.category else None,
            task_id=task_id,
            limit=args.limit,
            offset=args.offset,
        )
        if args.json:
            print(json.dumps(items, indent=2))
            return 0
        if not items:
            print("(no memories found)")
            return 0
        for a in items:
            scope = a.get("task_id") or "global"
            flag = "  (superseded)" if a.get("superseded") else ""
            print(f"{a['id']}  [{a['category']}] ({scope}) {a['content']}{flag}")
    return 0


def _cmd_show(args) -> int:
    with _memory_session() as mem:
        atom = mem.get(args.id)
        if atom is None:
            raise _CliError(f"no such memory: {args.id}")
        if args.json:
            print(json.dumps(atom, indent=2))
            return 0
        print(f"id:         {atom['id']}")
        print(f"category:   {atom['category']}")
        print(f"provenance: {atom['provenance']}")
        print(f"confidence: {atom['confidence']}")
        scope_line = atom.get("task_id") or "global"
        if atom.get("host"):
            scope_line += f" / {atom['host']}"
        print(f"scope:      {scope_line}")
        if atom.get("source"):
            source_line = atom["source"]
            if atom.get("source_ref"):
                source_line += f"  ({atom['source_ref']})"
            print(f"source:     {source_line}")
        print(f"created:    {atom['created_at']}")
        print(f"updated:    {atom['updated_at']}")
        if atom.get("superseded"):
            print("superseded: yes")
        print()
        print(atom["content"])
        neighbors = atom.get("neighbors") or []
        if neighbors:
            print("\nneighbors:")
            for n in neighbors:
                arrow = "->" if n["direction"] == "out" else "<-"
                print(f"  {arrow} {n['relation']} {n['id']}: {n['content']}")
        history = atom.get("history") or []
        if history:
            print("\nhistory:")
            for h in history:
                ts = datetime.fromtimestamp(h["ts"], UTC).isoformat()
                print(f"  {ts}  {h['action']}  {h.get('detail', '')}")
    return 0


def _cmd_edit(args) -> int:
    with _memory_session() as mem:
        content = args.content
        if content is None and sys.stdin.isatty():
            atom = mem.get(args.id)
            if atom is None:
                raise _CliError(f"no such memory: {args.id}")
            content = _edit_in_editor(atom["content"])
        atom = mem.update(
            args.id,
            content=content,
            category=Category(args.category) if args.category else None,
            confidence=args.confidence,
        )
        print(normalize_atom(atom)["id"])
    return 0


def _cmd_forget(args) -> int:
    if not args.yes:
        reply = input(f"forget {args.id}? [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            print("aborted")
            return 0
    with _memory_session() as mem:
        if not mem.forget(args.id):
            raise _CliError(f"no such memory: {args.id}")
        print(f"forgot {args.id}")
    return 0


def _cmd_forget_source(args) -> int:
    with _memory_session() as mem:
        n = mem.forget_source(args.source)
        print(f"forgot {n} memories from {args.source}")
    return 0


# -- sessions / distillation / answering -----------------------------------------------


def _cmd_log_turn(args) -> int:
    with _memory_session() as mem:
        mem.log_turn(args.session, args.role, args.content, task_id=args.task)
    return 0


def _cmd_improve(args) -> int:
    with _memory_session() as mem:
        report = mem.improve(session_ids=args.sessions, background=args.background)
        print(json.dumps(report, indent=2))
    return 0


def _cmd_answer(args) -> int:
    with _memory_session() as mem:
        print(mem.grounded_answer(args.question, task_id=args.task))
    return 0


def _cmd_context(args) -> int:
    with _memory_session() as mem:
        ctx = mem.context(args.query, task_id=args.task, limit=args.limit)
        if ctx:
            print(ctx)
    return 0


def _cmd_consolidate(args) -> int:
    with _memory_session() as mem:
        print(json.dumps(mem.consolidate(), indent=2))
    return 0


# -- ingest ---------------------------------------------------------------------------


def _cmd_ingest(args) -> int:
    settings = get_settings()
    if args.watch:
        status = probe_daemon(settings.server_url)
        if status == "up":
            _watch_remote(settings.server_url, args)
        elif status == "rejected":
            raise _CliError(
                f"a memstrand daemon is running at {settings.server_url} but rejected "
                "this client — set MEMSTRAND_TOKEN to the daemon's token"
            )
        else:
            _watch_local(args)
        return 0
    with _memory_session() as mem:
        print(json.dumps(mem.ingest(args.path, task_id=args.task, force=args.force), indent=2))
    return 0


def _watch_remote(url: str, args) -> None:
    import time

    from .client import RemoteMemory

    mem = RemoteMemory(url, token=os.environ.get("MEMSTRAND_TOKEN"))
    print(f"watching {args.path} via daemon at {url} (every {args.interval}s, Ctrl-C to stop)")
    try:
        while True:
            report = mem.ingest(args.path, task_id=args.task, force=args.force)
            if report.get("files_ingested") or report.get("removed"):
                print(json.dumps(report, indent=2))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        mem.close()


def _watch_local(args) -> None:
    from . import ingest as ingest_module
    from .api import Memory

    with Memory() as mem:
        print(f"watching {args.path} locally (every {args.interval}s, Ctrl-C to stop)")
        ingest_module.watch_path(
            mem,
            args.path,
            task_id=args.task,
            interval=args.interval,
            on_report=lambda r: print(json.dumps(r, indent=2)),
        )


# -- export / import / backup / restore ------------------------------------------------


def _cmd_export(args) -> int:
    with _memory_session() as mem:
        text = mem.export(args.fmt)
    if args.output:
        Path(args.output).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


def _cmd_import(args) -> int:
    text = sys.stdin.read() if args.file == "-" else Path(args.file).read_text()
    with _memory_session() as mem:
        print(json.dumps(mem.import_jsonl(text), indent=2))
    return 0


def _cmd_backup(args) -> int:
    settings = get_settings()
    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_path = settings.home / "backups" / f"memstrand-{ts}.jsonl"
    # Backups contain the full, unencrypted memory export -- the backups
    # dir (if memstrand itself creates it) and each backup file are
    # private, 0700/0600, exactly like the rest of MEMSTRAND_HOME.
    ensure_private_dir(out_path.parent)
    with _memory_session() as mem:
        text = mem.export("jsonl")
    if str(out_path).endswith(".gz"):
        with gzip.open(out_path, "wt") as f:
            f.write(text)
    else:
        out_path.write_text(text)
    secure_file(out_path)
    print(str(out_path))
    return 0


def _cmd_restore(args) -> int:
    path = Path(args.file)
    if not path.exists():
        raise _CliError(f"no such file: {path}")
    if not args.yes:
        reply = input(f"restore {path} into the current memory store? This merges, it does not replace. [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            print("aborted")
            return 0
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as f:
            text = f.read()
    else:
        text = path.read_text()
    with _memory_session() as mem:
        print(json.dumps(mem.import_jsonl(text), indent=2))
    return 0


# -- stats / config ---------------------------------------------------------------------


def _cmd_stats(args) -> int:
    with _memory_session() as mem:
        s = mem.stats()
    if args.json:
        print(json.dumps(s, indent=2))
        return 0
    print(f"atoms:    {s.get('atoms', 0)}")
    print(f"edges:    {s.get('edges', 0)}")
    print(f"sessions: {s.get('sessions', 0)} ({s.get('undistilled_sessions', 0)} undistilled)")
    by_category = s.get("by_category") or {}
    if by_category:
        print("by category:")
        for cat, n in sorted(by_category.items()):
            print(f"  {cat}: {n}")
    by_task = s.get("by_task") or {}
    if by_task:
        print("by task:")
        for task, n in sorted(by_task.items()):
            print(f"  {task}: {n}")
    return 0


def _cmd_config(args) -> int:
    settings = get_settings()
    rows = [
        ("home", str(settings.home)),
        ("server_url", settings.server_url),
        ("daemon", probe_daemon(settings.server_url)),
        ("default_task", settings.default_task or "(none)"),
        ("graph_backend", settings.graph_backend),
        ("embed_backend", settings.embed_backend),
        ("embed_model", settings.embed_model or "(default)"),
        ("embed_base_url", settings.embed_base_url or "(none)"),
        ("llm_base_url", settings.llm_base_url),
        ("llm_model", settings.llm_model),
        ("llm_api_key", _mask(settings.llm_api_key)),
        ("llm_api_key_command", settings.llm_api_key_command or "(none)"),
        ("qdrant_url", settings.qdrant_url or "(embedded)"),
        ("rerank_model", settings.rerank_model or "(off)"),
        ("consolidate_hours", str(settings.consolidate_hours)),
        ("token", _mask(os.environ.get("MEMSTRAND_TOKEN"))),
    ]
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"{k:<{width}}  {v}")
    return 0


# -- Claude Code hooks --------------------------------------------------------------------


def _cmd_hook_claude_code(args) -> int:
    """Stop/SessionEnd/PreCompact body. Must exit 0 and print nothing no
    matter what — even `get_memory()` failing must not surface here.

    Probes the daemon before doing anything else: if it's up but rejects
    this client (missing/wrong MEMSTRAND_TOKEN), opening a local `Memory`
    instead would crash on Qdrant's on-disk lock, so this skips the turn
    entirely (logged, not lost — the next successful run picks up
    everything new since the last recorded transcript line)."""
    from . import hooks

    settings = get_settings()
    if probe_daemon(settings.server_url) == "rejected":
        hooks.log_error(
            "daemon requires MEMSTRAND_TOKEN which this hook does not have — "
            "turns NOT logged; they will be picked up on the next run once fixed"
        )
        return 0

    mem = None
    try:
        stdin_json = _read_stdin_json()
        mem = get_memory()
        hooks.claude_code_hook(stdin_json, mem, task_from=args.task_from, improve=not args.no_improve)
    except Exception as exc:  # noqa: BLE001 - a hook must never fail the user's session
        hooks.log_error(f"cli hook claude-code: {exc}")
    finally:
        if mem is not None:
            try:
                mem.close()
            except Exception:  # noqa: BLE001, S110
                pass
    return 0


def _cmd_hook_claude_code_context(args) -> int:
    """UserPromptSubmit/SessionStart body. Only ever queries memory when
    the daemon is already up (see hooks.py docstring) so a cold local
    `Memory()` (fastembed load, disk stores) never blocks a user prompt."""
    from . import hooks

    ctx = ""
    mem = None
    try:
        settings = get_settings()
        if probe_daemon(settings.server_url) == "up":
            stdin_json = _read_stdin_json()
            mem = get_memory()
            ctx = hooks.claude_code_context_hook(
                stdin_json, mem, task_from=args.task_from, limit=args.limit
            )
    except Exception as exc:  # noqa: BLE001 - a hook must never fail the user's session
        hooks.log_error(f"cli hook claude-code-context: {exc}")
    finally:
        if mem is not None:
            try:
                mem.close()
            except Exception:  # noqa: BLE001, S110
                pass
    if ctx:
        print(ctx)
    return 0


def _cmd_hook_print_config(args) -> int:
    from .hooks import CLAUDE_CODE_SETTINGS_SNIPPET

    print(CLAUDE_CODE_SETTINGS_SNIPPET)
    return 0


# -- argument parser ------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memstrand", description="local-first typed memory for AI agents")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="run the memstrand daemon (REST API)")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--no-scheduler", action="store_true", help="disable the background consolidation scheduler")
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("mcp", help="run the MCP stdio server")
    p.set_defaults(func=_cmd_mcp)

    p = sub.add_parser("ui", help="open the memstrand web UI in a browser")
    p.set_defaults(func=_cmd_ui)

    p = sub.add_parser("remember", help="store one memory")
    p.add_argument("content", help='memory text, or "-" to read from stdin')
    p.add_argument("--category", default="fact", choices=[c.value for c in Category])
    p.add_argument("--provenance", default="explicit_statement", choices=[v.value for v in Provenance])
    p.add_argument("--confidence", type=float, default=0.8)
    p.add_argument("--task", default=None)
    p.add_argument("--host", default=None)
    p.add_argument("--ttl-days", type=float, default=None)
    p.add_argument("--source", default=None)
    p.set_defaults(func=_cmd_remember)

    p = sub.add_parser("recall", help="scoped hybrid recall")
    p.add_argument("query")
    p.add_argument("--task", default=None)
    p.add_argument("--host", default=None)
    p.add_argument(
        "--category", dest="categories", action="append", default=None,
        choices=[c.value for c in Category], help="repeatable",
    )
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--all", dest="include_superseded", action="store_true", help="include superseded memories")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_recall)

    p = sub.add_parser("list", help="list memories")
    p.add_argument("--category", default=None, choices=[c.value for c in Category])
    p.add_argument("--task", default=None)
    p.add_argument("--global", dest="global_only", action="store_true", help="only global (task-less) memories")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("show", help="show one memory in detail")
    p.add_argument("id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_show)

    p = sub.add_parser("edit", help="edit a memory (a changed content supersedes the old atom)")
    p.add_argument("id")
    p.add_argument("--content", default=None)
    p.add_argument("--category", default=None, choices=[c.value for c in Category])
    p.add_argument("--confidence", type=float, default=None)
    p.set_defaults(func=_cmd_edit)

    p = sub.add_parser("forget", help="delete a memory")
    p.add_argument("id")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=_cmd_forget)

    p = sub.add_parser("forget-source", help="delete every memory ingested from a source")
    p.add_argument("source")
    p.set_defaults(func=_cmd_forget_source)

    p = sub.add_parser("log-turn", help="append one raw session turn")
    p.add_argument("session")
    p.add_argument("role")
    p.add_argument("content")
    p.add_argument("--task", default=None)
    p.set_defaults(func=_cmd_log_turn)

    p = sub.add_parser("improve", help="distill raw sessions into typed memory")
    p.add_argument("--session", dest="sessions", action="append", default=None, help="repeatable")
    p.add_argument("--background", action="store_true")
    p.set_defaults(func=_cmd_improve)

    p = sub.add_parser("answer", help="answer a question using only stored memories")
    p.add_argument("question")
    p.add_argument("--task", default=None)
    p.set_defaults(func=_cmd_answer)

    p = sub.add_parser("context", help="print the context block memstrand would inject into a prompt")
    p.add_argument("query", nargs="?", default=None)
    p.add_argument("--task", default=None)
    p.add_argument("--limit", type=int, default=8)
    p.set_defaults(func=_cmd_context)

    p = sub.add_parser("consolidate", help="expire, decay, dedup, replay, and distill")
    p.set_defaults(func=_cmd_consolidate)

    p = sub.add_parser("ingest", help="ingest a file or folder of notes")
    p.add_argument("path")
    p.add_argument("--task", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--interval", type=float, default=5.0)
    p.set_defaults(func=_cmd_ingest)

    p = sub.add_parser("export", help="export every memory")
    p.add_argument("--format", dest="fmt", default="jsonl", choices=["jsonl", "markdown"])
    p.add_argument("-o", "--output", default=None)
    p.set_defaults(func=_cmd_export)

    p = sub.add_parser("import", help="import a jsonl export")
    p.add_argument("file", help='jsonl file, or "-" for stdin')
    p.set_defaults(func=_cmd_import)

    p = sub.add_parser("backup", help="export to a timestamped backup file")
    p.add_argument("-o", "--output", default=None, help="defaults to {home}/backups/memstrand-<ts>.jsonl")
    p.set_defaults(func=_cmd_backup)

    p = sub.add_parser("restore", help="restore (merge) from a backup file")
    p.add_argument("file")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=_cmd_restore)

    p = sub.add_parser("stats", help="show memory statistics")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_stats)

    p = sub.add_parser("config", help="show effective configuration")
    p.set_defaults(func=_cmd_config)

    p = sub.add_parser("hook", help="Claude Code integration hooks")
    hook_sub = p.add_subparsers(dest="hook_cmd", required=True)

    hp = hook_sub.add_parser("claude-code", help="Stop/SessionEnd/PreCompact hook body")
    hp.add_argument("--task-from", default="cwd", choices=["cwd", "none", "env"])
    hp.add_argument("--no-improve", action="store_true")
    hp.set_defaults(func=_cmd_hook_claude_code)

    hp = hook_sub.add_parser("claude-code-context", help="UserPromptSubmit/SessionStart hook body")
    hp.add_argument("--task-from", default="cwd", choices=["cwd", "none", "env"])
    hp.add_argument("--limit", type=int, default=8)
    hp.set_defaults(func=_cmd_hook_claude_code_context)

    hp = hook_sub.add_parser("print-config", help="print the Claude Code settings.json hooks snippet")
    hp.set_defaults(func=_cmd_hook_print_config)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args) or 0
    except _EXPECTED_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
