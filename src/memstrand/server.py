"""The memstrand daemon: one long-lived process owns the stores.

Embedded Qdrant and SQLite/Kuzu are single-process — the CLI, hooks,
Claude Code, Cursor, and Open WebUI can't each open their own `Memory`
without corrupting each other's state. This module exposes one shared
`Memory` over four surfaces on a single ASGI app:

  * MCP (streamable-HTTP), mounted at /mcp — the same tool surface as
    `mcp_server.py`'s stdio mode, built once via `build_mcp()`.
  * A small JSON REST API under /api, used by the web UI and the CLI.
  * The web UI itself at /.
  * An OpenAI-compatible chat-completions proxy under /v1 that injects
    memory context into every request — the piece that makes memory
    work for local LLMs that won't call tools on their own.

All blocking `Memory` calls from async handlers go through
`run_in_threadpool` so one slow embed/LLM call never stalls the event
loop for other clients.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.resources
import json
import logging
import os
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Mount, Route

from . import __version__
from .api import Memory
from .config import Settings, get_settings
from .mcp_server import build_mcp
from .pipeline.extract import LLMUnavailable
from .schema import Category, Provenance

logger = logging.getLogger("memstrand.server")

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_PROTECTED_PREFIXES = ("/api", "/mcp", "/v1")
_UPSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=600.0, pool=600.0)
_CATEGORY_VALUES = [c.value for c in Category]
_PROVENANCE_VALUES = [p.value for p in Provenance]


def _parse_category(value: str) -> Category:
    try:
        return Category(value)
    except ValueError:
        raise ValueError(f"invalid category {value!r}; must be one of {_CATEGORY_VALUES}") from None


def _origin_from_referer(referer: str) -> str | None:
    """Reduce a Referer header to just its origin (scheme://host[:port]),
    the same shape as an Origin header, so it can be checked against
    `own_origins` with the identical comparison. Returns None if the
    Referer can't be parsed into a scheme+host (treated as "no signal",
    same as a missing header)."""
    parts = urlsplit(referer)
    if not parts.scheme or not parts.netloc:
        return None
    return f"{parts.scheme}://{parts.netloc}"


def _parse_provenance(value: str) -> Provenance:
    try:
        return Provenance(value)
    except ValueError:
        raise ValueError(f"invalid provenance {value!r}; must be one of {_PROVENANCE_VALUES}") from None


async def _body(request: Request) -> dict:
    """An empty POST body is valid wherever every field is optional."""
    raw = await request.body()
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON body: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("request body must be a JSON object")  # noqa: TRY004 - bad input, not a type bug
    return data


def _json_errors(fn: Callable[[Request], Awaitable[Response]]) -> Callable[[Request], Awaitable[Response]]:
    """Uniform REST error shape: {"error": str} with the status code the
    handoff spec asks for. KeyError is reserved for "no such atom"
    (raised by `Memory.update`) everywhere else handlers validate
    required fields explicitly and raise ValueError instead."""

    @functools.wraps(fn)
    async def wrapped(request: Request) -> Response:
        try:
            return await fn(request)
        except LLMUnavailable as e:
            return JSONResponse({"error": str(e)}, status_code=503)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except KeyError as e:
            return JSONResponse({"error": f"not found: {e}"}, status_code=404)
        except Exception as e:  # last resort: never crash the daemon on a handler bug
            logger.exception("unhandled error in %s", fn.__name__)
            return JSONResponse({"error": str(e)}, status_code=500)

    return wrapped


# -- security middleware: loopback-by-default, optional bearer token, ---------
# -- and CSRF/DNS-rebinding guards for the mutating surface -------------------


class _SecurityMiddleware:
    """Pure-ASGI (not BaseHTTPMiddleware) so SSE/streamable-HTTP bodies on
    /mcp and /v1 pass through untouched instead of being buffered."""

    def __init__(self, app, settings: Settings, token: str | None):
        self.app = app
        self.token = token
        self.is_loopback = settings.server_host in _LOOPBACK_HOSTS
        self.own_origins = {
            f"http://127.0.0.1:{settings.server_port}",
            f"http://localhost:{settings.server_port}",
        }

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers") or []}
        path = scope.get("path", "")
        protected = path.startswith(_PROTECTED_PREFIXES)

        if self.is_loopback:
            hostname = headers.get("host", "").split(":")[0]
            if hostname not in _LOOPBACK_HOSTS:
                await self._reject(send, "untrusted Host header", 403)
                return

        if protected:
            if self.token is not None and headers.get("authorization") != f"Bearer {self.token}":
                await self._reject(send, "missing or invalid bearer token", 401)
                return
            method = scope.get("method", "GET")
            if method not in ("GET", "HEAD", "OPTIONS"):
                origin = headers.get("origin")
                if origin is None:
                    # Browsers that omit Origin on a same-origin request
                    # (older Safari, some fetch configurations) still send
                    # Referer; a non-browser client (CLI, curl) sends
                    # neither, so this stays fail-open for them exactly
                    # like the Origin-absent case did before.
                    referer = headers.get("referer")
                    if referer:
                        origin = _origin_from_referer(referer)
                if origin is not None and origin not in self.own_origins:
                    await self._reject(send, "cross-origin request rejected", 403)
                    return

        await self.app(scope, receive, send)

    @staticmethod
    async def _reject(send, message: str, status: int) -> None:
        body = json.dumps({"error": message}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})


# -- REST handlers --------------------------------------------------------------


async def _health(request: Request) -> Response:
    mem: Memory = request.app.state.mem
    atoms = await run_in_threadpool(mem.count)
    return JSONResponse({"ok": True, "version": __version__, "atoms": atoms})


async def _stats(request: Request) -> Response:
    mem: Memory = request.app.state.mem
    return JSONResponse(await run_in_threadpool(mem.stats))


@_json_errors
async def _remember(request: Request) -> Response:
    data = await _body(request)
    content = str(data.get("content") or "").strip()
    if not content:
        raise ValueError("content is required")
    mem: Memory = request.app.state.mem
    atom = await run_in_threadpool(
        mem.remember,
        content,
        _parse_category(data.get("category", "fact")),
        _parse_provenance(data.get("provenance", "explicit_statement")),
        float(data.get("confidence", 0.8)),
        data.get("session_id"),
        data.get("task_id"),
        data.get("host"),
        data.get("ttl_days"),
        data.get("source"),
        data.get("source_ref"),
        data.get("agent"),
    )
    return JSONResponse(Memory.atom_to_dict(atom))


@_json_errors
async def _recall(request: Request) -> Response:
    data = await _body(request)
    query = str(data.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    categories = [_parse_category(c) for c in data["categories"]] if data.get("categories") else None
    mem: Memory = request.app.state.mem
    results = await run_in_threadpool(
        mem.recall,
        query,
        data.get("session_id"),
        data.get("task_id"),
        data.get("host"),
        categories,
        int(data.get("limit", 10)),
        bool(data.get("include_superseded", False)),
    )
    return JSONResponse({"results": results})


@_json_errors
async def _atoms_list(request: Request) -> Response:
    params = request.query_params
    category = params.get("category") or None
    if category is not None:
        category = _parse_category(category).value
    # absent task_id = no filter (None); present-but-empty = global only ("")
    task_id = params.get("task_id")
    host = params.get("host") or None
    limit = int(params.get("limit", 100))
    offset = int(params.get("offset", 0))
    mem: Memory = request.app.state.mem

    def _query() -> tuple[list[dict], int]:
        all_atoms = mem.graph.list_atoms(category=category, task_id=task_id, host=host, limit=10**9, offset=0)
        total = len(all_atoms)
        page = all_atoms[offset : offset + limit]
        superseded = mem.graph.superseded_ids([a.id for a in page])
        atoms = [Memory.atom_to_dict(a, superseded=a.id in superseded) for a in page]
        return atoms, total

    atoms, total = await run_in_threadpool(_query)
    return JSONResponse({"atoms": atoms, "total": total})


async def _atom_detail(request: Request) -> Response:
    mem: Memory = request.app.state.mem
    atom_id = request.path_params["atom_id"]
    params = request.query_params
    # Default (no `scoped`) is the owner-facing browse view (CLI `show`,
    # web UI): any atom, neighbors unfiltered. Pass `?scoped=1&task_id=..`
    # (and optionally `host=..`) to get the same visibility rule recall()
    # uses — out-of-scope atoms 404, neighbors filtered to the query scope.
    scoped = params.get("scoped") in ("1", "true", "True")
    task_id = params.get("task_id")
    host = params.get("host")
    detail = await run_in_threadpool(mem.get, atom_id, task_id, host, scoped)
    if detail is None:
        return JSONResponse({"error": f"no such atom: {atom_id}"}, status_code=404)
    return JSONResponse(detail)


@_json_errors
async def _atom_update(request: Request) -> Response:
    mem: Memory = request.app.state.mem
    atom_id = request.path_params["atom_id"]
    data = await _body(request)
    category = _parse_category(data["category"]) if data.get("category") is not None else None
    atom = await run_in_threadpool(
        mem.update,
        atom_id,
        data.get("content"),
        category,
        data.get("confidence"),
    )
    return JSONResponse(Memory.atom_to_dict(atom))


async def _atom_delete(request: Request) -> Response:
    mem: Memory = request.app.state.mem
    atom_id = request.path_params["atom_id"]
    found = await run_in_threadpool(mem.forget, atom_id)
    if not found:
        return JSONResponse({"error": f"no such atom: {atom_id}"}, status_code=404)
    return JSONResponse({"forgot": atom_id})


@_json_errors
async def _turns(request: Request) -> Response:
    data = await _body(request)
    items = data.get("turns")
    if items is None:
        items = [data]
    mem: Memory = request.app.state.mem

    def _run() -> None:
        for t in items:
            session_id, role, content = t.get("session_id"), t.get("role"), t.get("content")
            if not session_id or not role or content is None:
                raise ValueError("each turn requires session_id, role, content")
            mem.log_turn(session_id, role, content, task_id=t.get("task_id"))

    await run_in_threadpool(_run)
    return JSONResponse({"ok": True})


@_json_errors
async def _improve(request: Request) -> Response:
    data = await _body(request)
    mem: Memory = request.app.state.mem
    report = await run_in_threadpool(
        mem.improve, data.get("session_ids"), bool(data.get("background", False))
    )
    return JSONResponse(report)


@_json_errors
async def _answer(request: Request) -> Response:
    data = await _body(request)
    question = str(data.get("question") or "").strip()
    if not question:
        raise ValueError("question is required")
    mem: Memory = request.app.state.mem
    ans = await run_in_threadpool(
        mem.grounded_answer, question, task_id=data.get("task_id"), host=data.get("host")
    )
    return JSONResponse({"answer": ans})


@_json_errors
async def _context(request: Request) -> Response:
    data = await _body(request)
    mem: Memory = request.app.state.mem
    ctx = await run_in_threadpool(
        mem.context,
        data.get("query"),
        data.get("task_id"),
        data.get("host"),
        int(data.get("limit", 8)),
        int(data.get("max_chars", 4000)),
    )
    return JSONResponse({"context": ctx})


@_json_errors
async def _consolidate(request: Request) -> Response:
    mem: Memory = request.app.state.mem
    return JSONResponse(await run_in_threadpool(mem.consolidate))


@_json_errors
async def _ingest(request: Request) -> Response:
    data = await _body(request)
    raw_path = data.get("path")
    if not raw_path:
        raise ValueError("path is required")
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise ValueError(f"path does not exist: {raw_path}")
    mem: Memory = request.app.state.mem
    report = await run_in_threadpool(
        mem.ingest, str(path), data.get("task_id"), bool(data.get("force", False))
    )
    return JSONResponse(report)


@_json_errors
async def _export(request: Request) -> Response:
    fmt = request.query_params.get("format", "jsonl")
    if fmt not in ("jsonl", "markdown"):
        raise ValueError(f"unknown export format: {fmt!r} (expected 'jsonl' or 'markdown')")
    mem: Memory = request.app.state.mem
    text = await run_in_threadpool(mem.export, fmt)
    media_type = "application/x-ndjson" if fmt == "jsonl" else "text/plain"
    return PlainTextResponse(text, media_type=media_type)


@_json_errors
async def _import(request: Request) -> Response:
    raw = await request.body()
    mem: Memory = request.app.state.mem
    report = await run_in_threadpool(mem.import_jsonl, raw.decode("utf-8", errors="replace"))
    return JSONResponse(report)


@_json_errors
async def _forget_source(request: Request) -> Response:
    data = await _body(request)
    source = data.get("source")
    if not source:
        raise ValueError("source is required")
    mem: Memory = request.app.state.mem
    n = await run_in_threadpool(mem.forget_source, source)
    return JSONResponse({"forgot": n})


async def _index(request: Request) -> Response:
    html = importlib.resources.files("memstrand.web").joinpath("index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


# -- OpenAI-compatible memory-injecting proxy ------------------------------------


def _last_user_text(messages: list[dict]) -> str | None:
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            joined = "\n".join(p for p in parts if p)
            return joined or None
    return None


def _inject_system_block(messages: list[dict], block: str) -> list[dict]:
    messages = list(messages)
    if messages and messages[0].get("role") == "system":
        existing = messages[0].get("content", "")
        messages[0] = {**messages[0], "content": f"{block}\n\n{existing}" if existing else block}
    else:
        messages.insert(0, {"role": "system", "content": block})
    return messages


def _sse_events(buffer: bytes) -> tuple[list[bytes], bytes]:
    """Split a growing SSE byte buffer on blank-line event boundaries.
    Returns (complete event blocks, leftover tail)."""
    events = buffer.split(b"\n\n")
    return events[:-1], events[-1]


def _extract_delta_content(event: bytes) -> str:
    text = ""
    for line in event.split(b"\n"):
        if not line.startswith(b"data:"):
            continue
        payload = line[len(b"data:") :].strip()
        if payload in (b"", b"[DONE]"):
            continue
        try:
            obj = json.loads(payload)
            text += obj["choices"][0]["delta"].get("content") or ""
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            continue
    return text


async def _v1_models(request: Request) -> Response:
    settings: Settings = request.app.state.settings
    client: httpx.AsyncClient = request.app.state.http_client
    try:
        resp = await client.get(
            f"{settings.llm_base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            timeout=_UPSTREAM_TIMEOUT,
        )
        return Response(
            resp.content, status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )
    except httpx.HTTPError:
        return JSONResponse({"data": [{"id": settings.llm_model, "object": "model"}]})


async def _v1_chat_completions(request: Request) -> Response:
    settings: Settings = request.app.state.settings
    mem: Memory = request.app.state.mem
    client: httpx.AsyncClient = request.app.state.http_client

    try:
        body = json.loads(await request.body())
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    disable = request.headers.get("x-memstrand-disable") == "1"
    task_id = request.headers.get("x-memstrand-task") or settings.default_task
    session_id = request.headers.get("x-memstrand-session") or f"proxy-{datetime.now(UTC).date().isoformat()}"
    messages = body.get("messages", [])
    user_text = _last_user_text(messages)

    if not disable:
        if user_text:
            await run_in_threadpool(mem.log_turn, session_id, "user", user_text, task_id)
        block = await run_in_threadpool(mem.context, user_text, task_id) if user_text else ""
        if block:
            body = {**body, "messages": _inject_system_block(messages, block)}

    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
    stream = bool(body.get("stream"))

    if not stream:
        try:
            resp = await client.post(url, headers=headers, json=body, timeout=_UPSTREAM_TIMEOUT)
        except httpx.HTTPError as e:
            return JSONResponse({"error": f"upstream unreachable: {e}"}, status_code=502)
        if not disable:
            try:
                reply = resp.json()["choices"][0]["message"]["content"]
                await run_in_threadpool(mem.log_turn, session_id, "assistant", reply, task_id)
            except (KeyError, IndexError, ValueError, json.JSONDecodeError):
                pass
        return Response(
            resp.content, status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )

    async def _stream_and_log():
        collected: list[str] = []
        buffer = b""
        try:
            async with client.stream("POST", url, headers=headers, json=body, timeout=_UPSTREAM_TIMEOUT) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk
                    buffer += chunk
                    events, buffer = _sse_events(buffer)
                    for event in events:
                        collected.append(_extract_delta_content(event))
        except httpx.HTTPError as e:
            yield f'data: {{"error": "upstream unreachable: {e}"}}\n\n'.encode()
        finally:
            if not disable and collected:
                await run_in_threadpool(mem.log_turn, session_id, "assistant", "".join(collected), task_id)

    return StreamingResponse(_stream_and_log(), media_type="text/event-stream")


# -- scheduler: periodic consolidate() + improve() -------------------------------


def _scheduler_loop(
    mem: Memory,
    consolidate_seconds: float,
    improve_seconds: float,
    first_delay: float,
    poll_interval: float,
    stop_event: threading.Event,
) -> None:
    if stop_event.wait(first_delay):
        return
    last_consolidate = 0.0
    last_improve = 0.0
    while not stop_event.is_set():
        now = time.monotonic()
        if now - last_consolidate >= consolidate_seconds:
            try:
                mem.consolidate()
            except Exception:  # the scheduler must never die
                logger.exception("scheduled consolidate() failed")
            last_consolidate = now
        if now - last_improve >= improve_seconds:
            try:
                if mem.sessions.undistilled_sessions():
                    mem.improve(None, background=True)
            except Exception:
                logger.exception("scheduled improve() failed")
            last_improve = now
        if stop_event.wait(poll_interval):
            return


class _McpBareDispatch:
    """Dispatches a bare "/mcp" request (no trailing slash) straight into
    the FastMCP streamable-HTTP app, bypassing `Mount`.

    `Mount("/mcp", ...)` only matches "/mcp/..." — its path regex requires
    a trailing slash before the sub-app's own path — so an exact "/mcp"
    request falls through to the outer `Router`'s redirect-slash fallback,
    a 307 to "/mcp/". That's fine for a browser but most MCP clients
    (including `claude mcp add --transport http`, which is told to point
    at plain "/mcp") never follow redirects on a POST/DELETE, so the bare
    path needs to work directly. A plain ASGI *class instance* (not a
    function) so Starlette's `Route` treats it as raw ASGI instead of
    wrapping it as `func(request) -> response`, which would buffer the
    SSE body streamable-HTTP needs."""

    def __init__(self, mcp_asgi_app):
        self.mcp_asgi_app = mcp_asgi_app

    async def __call__(self, scope, receive, send) -> None:
        scope = {**scope, "path": "/", "raw_path": b"/"}
        await self.mcp_asgi_app(scope, receive, send)


# -- app factory ------------------------------------------------------------------


def create_app(
    mem: Memory,
    settings: Settings | None = None,
    *,
    start_scheduler: bool = False,
    scheduler_first_delay: float = 60.0,
    scheduler_poll_interval: float = 1.0,
    scheduler_improve_interval: float = 600.0,
    upstream_transport: httpx.BaseTransport | None = None,
) -> Starlette:
    """Build the daemon's ASGI app around an already-constructed `Memory`
    (tests inject one with fake stores/embedder; `serve()` builds a real
    one). `start_scheduler=True` runs periodic consolidate()/improve() on
    a background thread for the life of the app."""
    settings = settings or get_settings()
    token = os.environ.get("MEMSTRAND_TOKEN")
    is_loopback = settings.server_host in _LOOPBACK_HOSTS
    if not is_loopback and not token:
        raise RuntimeError(
            f"refusing to start on non-loopback host {settings.server_host!r} "
            "without MEMSTRAND_TOKEN set"
        )

    fast_mcp = build_mcp(lambda: mem)
    mcp_asgi_app = fast_mcp.streamable_http_app()  # lazily creates the session manager
    session_manager = fast_mcp.session_manager

    stop_event = threading.Event()
    scheduler_thread: threading.Thread | None = None
    http_client = httpx.AsyncClient(transport=upstream_transport)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        nonlocal scheduler_thread
        # The mounted FastMCP app's own `lifespan=` is never invoked by
        # Starlette (Mount doesn't propagate the lifespan ASGI scope into
        # sub-apps), so the session manager is run explicitly here,
        # inside the parent app's lifespan, instead.
        async with session_manager.run():
            if start_scheduler:
                scheduler_thread = threading.Thread(
                    target=_scheduler_loop,
                    args=(
                        mem,
                        settings.consolidate_hours * 3600,
                        scheduler_improve_interval,
                        scheduler_first_delay,
                        scheduler_poll_interval,
                        stop_event,
                    ),
                    daemon=True,
                )
                scheduler_thread.start()
            try:
                yield
            finally:
                stop_event.set()
                if scheduler_thread is not None:
                    scheduler_thread.join(timeout=5)
                await http_client.aclose()

    routes = [
        Route("/api/health", _health, methods=["GET"]),
        Route("/api/stats", _stats, methods=["GET"]),
        Route("/api/remember", _remember, methods=["POST"]),
        Route("/api/recall", _recall, methods=["POST"]),
        Route("/api/atoms", _atoms_list, methods=["GET"]),
        Route("/api/atoms/{atom_id}", _atom_detail, methods=["GET"]),
        Route("/api/atoms/{atom_id}", _atom_update, methods=["PATCH"]),
        Route("/api/atoms/{atom_id}", _atom_delete, methods=["DELETE"]),
        Route("/api/turns", _turns, methods=["POST"]),
        Route("/api/improve", _improve, methods=["POST"]),
        Route("/api/answer", _answer, methods=["POST"]),
        Route("/api/context", _context, methods=["POST"]),
        Route("/api/consolidate", _consolidate, methods=["POST"]),
        Route("/api/ingest", _ingest, methods=["POST"]),
        Route("/api/export", _export, methods=["GET"]),
        Route("/api/import", _import, methods=["POST"]),
        Route("/api/forget_source", _forget_source, methods=["POST"]),
        Route("/v1/models", _v1_models, methods=["GET"]),
        Route("/v1/chat/completions", _v1_chat_completions, methods=["POST"]),
        Route("/mcp", _McpBareDispatch(mcp_asgi_app)),
        Mount("/mcp", app=mcp_asgi_app),
        Route("/", _index, methods=["GET"]),
    ]

    app = Starlette(
        routes=routes,
        lifespan=lifespan,
        middleware=[Middleware(_SecurityMiddleware, settings=settings, token=token)],
    )
    app.state.mem = mem
    app.state.settings = settings
    app.state.http_client = http_client
    return app


def serve(
    host: str | None = None,
    port: int | None = None,
    mem: Memory | None = None,
    scheduler: bool = True,
) -> None:
    """Build (or accept) a `Memory`, mount it behind `create_app()`, and
    block in uvicorn until interrupted. The daemon entry point for
    `python -m memstrand.server`, the CLI's `memstrand serve`, etc.
    `scheduler=False` disables the background consolidate()/improve() loop
    (the CLI's `memstrand serve --no-scheduler`)."""
    settings = get_settings()
    if host is not None or port is not None:
        settings = replace(
            settings,
            server_host=host or settings.server_host,
            server_port=port or settings.server_port,
        )
    mem = mem or Memory(settings=settings)
    app = create_app(mem, settings, start_scheduler=scheduler)

    url = settings.server_url
    print(f"memstrand daemon listening on {url}")
    print(f"  web UI:       {url}/")
    print(f"  REST API:     {url}/api")
    print(f"  MCP (http):   {url}/mcp")
    print(f"  OpenAI proxy: {url}/v1")

    try:
        uvicorn.run(app, host=settings.server_host, port=settings.server_port, log_level="info")
    finally:
        mem.close()


if __name__ == "__main__":
    serve()
