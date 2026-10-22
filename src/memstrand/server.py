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

Env vars read directly by this module (beyond `Settings`/`MEMSTRAND_TOKEN`):

  * `MEMSTRAND_ALLOWED_ORIGINS` — comma-separated list of extra Origin
    values trusted for the CSRF/same-origin check on mutating `/api`,
    `/mcp`, `/v1` requests, on top of the daemon's own origin(s). Each
    entry must be exactly `scheme://host[:port]` (http or https, no
    path/query/fragment, not `null`/`*`) — anything else, including a
    bare host or wildcard, makes the daemon refuse to start with a clear
    error naming the bad entry (fail closed, see `_parse_allowed_origins`).
    Useful for trusted local tools that proxy from a different port
    (e.g. the MCP inspector at `http://localhost:6274`) or for reaching
    the web UI from elsewhere on the LAN when bound to a non-loopback
    host: add that origin, e.g. `MEMSTRAND_ALLOWED_ORIGINS=http://192.168.1.5:7777`.
    Has no effect on the mandatory bearer token. Note: there is no
    same-origin-by-Host fallback — an Origin/Referer that merely matches
    the request's own Host header is NOT automatically trusted (that was
    a DNS-rebinding hole: an attacker who controls both headers could
    forge a match). Requests with neither Origin nor Referer (CLI, MCP
    clients, hooks) are unaffected by this check; the token is still
    mandatory for them.
"""

from __future__ import annotations

import contextlib
import functools
import hmac
import importlib.resources
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import uvicorn
from mcp.server.transport_security import TransportSecuritySettings
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
from .api import Memory, StoreLocked
from .config import ConfigError, Settings, get_settings, resolve_llm_api_key
from .mcp_server import build_mcp
from .pipeline.extract import LLMUnavailable
from .schema import Category, Provenance
from .scopes import agent_name

logger = logging.getLogger("memstrand.server")

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
# Hosts that mean "listen on every interface" -- there is no single "our
# own Host header" to validate against when bound this way, so the
# Host-header check below is skipped for these (the Origin/token checks
# still apply).
_WILDCARD_HOSTS = {"0.0.0.0", "::"}
_PROTECTED_PREFIXES = ("/api", "/mcp", "/v1")
_UPSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=600.0, pool=600.0)
_CATEGORY_VALUES = [c.value for c in Category]
_PROVENANCE_VALUES = [p.value for p in Provenance]
_VALID_ROLES = ("user", "assistant", "system", "tool")


def _parse_category(value: str) -> Category:
    try:
        return Category(value)
    except ValueError:
        raise ValueError(f"invalid category {value!r}; must be one of {_CATEGORY_VALUES}") from None


def _parse_scopes(value) -> list[str] | None:
    """Optional `scopes` field: a list of scope keys ("" = global). Absent/
    null = not given (callers fall back to the legacy `task_id`)."""
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError("scopes must be a list of strings")
    return value


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


def _parse_allowed_origins(raw: str) -> frozenset[str]:
    """Strictly parse `MEMSTRAND_ALLOWED_ORIGINS` into a set of normalized
    `scheme://host[:port]` origins, or raise `ValueError` naming the first
    bad entry. Called eagerly from `create_app()` (not from
    `_SecurityMiddleware.__init__`, which Starlette only instantiates
    lazily on the first request -- too late to make the daemon "refuse to
    start") so a malformed value fails fast and loud instead of silently
    widening (or being ignored and narrowing) the trust boundary.

    Each entry is stripped of surrounding whitespace and any trailing
    "/", then lowercased in its scheme+host (ports and paths are
    case-sensitive in general, but we also forbid paths here, so only
    scheme+host need normalizing). An entry is valid only if:
      * scheme is exactly "http" or "https"
      * netloc (host[:port]) is non-empty
      * there is no path (beyond the trailing "/" already stripped), no
        query string, and no fragment
      * the literal value isn't "null" (the Origin a sandboxed/opaque
        browser context sends, never a real origin to trust) or "*"
        (never a valid Origin value; accepting it would trust everything)
    """
    origins: set[str] = set()
    for entry in raw.split(","):
        candidate = entry.strip()
        if not candidate:
            continue
        normalized = candidate.removesuffix("/")
        lowered = normalized.lower()
        if lowered in ("null", "*"):
            raise ValueError(
                f"MEMSTRAND_ALLOWED_ORIGINS entry {candidate!r} is not allowed "
                "(expected a concrete scheme://host[:port] origin)"
            )
        parts = urlsplit(normalized)
        scheme = parts.scheme.lower()
        netloc = parts.netloc.lower()
        if (
            scheme not in ("http", "https")
            or not netloc
            or parts.path
            or parts.query
            or parts.fragment
        ):
            raise ValueError(
                f"MEMSTRAND_ALLOWED_ORIGINS entry {candidate!r} is not a valid origin "
                "(expected exactly scheme://host[:port], scheme http or https, "
                "no path/query/fragment)"
            )
        origins.add(f"{scheme}://{netloc}")
    return frozenset(origins)


def _parse_provenance(value: str) -> Provenance:
    try:
        return Provenance(value)
    except ValueError:
        raise ValueError(f"invalid provenance {value!r}; must be one of {_PROVENANCE_VALUES}") from None


def _parse_bounded_int(value, *, name: str, default: int, minimum: int = 0, maximum: int | None = None) -> int:
    """Parse a REST `limit`/`offset`-style parameter (query param or JSON
    body value) into a bounds-checked int, raising ValueError (-> 400) on
    anything malformed or out of range instead of letting a bad `limit`
    silently become a huge/negative query."""
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer") from None
    if parsed < minimum or (maximum is not None and parsed > maximum):
        bound = f">= {minimum}" + (f" and <= {maximum}" if maximum is not None else "")
        raise ValueError(f"{name} must be {bound}")
    return parsed


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
        except FileNotFoundError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except KeyError as e:
            return JSONResponse({"error": f"not found: {e}"}, status_code=404)
        except Exception as e:  # last resort: never crash the daemon on a handler bug
            logger.exception("unhandled error in %s", fn.__name__)
            return JSONResponse({"error": str(e)}, status_code=500)

    return wrapped


# -- security middleware: loopback-by-default, optional bearer token, ---------
# -- and CSRF/DNS-rebinding guards for the mutating surface -------------------


_SECURITY_HEADERS = (
    (b"x-frame-options", b"DENY"),
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
)
# The web UI (/) is the only route that legitimately serves its own inline
# <script>/<style> (see web/index.html) and needs `self` for same-origin
# fetches to /api; everything else (JSON/SSE/MCP) gets the tightest policy.
_CSP_INDEX = (
    b"default-src 'self'; script-src 'self' 'unsafe-inline'; "
    b"style-src 'self' 'unsafe-inline'; frame-ancestors 'none'"
)
_CSP_DEFAULT = b"frame-ancestors 'none'"


class _SecurityMiddleware:
    """Pure-ASGI (not BaseHTTPMiddleware) so SSE/streamable-HTTP bodies on
    /mcp and /v1 pass through untouched instead of being buffered."""

    def __init__(self, app, settings: Settings, token: str | None, extra_origins: frozenset[str] = frozenset()):
        self.app = app
        self.token = token
        self.is_loopback = settings.server_host in _LOOPBACK_HOSTS
        self.is_wildcard_bind = settings.server_host in _WILDCARD_HOSTS
        self.own_origins = {
            f"http://127.0.0.1:{settings.server_port}",
            f"http://localhost:{settings.server_port}",
            f"http://[::1]:{settings.server_port}",
        }
        if not self.is_loopback and not self.is_wildcard_bind:
            # Non-loopback binds are reachable from other hosts on the LAN;
            # the configured host is as much "our own origin" as localhost
            # is in loopback mode. A bearer token is mandatory to even get
            # this far (enforced in create_app()), so this isn't widening
            # the trust boundary.
            self.own_origins.add(f"http://{settings.server_host}:{settings.server_port}")
        # `extra_origins` is parsed and validated eagerly by
        # `_parse_allowed_origins()` in `create_app()` -- never re-read
        # MEMSTRAND_ALLOWED_ORIGINS here, since Starlette only constructs
        # this middleware lazily (on first request), which would be too
        # late to make a malformed value fail the daemon's startup.
        self.extra_origins = extra_origins
        # Host-header allowlist for non-loopback, non-wildcard binds (e.g.
        # MEMSTRAND_HOST=192.168.1.5): the configured bind host, plus the
        # host portion of every trusted origin (own + extra). A wildcard
        # bind (0.0.0.0/::) has no single "our own Host" to check against,
        # so it's exempt -- the Origin/token checks still apply to it.
        self._allowed_hosts = {settings.server_host} | {
            urlsplit(o).hostname for o in (self.own_origins | self.extra_origins)
        }

    def _origin_allowed(self, origin: str) -> bool:
        return origin in self.own_origins or origin in self.extra_origins

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers") or []}
        path = scope.get("path", "")
        protected = path.startswith(_PROTECTED_PREFIXES)
        send = self._wrap_send(send, path)

        if self.is_loopback:
            hostname = headers.get("host", "").split(":")[0]
            if hostname not in _LOOPBACK_HOSTS:
                await self._reject(send, "untrusted Host header", 403)
                return
        elif not self.is_wildcard_bind:
            # A specific non-loopback bind (e.g. MEMSTRAND_HOST=192.168.1.5)
            # has exactly one legitimate Host value (plus any configured
            # MEMSTRAND_ALLOWED_ORIGINS hosts) -- anything else is a
            # DNS-rebinding attempt (a public DNS name resolving to this
            # host, used to smuggle a browser request past same-origin
            # checks). A wildcard bind (0.0.0.0/::) has no single value to
            # check against, so it's exempt from this specific guard; the
            # mandatory token and the Origin check below still apply.
            hostname = headers.get("host", "").split(":")[0]
            if hostname not in self._allowed_hosts:
                await self._reject(send, "untrusted Host header", 403)
                return

        if protected:
            if self.token is not None:
                presented = headers.get("authorization", "")
                if not hmac.compare_digest(presented, f"Bearer {self.token}"):
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
                if origin is not None and not self._origin_allowed(origin):
                    await self._reject(send, "cross-origin request rejected", 403)
                    return

        await self.app(scope, receive, send)

    @staticmethod
    def _wrap_send(send, path: str):
        csp = _CSP_INDEX if path == "/" else _CSP_DEFAULT

        async def wrapped_send(message):
            if message["type"] == "http.response.start":
                response_headers = list(message.get("headers") or [])
                response_headers.extend(_SECURITY_HEADERS)
                response_headers.append((b"content-security-policy", csp))
                message = {**message, "headers": response_headers}
            await send(message)

        return wrapped_send

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
        _parse_scopes(data.get("scopes")),
        _parse_scopes(data.get("tags")),
    )
    return JSONResponse(Memory.atom_to_dict(atom))


@_json_errors
async def _recall(request: Request) -> Response:
    data = await _body(request)
    query = str(data.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    categories = [_parse_category(c) for c in data["categories"]] if data.get("categories") else None
    limit = _parse_bounded_int(data.get("limit", 10), name="limit", default=10, minimum=1, maximum=1000)
    mem: Memory = request.app.state.mem
    results = await run_in_threadpool(
        mem.recall,
        query,
        data.get("session_id"),
        data.get("task_id"),
        data.get("host"),
        categories,
        limit,
        bool(data.get("include_superseded", False)),
        scopes=_parse_scopes(data.get("scopes")),
        tags=_parse_scopes(data.get("tags")),
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
    agent = params.get("agent") or None
    tag = params.get("tag") or None
    limit = _parse_bounded_int(params.get("limit"), name="limit", default=100, minimum=0, maximum=1000)
    offset = _parse_bounded_int(params.get("offset"), name="offset", default=0, minimum=0)
    mem: Memory = request.app.state.mem

    # Contract with Memory.list_page (api.py): returns the page of atom
    # dicts (same shape as Memory.list, incl. `superseded`) plus the total
    # count matching the filters, both computed under Memory._lock in one
    # pass -- unlike the old approach here, this never touches mem.graph.*
    # directly and never materializes the full unfiltered table just to
    # count it.
    atoms, total = await run_in_threadpool(
        mem.list_page, category=category, task_id=task_id, host=host, limit=limit, offset=offset,
        agent=agent, tag=tag,
    )
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
    scopes = params.getlist("scope") or None
    detail = await run_in_threadpool(mem.get, atom_id, task_id, host, scoped, scopes)
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
            if role not in _VALID_ROLES:
                raise ValueError(f"invalid role {role!r}; must be one of {list(_VALID_ROLES)}")
            ts = t.get("ts")
            if ts is not None and (isinstance(ts, bool) or not isinstance(ts, (int, float))):
                raise ValueError("ts must be a number (epoch seconds)")
            mem.log_turn(
                session_id, role, content, task_id=t.get("task_id"),
                scopes=_parse_scopes(t.get("scopes")), agent=t.get("agent"), ts=ts,
            )

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
        mem.grounded_answer,
        question,
        task_id=data.get("task_id"),
        host=data.get("host"),
        scopes=_parse_scopes(data.get("scopes")),
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
        _parse_scopes(data.get("scopes")),
    )
    return JSONResponse({"context": ctx})


@_json_errors
async def _atom_scopes(request: Request) -> Response:
    """POST {"set": [...]?, "add": [...]?, "remove": [...]?} -> the atom."""
    mem: Memory = request.app.state.mem
    data = await _body(request)
    atom_id = request.path_params["atom_id"]
    if data.get("keep"):
        atom = await run_in_threadpool(mem.confirm_scopes, atom_id)
    else:
        atom = await run_in_threadpool(
            mem.set_scopes,
            atom_id,
            _parse_scopes(data.get("set")),
            _parse_scopes(data.get("add")),
            _parse_scopes(data.get("remove")),
        )
    return JSONResponse(Memory.atom_to_dict(atom))


@_json_errors
async def _session_turns(request: Request) -> Response:
    mem: Memory = request.app.state.mem
    limit_raw = request.query_params.get("limit")
    limit = _parse_bounded_int(limit_raw, name="limit", default=0, minimum=0) if limit_raw else None
    result = await run_in_threadpool(mem.session_turns, request.path_params["session_id"], limit)
    return JSONResponse(result)


@_json_errors
async def _session_mark_distilled(request: Request) -> Response:
    mem: Memory = request.app.state.mem
    data = await _body(request) if await request.body() else {}
    last = await run_in_threadpool(
        mem.mark_distilled, request.path_params["session_id"], bool(data.get("reset", False))
    )
    return JSONResponse({"last_turn_id": last})


@_json_errors
async def _turns_search(request: Request) -> Response:
    params = request.query_params
    query = (params.get("q") or "").strip()
    if not query:
        raise ValueError("q is required")
    limit = _parse_bounded_int(params.get("limit"), name="limit", default=20, minimum=1, maximum=500)
    mem: Memory = request.app.state.mem
    results = await run_in_threadpool(mem.search_turns, query, limit, params.get("session") or None)
    return JSONResponse({"results": results})


@_json_errors
async def _reviews(request: Request) -> Response:
    """GET ?limit=: read-only listing of pending scope reviews. POST
    {"limit", "session"}: the next reviews for that session, each returned
    once per session and counted as shown (state change => POST, so the
    mutating-request CSRF guard applies and a web page can't burn the queue)."""
    mem: Memory = request.app.state.mem
    if request.method == "GET":
        limit_raw, session = request.query_params.get("limit"), None
    else:
        data = await _body(request)
        limit_raw, session = data.get("limit"), data.get("session")
        if not isinstance(session, str) or not session:
            raise ValueError("session is required")
    limit = _parse_bounded_int(limit_raw, name="limit", default=3, minimum=1, maximum=100)
    reviews = await run_in_threadpool(mem.pending_scope_reviews, limit, session)
    return JSONResponse({"reviews": reviews})


@_json_errors
async def _atom_tags(request: Request) -> Response:
    """POST {"set": [...]?, "add": [...]?, "remove": [...]?} -> the atom."""
    mem: Memory = request.app.state.mem
    data = await _body(request)
    atom = await run_in_threadpool(
        mem.set_tags, request.path_params["atom_id"],
        _parse_scopes(data.get("set")), _parse_scopes(data.get("add")), _parse_scopes(data.get("remove")),
    )
    return JSONResponse(Memory.atom_to_dict(atom))


@_json_errors
async def _atom_related(request: Request) -> Response:
    params = request.query_params
    mem: Memory = request.app.state.mem
    min_shared = _parse_bounded_int(params.get("min_shared"), name="min_shared", default=2, minimum=1, maximum=8)
    limit = _parse_bounded_int(params.get("limit"), name="limit", default=10, minimum=1, maximum=100)
    related = await run_in_threadpool(mem.related, request.path_params["atom_id"], min_shared, limit)
    return JSONResponse({"related": related})


@_json_errors
async def _tags_list(request: Request) -> Response:
    params = request.query_params
    mem: Memory = request.app.state.mem
    limit = _parse_bounded_int(params.get("limit"), name="limit", default=100, minimum=1, maximum=1000)
    counts = await run_in_threadpool(mem.tag_counts, params.get("scope"), limit)
    return JSONResponse({"tags": [{"tag": t, "count": n} for t, n in counts]})


@_json_errors
async def _tags_rename(request: Request) -> Response:
    """POST {"old": "x", "new": "y"} renames/merges; {"old": "x", "new": null} drops."""
    data = await _body(request)
    old, new = data.get("old"), data.get("new")
    if not isinstance(old, str) or (new is not None and not isinstance(new, str)):
        raise ValueError("old must be a string; new a string or null")
    mem: Memory = request.app.state.mem
    return JSONResponse({"changed": await run_in_threadpool(mem.rename_tag, old, new)})


@_json_errors
async def _retag(request: Request) -> Response:
    data = await _body(request)
    atom_ids = _parse_scopes(data.get("atom_ids")) or []
    if not atom_ids:
        raise ValueError("atom_ids is required")
    mem: Memory = request.app.state.mem
    return JSONResponse(await run_in_threadpool(mem.retag, atom_ids, data.get("session_id") or None))


@_json_errors
async def _reindex(request: Request) -> Response:
    mem: Memory = request.app.state.mem
    return JSONResponse(await run_in_threadpool(mem.reindex))


@_json_errors
async def _scopes_list(request: Request) -> Response:
    mem: Memory = request.app.state.mem
    return JSONResponse({"scopes": await run_in_threadpool(mem.scopes_summary)})


@_json_errors
async def _scopes_rename(request: Request) -> Response:
    data = await _body(request)
    old, new = data.get("old"), data.get("new")
    if not isinstance(old, str) or not isinstance(new, str):
        raise ValueError("old and new are required strings")  # noqa: TRY004 - ValueError -> HTTP 400
    mem: Memory = request.app.state.mem
    return JSONResponse({"renamed": await run_in_threadpool(mem.rename_scope, old, new)})


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
    if "scope" in data:  # remove just this scope ("" = global) from the source's atoms
        scope = data["scope"]
        if not isinstance(scope, str):
            raise ValueError("scope must be a string")
        n = await run_in_threadpool(mem.forget_source, source, scope)
    else:
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
    api_key = await run_in_threadpool(resolve_llm_api_key, settings)
    try:
        resp = await client.get(
            f"{settings.llm_base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
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
    header_scopes = [x.strip() for x in request.headers.get("x-memstrand-scopes", "").split(",") if x.strip()]
    proxy_agent = agent_name("openai-proxy")
    session_id = request.headers.get("x-memstrand-session") or f"proxy-{datetime.now(UTC).date().isoformat()}"
    messages = body.get("messages", [])
    user_text = _last_user_text(messages)

    if not disable:
        if user_text:
            await run_in_threadpool(
                functools.partial(mem.log_turn, session_id, "user", user_text, task_id, agent=proxy_agent)
            )
        block = (
            await run_in_threadpool(functools.partial(mem.context, user_text, task_id, scopes=header_scopes or None))
            if user_text else ""
        )
        if block:
            body = {**body, "messages": _inject_system_block(messages, block)}

    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {await run_in_threadpool(resolve_llm_api_key, settings)}"}
    stream = bool(body.get("stream"))

    if not stream:
        try:
            resp = await client.post(url, headers=headers, json=body, timeout=_UPSTREAM_TIMEOUT)
        except httpx.HTTPError as e:
            return JSONResponse({"error": f"upstream unreachable: {e}"}, status_code=502)
        if not disable:
            try:
                reply = resp.json()["choices"][0]["message"]["content"]
                await run_in_threadpool(
                    functools.partial(mem.log_turn, session_id, "assistant", reply, task_id, agent=proxy_agent)
                )
            except (KeyError, IndexError, ValueError, json.JSONDecodeError):
                pass
        return Response(
            resp.content, status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )

    # Open the upstream request ourselves (instead of `async with
    # client.stream(...)`) so we can inspect the status code *before*
    # deciding whether the client gets a StreamingResponse at all. Most
    # OpenAI-compatible servers return a plain JSON error body (not SSE)
    # for a non-2xx response even when the request asked for `stream:
    # true` -- if we blindly wrapped that in a 200 text/event-stream, the
    # client would get a "stream" containing one chunk of raw JSON instead
    # of a real error it can detect.
    upstream_request = client.build_request("POST", url, headers=headers, json=body, timeout=_UPSTREAM_TIMEOUT)
    try:
        resp = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as e:
        return JSONResponse({"error": f"upstream unreachable: {e}"}, status_code=502)

    if not (200 <= resp.status_code < 300):
        error_body = await resp.aread()
        await resp.aclose()
        return Response(
            error_body, status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )

    async def _stream_and_log():
        collected: list[str] = []
        buffer = b""
        ok = False
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
                buffer += chunk
                events, buffer = _sse_events(buffer)
                for event in events:
                    collected.append(_extract_delta_content(event))
            ok = True
        except httpx.HTTPError as e:
            # Upstream died mid-stream, after we already committed to a 200
            # text/event-stream response -- the only way left to tell the
            # client something went wrong is a well-formed SSE error event
            # (json.dumps, not an f-string: `e`'s text can contain quotes
            # that would otherwise produce invalid JSON) followed by the
            # same [DONE] sentinel a normal stream ends with.
            payload = json.dumps({"error": f"upstream stream failed: {e}"})
            yield f"data: {payload}\n\n".encode()
            yield b"data: [DONE]\n\n"
        finally:
            await resp.aclose()
            if ok and not disable and collected:
                await run_in_threadpool(
                    functools.partial(
                        mem.log_turn, session_id, "assistant", "".join(collected), task_id, agent=proxy_agent
                    )
                )

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
    the MCPServer streamable-HTTP app, bypassing `Mount`.

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
    # Parsed eagerly, here, rather than inside `_SecurityMiddleware.__init__`:
    # Starlette only constructs middleware lazily (on the first request),
    # so validating there wouldn't make a malformed MEMSTRAND_ALLOWED_ORIGINS
    # fail the daemon's *startup* -- it would instead surface as a 500 (or
    # worse, pass silently) on whatever request happens to arrive first.
    # Raises ValueError (caught by `serve()`/the CLI like `ConfigError`)
    # naming the bad entry -- fail closed rather than starting with a
    # silently-narrowed or silently-widened trust boundary.
    extra_origins = _parse_allowed_origins(os.environ.get("MEMSTRAND_ALLOWED_ORIGINS", ""))

    mcp_server = build_mcp(lambda: mem)
    # streamable_http_path="/" so Mount("/mcp", ...) doesn't double the
    # path segment; passed here (not the constructor) in the v2 SDK.
    # Lazily creates the session manager.
    #
    # `host=` matters: the v2 SDK auto-enables its own DNS-rebinding
    # protection (rejecting requests whose Host header isn't in its
    # allowlist) whenever `host` is loopback -- and defaults `host` to
    # "127.0.0.1" if we don't pass it, regardless of what memstrand is
    # actually bound to. Leaving that default in place broke non-loopback
    # mode outright: a legitimate remote client's Host header (its LAN
    # IP/hostname) never matches "127.0.0.1"/"localhost"/"[::1]" and the
    # SDK rejected it with 421 before our own middleware ever saw the
    # request. So: pass the real bind host through, and for non-loopback
    # binds explicitly disable the SDK's own protection -- memstrand's
    # `_SecurityMiddleware` (Origin check) plus the mandatory
    # MEMSTRAND_TOKEN (enforced above) are the guard there instead. For
    # loopback binds we want the SDK's protection to stay on, so leave
    # `transport_security` unset and let it auto-enable.
    mcp_transport_security = (
        None if is_loopback else TransportSecuritySettings(enable_dns_rebinding_protection=False)
    )
    mcp_asgi_app = mcp_server.streamable_http_app(
        streamable_http_path="/", host=settings.server_host, transport_security=mcp_transport_security
    )
    session_manager = mcp_server.session_manager

    stop_event = threading.Event()
    scheduler_thread: threading.Thread | None = None
    http_client = httpx.AsyncClient(transport=upstream_transport)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        nonlocal scheduler_thread
        # The mounted MCPServer app's own `lifespan=` is never invoked by
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
        Route("/api/atoms/{atom_id}/scopes", _atom_scopes, methods=["POST"]),
        Route("/api/scopes", _scopes_list, methods=["GET"]),
        Route("/api/atoms/{atom_id}/tags", _atom_tags, methods=["POST"]),
        Route("/api/atoms/{atom_id}/related", _atom_related, methods=["GET"]),
        Route("/api/tags", _tags_list, methods=["GET"]),
        Route("/api/tags/rename", _tags_rename, methods=["POST"]),
        Route("/api/retag", _retag, methods=["POST"]),
        Route("/api/reindex", _reindex, methods=["POST"]),
        Route("/api/reviews", _reviews, methods=["GET", "POST"]),
        Route("/api/scopes/rename", _scopes_rename, methods=["POST"]),
        Route("/api/turns", _turns, methods=["POST"]),
        Route("/api/turns/search", _turns_search, methods=["GET"]),
        Route("/api/sessions/{session_id:path}/turns", _session_turns, methods=["GET"]),
        Route("/api/sessions/{session_id:path}/distilled", _session_mark_distilled, methods=["POST"]),
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
        middleware=[
            Middleware(_SecurityMiddleware, settings=settings, token=token, extra_origins=extra_origins)
        ],
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
    (the CLI's `memstrand serve --no-scheduler`).

    Startup failures -- `ConfigError` (a bad `MEMSTRAND_*` env var),
    `StoreLocked` (another process, almost always a second daemon, already
    holds this home's Qdrant store), or the plain `RuntimeError`/`ValueError`
    `create_app()` raises for a missing `MEMSTRAND_TOKEN` on a non-loopback
    bind or a malformed `MEMSTRAND_ALLOWED_ORIGINS` -- are caught here and
    printed as one clean `error: ...` line to stderr with exit code 1: no
    ready banner (it hasn't started), no traceback."""
    try:
        settings = get_settings()
        if host is not None or port is not None:
            settings = replace(
                settings,
                server_host=host or settings.server_host,
                server_port=port or settings.server_port,
            )
        owns_mem = mem is None
        try:
            mem = mem or Memory(settings=settings)
            app = create_app(mem, settings, start_scheduler=scheduler)
        except Exception:
            if owns_mem and mem is not None:
                with contextlib.suppress(Exception):
                    mem.close()
            raise
    except (ConfigError, StoreLocked, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

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
