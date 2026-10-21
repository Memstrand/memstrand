"""REST client over the memstrand daemon.

`RemoteMemory` mirrors the subset of `Memory`'s public API the CLI and
hooks need, so calling code never has to branch on which backend answered
it. The one shape difference: `remember`/`update` return the atom as a
plain `dict` (JSON over the wire), not a `MemoryAtom` — call
`normalize_atom()` on the result if you need one shape regardless of
backend.

`get_memory()` is the single entry point everything else should use: it
prefers the daemon (shared, lock-free) and only opens a local `Memory`
when nothing is listening.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal, Self

import httpx

from .config import Settings, get_settings
from .schema import Category, MemoryAtom, Provenance

if TYPE_CHECKING:
    # `.api` pulls in qdrant_client -> fastembed -> onnxruntime; `Memory`
    # (used only in annotations here) and `StoreLocked` (used only inside
    # `open_local_memory`, imported there) are both deferred so remote-mode
    # callers (hooks, the CLI against a running daemon, ...) never pay that
    # cost just to import this module.
    from .api import Memory

DaemonStatus = Literal["up", "rejected", "down"]


class RemoteError(RuntimeError):
    """A non-2xx response from the daemon, or the daemon being unreachable.
    The message is the daemon's own `{"error": ...}` body when available."""


class DaemonRejected(RemoteError):
    """The daemon answered but rejected this client: any non-200 response
    to the health probe (401/403/other), typically a missing or wrong
    `MEMSTRAND_TOKEN`. This is deliberately NOT treated as "daemon down" —
    something is listening and owns the stores, so falling back to a local
    `Memory` here would try to open stores the daemon already holds and
    crash on Qdrant's on-disk lock instead of surfacing the real reason."""


def normalize_atom(atom: MemoryAtom | dict) -> dict:
    """`RemoteMemory.remember`/`.update` return dicts; local `Memory`
    returns `MemoryAtom` objects. Call this to use one shape either way."""
    if isinstance(atom, dict):
        return atom
    from .api import Memory

    return Memory.atom_to_dict(atom)


# Sentinel mirroring `api._ALL_SCOPES` (not imported: `.api` is heavy).
_ALL_SCOPES: Any = object()


def _enum_value(v):
    return v.value if isinstance(v, (Category, Provenance)) else v


def _drop_none(d: dict) -> dict:
    """Keep falsy-but-meaningful values (e.g. `task_id=""` means "global
    only"); only `None` means "omit this filter entirely"."""
    return {k: v for k, v in d.items() if v is not None}


class RemoteMemory:
    """Talks to `memstrand serve`'s REST API. `client` is injectable so
    tests can pass an `httpx.Client(transport=httpx.MockTransport(...))`
    instead of hitting the network."""

    def __init__(
        self,
        url: str,
        token: str | None = None,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ):
        self.url = url.rstrip("/")
        self.token = token
        # applied per-request (not baked into the client) so an injected
        # test client still gets the Authorization header.
        self._auth_headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = client or httpx.Client(base_url=self.url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- transport ------------------------------------------------------------

    def _request(
        self, method: str, path: str, *, not_found_ok: bool = False, raw: bool = False, **kwargs
    ) -> Any:
        headers = {**self._auth_headers, **(kwargs.pop("headers", None) or {})}
        try:
            resp = self._client.request(method, path, headers=headers or None, **kwargs)
        except httpx.HTTPError as exc:
            raise RemoteError(f"cannot reach memstrand daemon at {self.url}: {exc}") from exc
        if not_found_ok and resp.status_code == 404:
            return None
        if resp.status_code >= 400:
            detail = resp.text
            try:
                detail = resp.json().get("error", detail)
            except ValueError:
                pass
            raise RemoteError(detail)
        if raw:
            return resp.text
        if not resp.content:
            return None
        return resp.json()

    # -- write path -----------------------------------------------------------

    def remember(
        self,
        content: str,
        category: Category = Category.fact,
        provenance: Provenance = Provenance.explicit_statement,
        confidence: float = 0.8,
        session_id: str | None = None,
        task_id: str | None = None,
        host: str | None = None,
        ttl_days: float | None = None,
        source: str | None = None,
        source_ref: str | None = None,
        agent: str | None = None,
        scopes: list[str] | None = None,
    ) -> dict:
        payload = _drop_none(
            {
                "content": content,
                "category": _enum_value(category),
                "provenance": _enum_value(provenance),
                "confidence": confidence,
                "session_id": session_id,
                "task_id": task_id,
                "host": host,
                "ttl_days": ttl_days,
                "source": source,
                "source_ref": source_ref,
                "agent": agent,
                "scopes": scopes,
            }
        )
        return self._request("POST", "/api/remember", json=payload)

    def log_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        task_id: str | None = None,
        scopes: list[str] | None = None,
        agent: str | None = None,
        ts: float | None = None,
    ) -> None:
        payload = _drop_none(
            {
                "session_id": session_id, "role": role, "content": content,
                "task_id": task_id, "scopes": scopes, "agent": agent, "ts": ts,
            }
        )
        self._request("POST", "/api/turns", json=payload)

    def log_turns(self, turns: list[dict]) -> None:
        """Batch form: `turns` is a list of {"session_id","role","content","task_id"?,"scopes"?}
        dicts. Used by hooks.py so a burst of transcript turns is one request."""
        self._request("POST", "/api/turns", json={"turns": turns})

    def forget(self, atom_id: str) -> bool:
        result = self._request("DELETE", f"/api/atoms/{atom_id}", not_found_ok=True)
        return bool(result and result.get("forgot"))

    def forget_source(self, source: str, task_id: str | None = _ALL_SCOPES) -> int:
        """Same contract as `Memory.forget_source`: by default delete the
        source's atoms in every scope; passing `task_id` (None = global)
        removes only that scope."""
        payload: dict = {"source": source}
        if task_id is not _ALL_SCOPES:
            payload["scope"] = task_id or ""
        result = self._request("POST", "/api/forget_source", json=payload)
        return result.get("forgot", 0) if result else 0

    def update(
        self,
        atom_id: str,
        content: str | None = None,
        category: Category | None = None,
        confidence: float | None = None,
    ) -> dict:
        payload = _drop_none(
            {"content": content, "category": _enum_value(category), "confidence": confidence}
        )
        return self._request("PATCH", f"/api/atoms/{atom_id}", json=payload)

    # -- scoped read path -------------------------------------------------------

    def recall(
        self,
        query: str,
        session_id: str | None = None,
        task_id: str | None = None,
        host: str | None = None,
        categories: list[Category] | None = None,
        limit: int = 10,
        include_superseded: bool = False,
        track: bool = True,
        scopes: list[str] | None = None,
    ) -> list[dict]:
        # `track` is a local-only internal knob (judge/consolidation callers);
        # the daemon always tracks recall side effects, so it's not sent.
        payload = _drop_none(
            {
                "query": query,
                "session_id": session_id,
                "task_id": task_id,
                "host": host,
                "categories": [_enum_value(c) for c in categories] if categories else None,
                "limit": limit,
                "include_superseded": include_superseded,
                "scopes": scopes,
            }
        )
        result = self._request("POST", "/api/recall", json=payload)
        return result.get("results", []) if result else []

    def get(
        self,
        atom_id: str,
        task_id: str | None = None,
        host: str | None = None,
        scoped: bool = False,
        scopes: list[str] | None = None,
    ) -> dict | None:
        """Mirrors `Memory.get`: default (scoped=False) is the owner-facing
        browse view (any atom, neighbors unfiltered). Pass scoped=True with
        task_id/host to apply the same visibility rule recall() uses."""
        params: dict = _drop_none({"task_id": task_id, "host": host})
        if scopes:
            params["scope"] = list(scopes)
        if scoped:
            params["scoped"] = "1"
        return self._request("GET", f"/api/atoms/{atom_id}", params=params, not_found_ok=True)

    def list(
        self,
        category: Category | None = None,
        task_id: str | None = None,
        host: str | None = None,
        limit: int = 100,
        offset: int = 0,
        agent: str | None = None,
    ) -> list[dict]:
        # task_id is kept even when "" (absent = all tasks, "" = global only).
        params = _drop_none(
            {
                "category": _enum_value(category),
                "task_id": task_id,
                "host": host,
                "agent": agent,
                "limit": limit,
                "offset": offset,
            }
        )
        result = self._request("GET", "/api/atoms", params=params)
        return result.get("atoms", []) if result else []

    def count(self) -> int:
        return self.stats().get("atoms", 0)

    def stats(self) -> dict:
        return self._request("GET", "/api/stats") or {}

    # -- context injection / grounded answering ----------------------------------

    def context(
        self,
        query: str | None = None,
        task_id: str | None = None,
        host: str | None = None,
        limit: int = 8,
        max_chars: int = 4000,
        scopes: list[str] | None = None,
    ) -> str:
        payload = _drop_none(
            {
                "query": query, "task_id": task_id, "host": host,
                "limit": limit, "max_chars": max_chars, "scopes": scopes,
            }
        )
        result = self._request("POST", "/api/context", json=payload)
        return result.get("context", "") if result else ""

    def grounded_answer(
        self,
        question: str,
        task_id: str | None = None,
        host: str | None = None,
        scopes: list[str] | None = None,
        **_unused,
    ) -> str:
        """`**_unused` absorbs local-only `recall()` kwargs (session_id,
        categories, limit, ...) that the `/api/answer` endpoint doesn't
        accept — only `task_id`/`host`/`scopes` make the trip."""
        payload = _drop_none({"question": question, "task_id": task_id, "host": host, "scopes": scopes})
        result = self._request("POST", "/api/answer", json=payload)
        return result.get("answer", "") if result else ""

    # -- scope management -----------------------------------------------------------

    def set_scopes(
        self,
        atom_id: str,
        scopes: list[str] | None = None,
        add: list[str] | None = None,
        remove: list[str] | None = None,
    ) -> dict:
        payload = _drop_none({"set": scopes, "add": add, "remove": remove})
        return self._request("POST", f"/api/atoms/{atom_id}/scopes", json=payload)

    def confirm_scopes(self, atom_id: str) -> dict:
        return self._request("POST", f"/api/atoms/{atom_id}/scopes", json={"keep": True})

    def pending_scope_reviews(self, limit: int = 3, session_key: str | None = None) -> list[dict]:
        if session_key:
            result = self._request("POST", "/api/reviews", json={"limit": limit, "session": session_key})
        else:
            result = self._request("GET", "/api/reviews", params={"limit": limit})
        return result.get("reviews", []) if result else []

    def session_turns(self, ref: str, limit: int | None = None) -> dict:
        params = _drop_none({"limit": limit})
        result = self._request("GET", f"/api/sessions/{ref}/turns", params=params, not_found_ok=True)
        if result is None:
            raise KeyError(f"no such session: {ref}")
        return result

    def mark_distilled(self, session_id: str, reset: bool = False) -> int:
        result = self._request("POST", f"/api/sessions/{session_id}/distilled", json={"reset": reset})
        return int(result.get("last_turn_id", 0)) if result else 0

    def search_turns(self, query: str, limit: int = 20, session: str | None = None) -> list[dict]:
        params = _drop_none({"q": query, "limit": limit, "session": session})
        result = self._request("GET", "/api/turns/search", params=params)
        return result.get("results", []) if result else []

    def rename_scope(self, old: str, new: str) -> int:
        result = self._request("POST", "/api/scopes/rename", json={"old": old, "new": new})
        return result.get("renamed", 0) if result else 0

    def scopes_summary(self) -> dict[str, int]:
        result = self._request("GET", "/api/scopes")
        return result.get("scopes", {}) if result else {}

    # -- sessions / distillation / consolidation ---------------------------------

    def improve(self, session_ids: list[str] | None = None, background: bool = False) -> dict:
        payload = _drop_none({"session_ids": session_ids, "background": background})
        return self._request("POST", "/api/improve", json=payload) or {}

    def consolidate(self) -> dict:
        return self._request("POST", "/api/consolidate") or {}

    # -- export / import / ingest -------------------------------------------------

    def export(self, fmt: str = "jsonl") -> str:
        return self._request("GET", "/api/export", params={"format": fmt}, raw=True)

    def import_jsonl(self, text: str) -> dict:
        return self._request("POST", "/api/import", content=text.encode("utf-8")) or {}

    def ingest(self, path, task_id: str | None = None, force: bool = False) -> dict:
        payload = _drop_none({"path": str(path), "task_id": task_id, "force": force})
        return self._request("POST", "/api/ingest", json=payload) or {}


def _probe_response(url: str, timeout: float, token: str | None) -> httpx.Response | None:
    """One `/api/health` GET. Returns the response, or `None` if the
    daemon is unreachable (connection refused/timeout/any other
    `httpx.HTTPError`, or a malformed `url`) — the "down" case. A
    response object (any status code) means something is listening."""
    headers = {"Authorization": f"Bearer {token}"} if token else None
    try:
        return httpx.get(f"{url}/api/health", timeout=timeout, headers=headers)
    except (httpx.HTTPError, httpx.InvalidURL):
        return None


def probe_daemon(
    url: str | None = None, token: str | None = None, timeout: float = 0.5
) -> DaemonStatus:
    """Tri-state health probe. `url` defaults to `settings.server_url`.

    - `"up"`: the daemon answered 200 — safe to use as `RemoteMemory`.
    - `"rejected"`: the daemon answered with anything else (401, 403, a
      5xx, ...) — something IS listening and owns the stores, it just
      didn't like this client (usually a missing/wrong `MEMSTRAND_TOKEN`).
      This must never be treated the same as "down": the old behavior
      (any non-200 -> "down") sent callers straight into a local `Memory`
      that then crashed on Qdrant's on-disk "already accessed by another
      instance" lock, or silently raced the daemon's SQLite WAL.
    - `"down"`: connection refused, timeout, or any other transport
      error — nothing is listening, a local `Memory` is safe to open.
    """
    url = (url or get_settings().server_url).rstrip("/")
    if token is None:
        token = os.environ.get("MEMSTRAND_TOKEN")
    resp = _probe_response(url, timeout, token)
    if resp is None:
        return "down"
    return "up" if resp.status_code == 200 else "rejected"


def daemon_running(
    url: str | None = None, timeout: float = 0.5, token: str | None = None
) -> bool:
    """Thin bool wrapper over `probe_daemon`, kept for compatibility with
    callers that only care "is it safe to treat the daemon as up" — `True`
    only for `"up"`; both `"rejected"` and `"down"` are `False`. Prefer
    `probe_daemon` directly wherever the "rejected" case needs its own
    handling (it almost always does, see `probe_daemon`)."""
    return probe_daemon(url, timeout=timeout, token=token) == "up"


def _wrap_qdrant_lock_error(settings: Settings, exc: RuntimeError) -> RemoteError:
    return RemoteError(
        f"the memstrand store at {settings.home} is locked by another process "
        "(is a daemon running on a different MEMSTRAND_PORT?)"
    )


def open_local_memory(settings: Settings | None = None) -> Memory:
    """Construct a local `Memory`, turning Qdrant's on-disk single-writer
    lock into an actionable message instead of a raw
    "Storage folder ... already accessed by another instance" error.

    `Memory()` now raises `StoreLocked` (a `RuntimeError`) itself, eagerly,
    at construction time — the vector store is no longer built lazily on
    first embed, so this is always caught here rather than surfacing later
    inside `remember()`/`recall()`. The raw substring check stays as a
    fallback for any other RuntimeError shaped like Qdrant's own lock
    message that doesn't come wrapped in `StoreLocked` (e.g. a different
    code path reaching the same embedded store)."""
    from .api import Memory, StoreLocked

    settings = settings or get_settings()
    try:
        return Memory(settings=settings)
    except StoreLocked as exc:
        raise _wrap_qdrant_lock_error(settings, exc) from exc
    except RuntimeError as exc:
        if "already accessed by another instance" in str(exc):
            raise _wrap_qdrant_lock_error(settings, exc) from exc
        raise


def get_memory(prefer_remote: bool = True, timeout: float = 30.0) -> RemoteMemory | Memory:
    """`RemoteMemory` if the daemon is up, a local `Memory()` if nothing
    answers, or `DaemonRejected` if something answers but refuses this
    client (see `probe_daemon`) — the caller gets the real reason instead
    of a confusing Qdrant lock crash.

    Preferring remote matters because the embedded Qdrant/SQLite stores
    are single-writer: opening them locally while a daemon already holds
    them will fail outright (Qdrant's on-disk lock) or silently race
    (SQLite WAL from two processes). The daemon, when running, is always
    the source of truth; local mode exists for one-shot CLI use with no
    daemon around, and for tests. `timeout` is the per-request timeout for
    the remote client (long-running calls like a synchronous `improve`
    need more than the default).
    """
    settings = get_settings()
    token = os.environ.get("MEMSTRAND_TOKEN")
    if prefer_remote:
        url = settings.server_url.rstrip("/")
        resp = _probe_response(url, 0.5, token)
        if resp is not None:
            if resp.status_code == 200:
                return RemoteMemory(settings.server_url, token=token, timeout=timeout)
            raise DaemonRejected(
                f"a memstrand daemon is running at {settings.server_url} but rejected "
                f"this client (HTTP {resp.status_code}) — set MEMSTRAND_TOKEN to the "
                "daemon's token"
            )
    return open_local_memory(settings)
