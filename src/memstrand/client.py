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
from typing import Any, Self

import httpx

from .api import Memory
from .config import get_settings
from .schema import Category, MemoryAtom, Provenance


class RemoteError(RuntimeError):
    """A non-2xx response from the daemon, or the daemon being unreachable.
    The message is the daemon's own `{"error": ...}` body when available."""


def normalize_atom(atom: MemoryAtom | dict) -> dict:
    """`RemoteMemory.remember`/`.update` return dicts; local `Memory`
    returns `MemoryAtom` objects. Call this to use one shape either way."""
    if isinstance(atom, dict):
        return atom
    return Memory.atom_to_dict(atom)


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
    ) -> dict:
        # `agent` has no REST field yet (see server contract) — dropped silently.
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
            }
        )
        return self._request("POST", "/api/remember", json=payload)

    def log_turn(
        self, session_id: str, role: str, content: str, task_id: str | None = None
    ) -> None:
        payload = _drop_none(
            {"session_id": session_id, "role": role, "content": content, "task_id": task_id}
        )
        self._request("POST", "/api/turns", json=payload)

    def log_turns(self, turns: list[dict]) -> None:
        """Batch form: `turns` is a list of {"session_id","role","content","task_id"?}
        dicts. Used by hooks.py so a burst of transcript turns is one request."""
        self._request("POST", "/api/turns", json={"turns": turns})

    def forget(self, atom_id: str) -> bool:
        result = self._request("DELETE", f"/api/atoms/{atom_id}", not_found_ok=True)
        return bool(result and result.get("forgot"))

    def forget_source(self, source: str) -> int:
        result = self._request("POST", "/api/forget_source", json={"source": source})
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
    ) -> dict | None:
        """Mirrors `Memory.get`: default (scoped=False) is the owner-facing
        browse view (any atom, neighbors unfiltered). Pass scoped=True with
        task_id/host to apply the same visibility rule recall() uses."""
        params = _drop_none({"task_id": task_id, "host": host})
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
    ) -> list[dict]:
        # task_id is kept even when "" (absent = all tasks, "" = global only).
        params = _drop_none(
            {
                "category": _enum_value(category),
                "task_id": task_id,
                "host": host,
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
    ) -> str:
        payload = _drop_none(
            {"query": query, "task_id": task_id, "host": host, "limit": limit, "max_chars": max_chars}
        )
        result = self._request("POST", "/api/context", json=payload)
        return result.get("context", "") if result else ""

    def grounded_answer(self, question: str, task_id: str | None = None, host: str | None = None, **_unused) -> str:
        """`**_unused` absorbs local-only `recall()` kwargs (session_id,
        categories, limit, ...) that the `/api/answer` endpoint doesn't
        accept — only `task_id`/`host` make the trip."""
        payload = _drop_none({"question": question, "task_id": task_id, "host": host})
        result = self._request("POST", "/api/answer", json=payload)
        return result.get("answer", "") if result else ""

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


def daemon_running(url: str | None = None, timeout: float = 0.5) -> bool:
    """Quick health probe. `url` defaults to `settings.server_url`."""
    url = (url or get_settings().server_url).rstrip("/")
    try:
        resp = httpx.get(f"{url}/api/health", timeout=timeout)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


def get_memory(prefer_remote: bool = True) -> RemoteMemory | Memory:
    """`RemoteMemory` if the daemon is up, else a local `Memory()`.

    Preferring remote matters because the embedded Qdrant/SQLite stores
    are single-writer: opening them locally while a daemon already holds
    them will fail outright (Qdrant's on-disk lock) or silently race
    (SQLite WAL from two processes). The daemon, when running, is always
    the source of truth; local mode exists for one-shot CLI use with no
    daemon around, and for tests.
    """
    settings = get_settings()
    if prefer_remote and daemon_running(settings.server_url):
        token = os.environ.get("MEMSTRAND_TOKEN")
        return RemoteMemory(settings.server_url, token=token)
    return Memory(settings=settings)
