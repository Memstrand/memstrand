"""Scope identity: turn a filesystem path into a stable scope key.

See docs/design/scoping-v2.md §1. A path inside a git repo with a remote
resolves to the normalized remote URL (`github.com/org/repo`), so the same
repo cloned under any folder name, on any machine, gets the same key.
Anything else resolves to a machine-qualified path
(`local:<hostname>:<abs path>`), which can't collide across machines on a
shared server.

Kept free of heavy imports (no qdrant/fastembed): the Claude Code hooks
call this on every prompt.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
from pathlib import Path

LOCAL_PREFIX = "local:"

# Hosts whose repo paths are case-insensitive, so folding case can't merge
# two distinct repos there. Everything else keeps path case as-is.
_CASE_INSENSITIVE_HOSTS = {"github.com"}

_SCP_LIKE = re.compile(r"^(?:[\w.-]+@)?(?P<host>[\w.-]+):(?P<path>(?!//).+)$")


def normalize_remote_url(url: str) -> str | None:
    """`git@github.com:Org/Repo.git` / `https://user:tok@host/org/repo` /
    `ssh://git@host:2222/org/repo` -> `host/org/repo`. Credentials, scheme,
    port, `.git`, and trailing slashes are dropped; the host is lowercased,
    the path only for `_CASE_INSENSITIVE_HOSTS`. Returns None for anything
    that doesn't look like a remote (a local path remote, an empty string)."""
    url = url.strip()
    if not url:
        return None
    if "://" in url:
        scheme, rest = url.split("://", 1)
        if scheme.lower() == "file":
            return None
        authority, _, path = rest.partition("/")
        host = authority.rsplit("@", 1)[-1].split(":", 1)[0]
    else:
        m = _SCP_LIKE.match(url)
        if not m or url.startswith(("/", ".", "~")):
            return None
        host, path = m.group("host"), m.group("path")
    host = host.lower()
    path = path.strip("/")
    path = path.removesuffix(".git")
    path = path.strip("/")
    if not host or not path:
        return None
    if host in _CASE_INSENSITIVE_HOSTS:
        path = path.lower()
    return f"{host}/{path}"


def git_root(path: str | Path) -> Path | None:
    """Nearest ancestor (or `path` itself) containing `.git` -- a directory
    for a normal clone, a file for a worktree/submodule."""
    try:
        start = Path(path).expanduser().resolve()
    except OSError:
        return None
    if start.is_file():
        start = start.parent
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _remote_url(root: Path) -> str | None:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True, text=True, timeout=5, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    url = run("remote", "get-url", "origin")
    if url:
        return url
    remotes = (run("remote") or "").split()
    return run("remote", "get-url", remotes[0]) if remotes else None


def _config_mtime(root: Path) -> float:
    """mtime of the repo's config file (cache key). Handles worktrees, where
    `.git` is a `gitdir: ...` pointer file."""
    dotgit = root / ".git"
    try:
        if dotgit.is_file():
            target = dotgit.read_text().strip().removeprefix("gitdir:").strip()
            gitdir = (root / target).resolve()
            common = gitdir / "commondir"
            if common.exists():
                gitdir = (gitdir / common.read_text().strip()).resolve()
            return (gitdir / "config").stat().st_mtime
        return (dotgit / "config").stat().st_mtime
    except OSError:
        return 0.0


def local_scope(path: str | Path, hostname: str | None = None) -> str:
    resolved = Path(path).expanduser().resolve()
    return f"{LOCAL_PREFIX}{hostname or socket.gethostname()}:{resolved}"


class ScopeResolver:
    """Resolves paths to scope keys, caching per git root in a JSON file
    (keyed on the repo config's mtime, so a changed remote is picked up)."""

    def __init__(self, cache_path: Path | None = None, hostname: str | None = None):
        self.cache_path = cache_path
        self.hostname = hostname or socket.gethostname()
        self._cache: dict[str, dict] | None = None
        self._dirty = False

    def _load(self) -> dict[str, dict]:
        if self._cache is None:
            self._cache = {}
            if self.cache_path is not None and self.cache_path.exists():
                try:
                    data = json.loads(self.cache_path.read_text())
                    if isinstance(data, dict):
                        self._cache = data
                except (OSError, json.JSONDecodeError):
                    pass
        return self._cache

    def save(self) -> None:
        if not self._dirty or self.cache_path is None or self._cache is None:
            return
        from .config import ensure_private_dir, secure_file

        ensure_private_dir(self.cache_path.parent)
        tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._cache, indent=2))
        secure_file(tmp)
        os.replace(tmp, self.cache_path)
        self._dirty = False

    def resolve(self, path: str | Path) -> str:
        """Scope key for `path`: its repo's normalized remote URL, else
        `local:<hostname>:<repo root or path>`."""
        root = git_root(path)
        if root is None:
            return local_scope(path, self.hostname)
        cache = self._load()
        mtime = _config_mtime(root)
        entry = cache.get(str(root))
        if entry is not None and entry.get("mtime") == mtime:
            return entry["scope"]
        url = _remote_url(root)
        scope = normalize_remote_url(url) if url else None
        scope = scope or local_scope(root, self.hostname)
        cache[str(root)] = {"scope": scope, "mtime": mtime}
        self._dirty = True
        return scope


def resolve_scope(path: str | Path, cache_path: Path | None = None) -> str:
    """One-shot `ScopeResolver(cache_path).resolve(path)`, saving the cache."""
    resolver = ScopeResolver(cache_path)
    try:
        return resolver.resolve(path)
    finally:
        resolver.save()


def agent_name(kind: str) -> str:
    """Who wrote a memory: `<kind>@<hostname>`, e.g. `claude-code@M-TXWV62R02T`.
    Computed client-side, so it stays right against a central server."""
    return f"{kind}@{socket.gethostname()}"


def short_name(scope: str) -> str:
    """Human label: the last path segment (`github.com/org/repo` -> `repo`,
    `local:host:/Users/me/.claude` -> `.claude`). "" (global) -> `global`."""
    if not scope:
        return "global"
    if scope.startswith(LOCAL_PREFIX):
        scope = scope.split(":", 2)[-1]
    return scope.rstrip("/").rsplit("/", 1)[-1] or scope


class AmbiguousScope(ValueError):
    """A short name matches more than one known scope."""

    def __init__(self, name: str, matches: list[str]):
        self.name = name
        self.matches = matches
        super().__init__(
            f"scope {name!r} is ambiguous; use one of: " + ", ".join(sorted(matches))
        )


def resolve_alias(name: str, known: list[str]) -> str:
    """Map a user-typed scope (`--repo memstrand`) to a full key. An exact
    key match wins; else a unique `short_name` match; else `name` is taken
    literally (a new or named scope like `notes`). "global" always means
    the global scope "" -- it's reserved, so a named scope can't be called
    "global". Raises AmbiguousScope when a short name matches several
    known keys."""
    if name in ("", "global"):
        return ""
    if name in known:
        return name
    matches = [k for k in known if k and short_name(k) == name]
    if len(matches) > 1:
        raise AmbiguousScope(name, matches)
    return matches[0] if matches else name
