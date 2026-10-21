"""Runtime configuration: environment-driven settings, single source of truth.

Every other module reads config through `get_settings()` instead of poking
`os.environ` directly, so tests can monkeypatch env vars freely and
production code and tests always agree on defaults. `Settings` is a plain
dataclass snapshot, not a live view — call `get_settings()` again after
changing the environment.
"""

from __future__ import annotations

import math
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


class ConfigError(ValueError):
    """An environment variable memstrand reads at `get_settings()` time
    has a value outside its valid range (non-numeric, NaN/inf, negative,
    out of bounds, ...). Names the offending variable and the expected
    range/format instead of letting a bare `ValueError` (or, worse, a
    silently-accepted nonsense value) through -- e.g.
    `MEMSTRAND_CONSOLIDATE_HOURS=nan` used to be accepted at face value
    and permanently disabled the consolidation scheduler with no error
    anywhere."""


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as e:
        raise ConfigError(f"{name}={raw!r} is not a valid number (expected a float)") from e
    if math.isnan(value) or math.isinf(value):
        raise ConfigError(f"{name}={raw!r} must be a finite number (got {value})")
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as e:
        raise ConfigError(f"{name}={raw!r} is not a valid integer") from e
    return value


def _validate_port(name: str, value: int) -> int:
    if not (1 <= value <= 65535):
        raise ConfigError(f"{name}={value} is out of range (expected an integer in 1-65535)")
    return value


def _validate_positive_finite_float(name: str, value: float) -> float:
    """Finite and > 0 -- the gate `_env_float` already rejects NaN/inf
    and non-numeric input; this additionally rejects zero/negative values
    that `_env_float` alone would happily accept (e.g.
    `MEMSTRAND_CONSOLIDATE_HOURS=0` or `=-5` silently breaking the
    scheduler in a different way than `=nan` does)."""
    if not (value > 0) or math.isnan(value) or math.isinf(value):
        raise ConfigError(f"{name}={value} must be a finite number greater than 0")
    return value


def _validate_unit_interval(name: str, value: float) -> float:
    if math.isnan(value) or not (0.0 <= value <= 1.0):
        raise ConfigError(f"{name}={value} is out of range (expected a float in 0.0-1.0)")
    return value


@dataclass
class Settings:
    """One process's view of its environment. `home` is created lazily by
    whoever first needs it on disk (Memory, ingest, etc.) — constructing
    Settings never touches the filesystem."""

    home: Path = field(
        default_factory=lambda: Path(os.environ.get("MEMSTRAND_HOME", "~/.memstrand")).expanduser()
    )
    qdrant_url: str | None = field(default_factory=lambda: os.environ.get("MEMSTRAND_QDRANT_URL") or None)
    graph_backend: str = field(
        default_factory=lambda: os.environ.get("MEMSTRAND_GRAPH_BACKEND", "sqlite")
    )
    llm_base_url: str = field(
        default_factory=lambda: os.environ.get("MEMSTRAND_LLM_BASE_URL", "http://localhost:47811/v1")
    )
    llm_model: str = field(default_factory=lambda: os.environ.get("MEMSTRAND_LLM_MODEL", "local"))
    llm_api_key: str = field(default_factory=lambda: os.environ.get("MEMSTRAND_LLM_API_KEY", "local"))
    # Shell command whose stdout is the LLM bearer key, re-run (with a short
    # cache) instead of using the static `llm_api_key` -- for gateways that
    # mint short-lived tokens. See `resolve_llm_api_key`.
    llm_api_key_command: str | None = field(
        default_factory=lambda: os.environ.get("MEMSTRAND_LLM_API_KEY_COMMAND") or None
    )
    embed_backend: str = field(
        default_factory=lambda: os.environ.get("MEMSTRAND_EMBED_BACKEND", "fastembed")
    )
    embed_model: str | None = field(default_factory=lambda: os.environ.get("MEMSTRAND_EMBED_MODEL") or None)
    embed_base_url: str | None = field(
        default_factory=lambda: os.environ.get("MEMSTRAND_EMBED_BASE_URL") or None
    )
    rerank_model: str | None = field(default_factory=lambda: os.environ.get("MEMSTRAND_RERANK_MODEL") or None)
    server_host: str = field(default_factory=lambda: os.environ.get("MEMSTRAND_HOST", "127.0.0.1"))
    server_port: int = field(default_factory=lambda: _env_int("MEMSTRAND_PORT", 7777))
    consolidate_hours: float = field(
        default_factory=lambda: _env_float("MEMSTRAND_CONSOLIDATE_HOURS", 24)
    )
    default_task: str | None = field(default_factory=lambda: os.environ.get("MEMSTRAND_TASK") or None)
    # Recall drops any hit whose absolute relevance score (see
    # `ranking.absolute_relevance`) falls below this threshold. Default 0.0
    # (no filtering) because the "right" cutoff is MODEL-DEPENDENT: a hash
    # embedder's cosine similarities and a real embedding model's (e.g.
    # fastembed, OpenAI) live on different distributions, and an unrelated
    # pair can score anywhere from ~0.0 to ~0.3+ depending on the model. Tune
    # this per deployment once you've observed your own model's baseline for
    # unrelated content — e.g. 0.2-0.3 is a reasonable starting point for
    # most sentence embedding models.
    min_relevance: float = field(default_factory=lambda: _env_float("MEMSTRAND_MIN_RELEVANCE", 0.0))

    def __post_init__(self) -> None:
        if self.embed_base_url is None:
            self.embed_base_url = self.llm_base_url

    @property
    def db_path(self) -> Path:
        return self.home / "memstrand.db"

    @property
    def kuzu_path(self) -> Path:
        return self.home / "kuzu"

    @property
    def qdrant_path(self) -> Path:
        return self.home / "qdrant"

    @property
    def server_url(self) -> str:
        return f"http://{self.server_host}:{self.server_port}"


def get_settings() -> Settings:
    """Fresh `Settings` read from the current environment.

    Raises `ConfigError` (a `ValueError`) if any environment-driven
    numeric setting is out of its valid range: `MEMSTRAND_PORT` must be
    an integer in 1-65535; `MEMSTRAND_CONSOLIDATE_HOURS` must be a finite
    number > 0 (a NaN/inf/non-positive value used to be silently accepted
    and permanently disabled the consolidation scheduler); `MEMSTRAND_MIN_RELEVANCE`
    must be a finite number in 0.0-1.0. `_env_float`/`_env_int` already
    reject non-numeric and NaN/inf values; this adds the range checks on
    top, once per setting, in one place."""
    settings = Settings()
    _validate_port("MEMSTRAND_PORT", settings.server_port)
    _validate_positive_finite_float("MEMSTRAND_CONSOLIDATE_HOURS", settings.consolidate_hours)
    _validate_unit_interval("MEMSTRAND_MIN_RELEVANCE", settings.min_relevance)
    return settings


_API_KEY_CACHE_TTL = 60.0
_api_key_cache: dict[str, tuple[str, float]] = {}
_api_key_cache_lock = threading.Lock()  # guards the two dicts only, never held while shelling out
_api_key_refresh_locks: dict[str, threading.Lock] = {}


def _run_key_command(command: str) -> str | None:
    try:
        result = subprocess.run(  # shell=True: the command comes from the user's own env
            command, shell=True, capture_output=True, text=True, timeout=30, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def resolve_llm_api_key(settings: Settings) -> str:
    """The bearer key to send to the LLM endpoint right now.

    Without `MEMSTRAND_LLM_API_KEY_COMMAND` this is just the static
    `MEMSTRAND_LLM_API_KEY`. With it, the command's stripped stdout is used
    and cached for `_API_KEY_CACHE_TTL` seconds, so a long-running daemon
    keeps picking up refreshed tokens (same idea as Claude Code's
    `apiKeyHelper`). Never raises: if the command fails, the last good
    token is reused, else the static key -- the LLM call itself then
    surfaces the auth error, which every caller already handles.

    Refreshes are serialized per command and never block other callers:
    while one thread runs a (possibly slow) refresh, others get the stale
    cached token immediately. Only the very first fetch, with nothing
    cached yet, waits for the command."""
    command = settings.llm_api_key_command
    if not command:
        return settings.llm_api_key
    with _api_key_cache_lock:
        cached = _api_key_cache.get(command)
        if cached is not None and time.monotonic() - cached[1] < _API_KEY_CACHE_TTL:
            return cached[0]
        refresh_lock = _api_key_refresh_locks.setdefault(command, threading.Lock())
    if not refresh_lock.acquire(blocking=cached is None):
        return cached[0]  # another thread is refreshing; the stale token is fine meanwhile
    try:
        with _api_key_cache_lock:
            current = _api_key_cache.get(command)
            if current is not None and time.monotonic() - current[1] < _API_KEY_CACHE_TTL:
                return current[0]  # refreshed while we waited
        token = _run_key_command(command)
        if token is None:
            return current[0] if current is not None else settings.llm_api_key
        with _api_key_cache_lock:
            _api_key_cache[command] = (token, time.monotonic())
        return token
    finally:
        refresh_lock.release()


def ensure_private_dir(path: Path | str) -> Path:
    """Create `path` (and any missing parents) with 0o700 permissions if
    it doesn't exist yet. A directory that already exists — including a
    user-supplied, non-empty MEMSTRAND_HOME — is returned untouched: this
    only tightens permissions on directories memstrand itself creates, it
    never re-chmods something that was already there.

    Every module that creates a directory under MEMSTRAND_HOME (the home
    dir itself, the sqlite/kuzu/qdrant subdirs, the ingest state file's
    parent, export backups, ...) should go through this instead of a bare
    `os.makedirs`/`Path.mkdir`, so file permissions are hardened in one
    place. Thread-safety note: the umask toggle below is process-wide and
    scoped as tightly as possible (restored in `finally`), but a
    concurrent unrelated `mkdir` on another thread during that narrow
    window could race it — acceptable for how rarely this runs (once per
    store/home at construction time).
    """
    p = Path(path)
    if p.exists():
        return p
    old_umask = os.umask(0o077)
    try:
        p.mkdir(parents=True, exist_ok=True)
    finally:
        os.umask(old_umask)
    try:
        p.chmod(0o700)
    except OSError:
        pass
    return p


def secure_file(path: Path | str) -> Path:
    """chmod an existing file to 0o600 (owner read/write only). No-op if
    the file doesn't exist yet (the sqlite db's `-wal`/`-shm` siblings in
    particular are created lazily by sqlite itself, after the point
    callers can reasonably call this) — callers that need those covered
    too should call this again after they're known to exist."""
    p = Path(path)
    if p.exists():
        try:
            p.chmod(0o600)
        except OSError:
            pass
    return p
