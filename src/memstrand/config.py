"""Runtime configuration: environment-driven settings, single source of truth.

Every other module reads config through `get_settings()` instead of poking
`os.environ` directly, so tests can monkeypatch env vars freely and
production code and tests always agree on defaults. `Settings` is a plain
dataclass snapshot, not a live view — call `get_settings()` again after
changing the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


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
    """Fresh `Settings` read from the current environment."""
    return Settings()
