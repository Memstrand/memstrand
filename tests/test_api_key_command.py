"""MEMSTRAND_LLM_API_KEY_COMMAND: per-use key resolution for short-lived gateway tokens."""

from __future__ import annotations

import shlex
import sys

import pytest

from memstrand import config
from memstrand.config import Settings, resolve_llm_api_key
from memstrand.embeddings import OpenAIEmbedder, make_embedder


@pytest.fixture(autouse=True)
def _clear_cache():
    config._api_key_cache.clear()
    yield
    config._api_key_cache.clear()


def _counting_command(tmp_path, token: str = "tok") -> tuple[str, object]:
    """A command that prints `token<n>` where n counts invocations."""
    counter = tmp_path / "count"
    script = (
        f"import pathlib; p = pathlib.Path({str(counter)!r}); "
        "n = int(p.read_text()) + 1 if p.exists() else 1; "
        f"p.write_text(str(n)); print({token!r} + str(n))"
    )
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}", counter


def test_static_key_without_command(monkeypatch):
    monkeypatch.setenv("MEMSTRAND_LLM_API_KEY", "static")
    assert resolve_llm_api_key(Settings()) == "static"


def test_command_output_is_used_and_cached(monkeypatch, tmp_path):
    command, counter = _counting_command(tmp_path)
    monkeypatch.setenv("MEMSTRAND_LLM_API_KEY_COMMAND", command)
    settings = Settings()
    assert resolve_llm_api_key(settings) == "tok1"
    assert resolve_llm_api_key(settings) == "tok1"
    assert counter.read_text() == "1"


def test_command_rerun_after_ttl(monkeypatch, tmp_path):
    command, _ = _counting_command(tmp_path)
    monkeypatch.setenv("MEMSTRAND_LLM_API_KEY_COMMAND", command)
    monkeypatch.setattr(config, "_API_KEY_CACHE_TTL", 0.0)
    settings = Settings()
    assert resolve_llm_api_key(settings) == "tok1"
    assert resolve_llm_api_key(settings) == "tok2"


def test_failing_command_falls_back_to_static_key(monkeypatch):
    monkeypatch.setenv("MEMSTRAND_LLM_API_KEY", "static")
    monkeypatch.setenv("MEMSTRAND_LLM_API_KEY_COMMAND", f"{shlex.quote(sys.executable)} -c 'raise SystemExit(1)'")
    assert resolve_llm_api_key(Settings()) == "static"


def test_failing_command_reuses_last_good_token(monkeypatch, tmp_path):
    command, _ = _counting_command(tmp_path)
    monkeypatch.setenv("MEMSTRAND_LLM_API_KEY_COMMAND", command)
    monkeypatch.setattr(config, "_API_KEY_CACHE_TTL", 0.0)
    settings = Settings()
    assert resolve_llm_api_key(settings) == "tok1"

    def boom(*a, **k):
        raise OSError("helper gone")

    monkeypatch.setattr(config.subprocess, "run", boom)
    assert resolve_llm_api_key(settings) == "tok1"


def test_empty_output_is_a_failure(monkeypatch):
    monkeypatch.setenv("MEMSTRAND_LLM_API_KEY", "static")
    monkeypatch.setenv("MEMSTRAND_LLM_API_KEY_COMMAND", f"{shlex.quote(sys.executable)} -c 'pass'")
    assert resolve_llm_api_key(Settings()) == "static"


class _ConstEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def test_memory_resolves_key_per_use(tmp_path, monkeypatch):
    from memstrand import Memory

    command, _ = _counting_command(tmp_path)
    monkeypatch.setenv("MEMSTRAND_LLM_API_KEY_COMMAND", command)
    monkeypatch.setattr(config, "_API_KEY_CACHE_TTL", 0.0)
    mem = Memory.in_memory(embedder=_ConstEmbedder())
    assert mem.llm_api_key == "tok1"
    assert mem.llm_api_key == "tok2"


def test_openai_embedder_resolves_key_per_request(monkeypatch, tmp_path):
    command, _ = _counting_command(tmp_path)
    monkeypatch.setenv("MEMSTRAND_EMBED_BACKEND", "openai")
    monkeypatch.setenv("MEMSTRAND_EMBED_MODEL", "m")
    monkeypatch.setenv("MEMSTRAND_LLM_API_KEY_COMMAND", command)
    monkeypatch.setattr(config, "_API_KEY_CACHE_TTL", 0.0)
    embedder = make_embedder(Settings())
    assert isinstance(embedder, OpenAIEmbedder)
    assert embedder.api_key == "tok1"
    assert embedder.api_key == "tok2"
    assert OpenAIEmbedder("http://x", "m", "plain").api_key == "plain"


def test_slow_refresh_never_blocks_other_callers(monkeypatch, tmp_path):
    """Review warning: while one thread runs a slow refresh, other callers
    get the stale token immediately instead of waiting on a global lock."""
    import threading
    import time

    monkeypatch.setenv("MEMSTRAND_LLM_API_KEY_COMMAND", "helper")
    settings = Settings()
    config._api_key_cache["helper"] = ("stale", time.monotonic() - 3600)
    started, release = threading.Event(), threading.Event()

    def slow_command(command):
        started.set()
        release.wait(5)
        return "fresh"

    monkeypatch.setattr(config, "_run_key_command", slow_command)
    results: list[str] = []
    refresher = threading.Thread(target=lambda: results.append(resolve_llm_api_key(settings)))
    refresher.start()
    assert started.wait(5)
    t0 = time.monotonic()
    assert resolve_llm_api_key(settings) == "stale"
    assert time.monotonic() - t0 < 1.0
    release.set()
    refresher.join(5)
    assert results == ["fresh"]
    assert resolve_llm_api_key(settings) == "fresh"
