"""Embedding backends behind one shape: `.embed(texts) -> iterable[list[float]]`.

fastembed is the local, zero-key default (ONNX models, no torch, no
network after the first model download). An OpenAI-compatible
`/embeddings` endpoint (Ollama, llama.cpp, LM Studio, or real OpenAI) is
the alternative for people already running a bigger embedding model
somewhere. Both — and the HashEmbedder test double used across the test
suite — share this one interface, so `Memory` never needs to know which
one it's holding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from .config import Settings


def _quiet_onnxruntime() -> None:
    """Best-effort: silence onnxruntime's default logger before fastembed
    loads a model, so local-mode daemons/CLI don't spew its hardware-probe
    warning (e.g. "Skipping pci_bus_id for PCI path ..." on some Linux
    VMs) to stderr. fastembed depends on onnxruntime, so by the time a
    model is actually constructed it's importable; this is still wrapped
    defensively since it's cosmetic, not load-bearing."""
    try:
        import onnxruntime

        onnxruntime.set_default_logger_severity(3)  # 3 = ERROR, suppresses WARNING
    except (ImportError, AttributeError):
        pass


class EmbeddingUnavailable(RuntimeError):
    """The configured embedding backend didn't answer (connection refused,
    timed out, or returned an error status). This is transient and holds
    no permanent failed state anywhere: `Memory` never caches this as a
    terminal failure, so a later call (once the backend is up) retries
    from scratch and succeeds normally. Raised instead of letting the raw
    httpx exception escape, so `memstrand serve` can start with the
    embedding backend down and only the first real embed/recall/remember
    call surfaces a clear, actionable error."""


class FastEmbedder:
    """Lazy-loaded fastembed `TextEmbedding`. Importing/constructing this
    class never touches the network or loads an ONNX model — that only
    happens on first `.embed()` call, so tests that never call it don't
    need the model downloaded."""

    def __init__(self, model: str | None = None):
        self._model_name = model
        self._model = None

    def _load(self):
        if self._model is None:
            _quiet_onnxruntime()
            from fastembed import TextEmbedding

            kwargs = {"model_name": self._model_name} if self._model_name else {}
            self._model = TextEmbedding(**kwargs)
        return self._model

    def embed(self, texts):
        return [list(v) for v in self._load().embed(list(texts))]


class OpenAIEmbedder:
    """OpenAI-compatible `/embeddings` endpoint."""

    def __init__(self, base_url: str, model: str, api_key: str = "local"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def embed(self, texts):
        texts = list(texts)
        if not texts:
            return []
        try:
            resp = httpx.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": texts},
                timeout=120,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            # Covers both transport failures (ConnectError, TimeoutException,
            # ...) and HTTP error statuses (HTTPStatusError) -- either way the
            # backend didn't give us an embedding, and there's nothing stored
            # as a result of this call.
            raise EmbeddingUnavailable(
                f"embedding backend at {self.base_url} unreachable: {e}"
            ) from e
        try:
            return [row["embedding"] for row in resp.json()["data"]]
        except (KeyError, IndexError, ValueError) as e:
            raise EmbeddingUnavailable(
                f"embedding backend at {self.base_url} unreachable: malformed response: {e}"
            ) from e


def make_embedder(settings: Settings):
    """Build the embedder `Settings` asks for."""
    if settings.embed_backend == "openai":
        if not settings.embed_model:
            raise ValueError(
                "MEMSTRAND_EMBED_MODEL is required when MEMSTRAND_EMBED_BACKEND=openai"
            )
        return OpenAIEmbedder(
            settings.embed_base_url or settings.llm_base_url, settings.embed_model, settings.llm_api_key
        )
    return FastEmbedder(settings.embed_model)


class Reranker:
    """Cross-encoder reranking via fastembed's `TextCrossEncoder`. Off by
    default — most local setups skip this; set `MEMSTRAND_RERANK_MODEL`
    (e.g. `Xenova/ms-marco-MiniLM-L-6-v2`) to turn it on."""

    def __init__(self, model: str):
        self.model = model
        self._model = None

    def _load(self):
        if self._model is None:
            _quiet_onnxruntime()
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
            except ImportError as e:
                raise ImportError(
                    "Reranker requires fastembed's cross-encoder support. "
                    "Install/upgrade with: pip install -U fastembed"
                ) from e
            self._model = TextCrossEncoder(model_name=self.model)
        return self._model

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        return list(self._load().rerank(query, docs))
