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
        resp = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        resp.raise_for_status()
        return [row["embedding"] for row in resp.json()["data"]]


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
