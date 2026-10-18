"""A minimal OpenAI-compatible stub server used across the integration
suite in place of a real LLM/embedding backend.

Endpoints:
  - `POST /v1/embeddings`  -- deterministic hash-bucket vectors (same
    scheme as the `HashEmbedder`/`_EmbeddingStubHandler` test doubles used
    elsewhere in the suite), so a real `OpenAIEmbedder` round-trips
    something a hybrid recall can actually match on.
  - `POST /v1/chat/completions` -- stream and non-stream. Three behaviors,
    chosen purely by comparing the request's system prompt against
    memstrand's own prompt constants (imported, not duplicated):
      * `EXTRACTION_SYSTEM_PROMPT` (pipeline/extract.py) -> a canned JSON
        array of candidate atoms, built by scanning the transcript for
        `REMEMBER_INSTRUCTION:`/`REMEMBER_FACT:` markers (whichever role
        said it) -- lets tests control exactly what the "LLM" extracts
        without needing a real model, while the actual trust-boundary
        enforcement (user-turn corroboration) still runs for real in
        product code, since the second (user-only) extraction pass hits
        this same marker-scanning logic on a different, filtered
        transcript.
      * `JUDGE_SYSTEM_PROMPT` (pipeline/distill.py) -> a canned "add"
        verdict.
      * anything else -> echoes the request's system prompt back as the
        reply content, so proxy tests can assert memory injection
        happened by grepping the reply.
  - `GET /v1/models` -- one canned model id.
  - `POST /_control` -- set controllable failure modes for subsequent
    requests (500, a 404 "bad model" response, artificial latency);
    `GET /_control` reads them back; `POST /_control/reset` clears them.

Runs in a background thread via uvicorn, bound to an OS-assigned port
(`StubOpenAI(port=None)`) or a caller-chosen one (so a daemon can be
pointed at a URL before the stub is actually listening there, to
exercise "embedding backend currently down" startup behavior).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import socket
import threading
import time
from dataclasses import dataclass, field

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from memstrand.pipeline.distill import JUDGE_SYSTEM_PROMPT
from memstrand.pipeline.extract import EXTRACTION_SYSTEM_PROMPT

_INSTRUCTION_MARKER = "REMEMBER_INSTRUCTION:"
_FACT_MARKER = "REMEMBER_FACT:"


def free_port() -> int:
    """Bind to an OS-assigned port, then release it. Small reuse-race
    window, standard practice for "give me a free port" test fixtures
    (same pattern as `tests/test_daemon_probe.py`'s `_closed_port`)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _hash_embed(text: str, dim: int) -> list[float]:
    v = [0.0] * dim
    for w in text.lower().split():
        idx = int(hashlib.md5(w.encode()).hexdigest(), 16) % dim
        v[idx] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _system_content(messages: list[dict]) -> str:
    for m in messages:
        if m.get("role") == "system":
            content = m.get("content")
            return content if isinstance(content, str) else ""
    return ""


def _last_user_content(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content")
            return content if isinstance(content, str) else ""
    return ""


def _extraction_reply(user_content: str) -> str:
    """Scan every transcript line for the marker strings and emit one
    candidate atom per hit, regardless of which role said it -- the
    product-code trust boundary (not this stub) is what's supposed to
    decide whether an assistant-authored "global" claim is trustworthy."""
    atoms = []
    for raw_line in user_content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # transcript lines look like "USER: ..." / "ASSISTANT: ..."
        _, _, rest = line.partition(": ")
        text = rest or line
        if _INSTRUCTION_MARKER in text:
            content = text.split(_INSTRUCTION_MARKER, 1)[1].strip()
            if content:
                atoms.append(
                    {
                        "category": "instruction",
                        "content": content,
                        "provenance": "explicit_statement",
                        "confidence": 0.95,
                        "scope": "global",
                    }
                )
        if _FACT_MARKER in text:
            content = text.split(_FACT_MARKER, 1)[1].strip()
            if content:
                atoms.append(
                    {
                        "category": "fact",
                        "content": content,
                        "provenance": "explicit_statement",
                        "confidence": 0.9,
                        "scope": "task",
                    }
                )
    return json.dumps(atoms)


def _judge_reply() -> str:
    return json.dumps({"action": "add", "target_id": None, "relation": "related_to", "reason": "stub-judge"})


@dataclass
class _Control:
    lock: threading.Lock = field(default_factory=threading.Lock)
    chat_fail: str | None = None  # None | "500" | "slow"
    embeddings_fail: str | None = None  # None | "500"
    models_fail: str | None = None  # None | "404"
    chat_delay: float = 0.0
    mid_stream_reset: bool = False
    dim: int = 384

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "chat_fail": self.chat_fail,
                "embeddings_fail": self.embeddings_fail,
                "models_fail": self.models_fail,
                "chat_delay": self.chat_delay,
                "mid_stream_reset": self.mid_stream_reset,
            }

    def update(self, **kwargs) -> None:
        with self.lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def reset(self) -> None:
        self.update(
            chat_fail=None, embeddings_fail=None, models_fail=None, chat_delay=0.0, mid_stream_reset=False
        )


class StubOpenAI:
    """A background OpenAI-compatible stub. `.start()`/`.stop()`; `.url`
    is the `.../v1` base once started."""

    def __init__(self, dim: int = 384, port: int | None = None):
        self._control = _Control(dim=dim)
        self._requested_port = port
        self.port: int | None = port
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self.received_chat: list[dict] = []

    # -- control knobs used by tests -----------------------------------------

    def set_fail(self, **kwargs) -> None:
        """e.g. `stub.set_fail(chat_fail="500")`, `stub.set_fail(embeddings_fail="500")`."""
        self._control.update(**kwargs)

    def reset(self) -> None:
        self._control.reset()
        self.received_chat.clear()

    # -- ASGI app -------------------------------------------------------------

    def _build_app(self) -> Starlette:
        control = self._control

        async def embeddings(request: Request) -> Response:
            if control.embeddings_fail == "500":
                return JSONResponse({"error": "stub embeddings failure"}, status_code=500)
            body = json.loads(await request.body() or b"{}")
            texts = body.get("input") or []
            if isinstance(texts, str):
                texts = [texts]
            data = [{"embedding": _hash_embed(t, control.dim), "index": i} for i, t in enumerate(texts)]
            return JSONResponse({"data": data, "model": body.get("model", "stub-embed")})

        async def models(request: Request) -> Response:
            if control.models_fail == "404":
                return JSONResponse({"error": "model not found"}, status_code=404)
            return JSONResponse({"data": [{"id": "stub-llm", "object": "model"}], "object": "list"})

        async def chat_completions(request: Request) -> Response:
            body = json.loads(await request.body() or b"{}")
            self.received_chat.append(body)
            stream = bool(body.get("stream"))

            if control.chat_delay:
                await asyncio.sleep(control.chat_delay)

            if control.chat_fail == "500":
                payload = {"error": {"message": "stub chat failure"}}
                if stream:
                    # Most real OpenAI-compatible servers answer a non-2xx
                    # with a plain JSON body even for a streaming request --
                    # the memstrand proxy is expected to detect this status
                    # code before ever wrapping anything in an SSE response.
                    return JSONResponse(payload, status_code=500)
                return JSONResponse(payload, status_code=500)

            messages = body.get("messages", [])
            system = _system_content(messages)
            if system.strip() == EXTRACTION_SYSTEM_PROMPT.strip():
                reply = _extraction_reply(_last_user_content(messages))
            elif system.strip() == JUDGE_SYSTEM_PROMPT.strip():
                reply = _judge_reply()
            else:
                reply = f"ECHO_SYSTEM::{system}\nECHO_USER::{_last_user_content(messages)}"

            if not stream:
                return JSONResponse(
                    {
                        "id": "chatcmpl-stub",
                        "object": "chat.completion",
                        "model": body.get("model", "stub-llm"),
                        "choices": [
                            {"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}
                        ],
                    }
                )

            async def _gen():
                words = reply.split(" ")
                words = words or [reply]
                for i, w in enumerate(words):
                    if control.mid_stream_reset and i == max(1, len(words) // 2):
                        # Simulate the upstream connection dying mid-stream.
                        raise ConnectionResetError("stub: simulated mid-stream reset")
                    chunk = w if i == 0 else f" {w}"
                    event = {
                        "id": "chatcmpl-stub",
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": chunk}}],
                    }
                    yield f"data: {json.dumps(event)}\n\n".encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(_gen(), media_type="text/event-stream")

        async def control_get(request: Request) -> Response:
            return JSONResponse(control.snapshot())

        async def control_post(request: Request) -> Response:
            body = json.loads(await request.body() or b"{}")
            control.update(**body)
            return JSONResponse(control.snapshot())

        async def control_reset(request: Request) -> Response:
            control.reset()
            return JSONResponse(control.snapshot())

        return Starlette(
            routes=[
                Route("/v1/embeddings", embeddings, methods=["POST"]),
                Route("/v1/models", models, methods=["GET"]),
                Route("/v1/chat/completions", chat_completions, methods=["POST"]),
                Route("/_control", control_get, methods=["GET"]),
                Route("/_control", control_post, methods=["POST"]),
                Route("/_control/reset", control_reset, methods=["POST"]),
            ]
        )

    # -- lifecycle --------------------------------------------------------------

    def start(self, timeout: float = 10.0) -> StubOpenAI:
        port = self._requested_port if self._requested_port is not None else free_port()
        self.port = port
        config = uvicorn.Config(
            self._build_app(), host="127.0.0.1", port=port, log_level="warning", lifespan="off"
        )
        self._server = uvicorn.Server(config)

        def _run():
            asyncio.run(self._server.serve())

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if getattr(self._server, "started", False):
                return self
            time.sleep(0.02)
        raise TimeoutError(f"stub openai server never started on port {port}")

    def stop(self, timeout: float = 5.0) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    @property
    def url(self) -> str:
        """Base URL including `/v1`, e.g. for `MEMSTRAND_EMBED_BASE_URL`/
        `MEMSTRAND_LLM_BASE_URL`."""
        return f"http://127.0.0.1:{self.port}/v1"
