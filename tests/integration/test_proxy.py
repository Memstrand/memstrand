"""Scenario #7: the `/v1/chat/completions` memory-injecting proxy, driven
with raw `httpx` against a real daemon and a real (stub) upstream.

- Non-stream: the reply contains the injected memory block (the stub
  echoes the system prompt back), and exactly one user + one assistant
  turn get logged for the session.
- Stream: SSE chunks + a `[DONE]` sentinel, and exactly one new assistant
  turn gets logged once the stream completes.
- Upstream 500, both non-stream and `stream: true` -- the client must get
  a plain HTTP 500 JSON body, never a 200 wrapped in SSE framing (the
  proxy inspects the upstream status code before committing to a
  `StreamingResponse`; see server.py's `_v1_chat_completions`).
- Upstream unreachable (closed port) -- 502.
"""

from __future__ import annotations

import json

import httpx
import pytest

from .conftest import daemon_client_env, run_cli
from .stub_openai import free_port

pytestmark = pytest.mark.integration


def _turns(exported: str) -> list[dict]:
    out = []
    for line in exported.splitlines():
        obj = json.loads(line)
        if obj.get("type") == "turn":
            out.append(obj)
    return out


def test_non_stream_injects_memory_and_logs_turns_once(start_daemon, stub):
    daemon = start_daemon(stub=stub)
    env = daemon_client_env(daemon)

    result = run_cli(
        ["remember", "always mention that the meeting is at 3pm", "--category", "instruction"], env
    )
    assert result.returncode == 0, result.stderr

    resp = httpx.post(
        f"{daemon.url}/v1/chat/completions",
        json={
            "model": "stub-llm",
            "messages": [{"role": "user", "content": "what's on the agenda today?"}],
        },
        headers={"x-memstrand-session": "proxy-test-non-stream"},
        timeout=10.0,
    )
    assert resp.status_code == 200, resp.text
    reply = resp.json()["choices"][0]["message"]["content"]
    assert "the meeting is at 3pm" in reply
    assert "Memory (memstrand)" in reply

    export = run_cli(["export"], env)
    assert export.returncode == 0, export.stderr
    turns = [t for t in _turns(export.stdout) if t["session_id"] == "proxy-test-non-stream"]
    assert len(turns) == 2, turns
    roles = sorted(t["role"] for t in turns)
    assert roles == ["assistant", "user"]
    user_turn = next(t for t in turns if t["role"] == "user")
    assert user_turn["content"] == "what's on the agenda today?"


def test_stream_emits_sse_and_logs_one_assistant_turn(start_daemon, stub):
    daemon = start_daemon(stub=stub)
    env = daemon_client_env(daemon)

    with httpx.stream(
        "POST",
        f"{daemon.url}/v1/chat/completions",
        json={
            "model": "stub-llm",
            "stream": True,
            "messages": [{"role": "user", "content": "say something short"}],
        },
        headers={"x-memstrand-session": "proxy-test-stream"},
        timeout=10.0,
    ) as resp:
        assert resp.status_code == 200, resp.read()
        body = b"".join(resp.iter_bytes())

    text = body.decode()
    assert "data: " in text
    assert text.rstrip().endswith("data: [DONE]")

    export = run_cli(["export"], env)
    assert export.returncode == 0, export.stderr
    turns = [t for t in _turns(export.stdout) if t["session_id"] == "proxy-test-stream"]
    assert len(turns) == 2, turns
    assistant_turns = [t for t in turns if t["role"] == "assistant"]
    assert len(assistant_turns) == 1
    assert assistant_turns[0]["content"].strip() != ""


def test_upstream_500_is_not_wrapped_as_200_sse(start_daemon, stub):
    daemon = start_daemon(stub=stub)
    stub.set_fail(chat_fail="500")
    try:
        resp = httpx.post(
            f"{daemon.url}/v1/chat/completions",
            json={"model": "stub-llm", "messages": [{"role": "user", "content": "hi"}]},
            timeout=10.0,
        )
        assert resp.status_code == 500
        assert resp.headers.get("content-type", "").startswith("application/json")
        assert "stub chat failure" in resp.text

        with httpx.stream(
            "POST",
            f"{daemon.url}/v1/chat/completions",
            json={"model": "stub-llm", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
            timeout=10.0,
        ) as stream_resp:
            assert stream_resp.status_code == 500
            assert not stream_resp.headers.get("content-type", "").startswith("text/event-stream")
            body = b"".join(stream_resp.iter_bytes())
        assert b"stub chat failure" in body
    finally:
        stub.reset()


def test_upstream_unreachable_is_502(start_daemon, stub):
    closed_port = free_port()  # reserved, nothing listening
    daemon = start_daemon(
        embed_base_url=stub.url,
        llm_base_url=f"http://127.0.0.1:{closed_port}/v1",
    )
    resp = httpx.post(
        f"{daemon.url}/v1/chat/completions",
        json={"model": "stub-llm", "messages": [{"role": "user", "content": "hi"}]},
        timeout=10.0,
    )
    assert resp.status_code == 502
    assert "upstream unreachable" in resp.json()["error"]
