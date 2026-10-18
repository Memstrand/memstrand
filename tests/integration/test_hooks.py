"""Scenario #4: the Claude Code hook commands (`memstrand hook claude-code`
and `memstrand hook claude-code-context`), run as real CLI subprocesses fed
a realistic hook stdin JSON and a realistic Claude Code transcript JSONL.

Covers: only `{"type": "text"}` blocks count as loggable turns (tool_use/
tool_result/thinking blocks and `isMeta` entries are excluded); incremental
re-runs only log what's new since the last watermark; a token-rejected
daemon logs nothing but still exits 0 with empty stdout, recording the
rejection reason to `hook.log`; the context hook prints a remembered
global preference."""

from __future__ import annotations

import json

import pytest

from .conftest import daemon_client_env, run_cli

pytestmark = pytest.mark.integration


def _write_transcript(path, lines: list[dict]) -> None:
    with path.open("w") as f:
        for entry in lines:
            f.write(json.dumps(entry) + "\n")


def _turn_lines(exported: str) -> list[dict]:
    out = []
    for line in exported.splitlines():
        obj = json.loads(line)
        if obj.get("type") == "turn":
            out.append(obj)
    return out


_BASE_TRANSCRIPT = [
    {"type": "user", "message": {"role": "user", "content": "remember to buy more coffee"}},
    {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "I should acknowledge this."},
                {"type": "text", "text": "Noted, I'll remember that for you."},
            ],
        },
    },
    {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}],
        },
    },
    {
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "file1\nfile2"}],
        },
    },
    {
        "type": "user",
        "isMeta": True,
        "message": {"role": "user", "content": "this is internal bookkeeping, not a real turn"},
    },
    {"type": "summary", "summary": "unrelated entry type, not user/assistant"},
]


def test_claude_code_hook_logs_only_text_turns_and_is_incremental(start_daemon, stub, tmp_path):
    daemon = start_daemon(stub=stub)
    env = daemon_client_env(daemon)

    transcript_path = tmp_path / "transcript.jsonl"
    _write_transcript(transcript_path, _BASE_TRANSCRIPT)

    stdin_json = {
        "session_id": "sess-abc",
        "transcript_path": str(transcript_path),
        "cwd": str(tmp_path),
        "hook_event_name": "Stop",
    }

    result = run_cli(
        ["hook", "claude-code", "--no-improve", "--task-from", "none"],
        env,
        input=json.dumps(stdin_json),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == ""
    assert result.stderr == ""

    export = run_cli(["export"], env)
    assert export.returncode == 0, export.stderr
    turns = _turn_lines(export.stdout)
    assert len(turns) == 2, turns
    contents = {t["content"] for t in turns}
    assert "remember to buy more coffee" in contents
    assert "Noted, I'll remember that for you." in contents

    # Append a new transcript line and re-run: only the new one gets logged.
    with transcript_path.open("a") as f:
        f.write(json.dumps({"type": "user", "message": {"role": "user", "content": "one more thing"}}) + "\n")

    result = run_cli(
        ["hook", "claude-code", "--no-improve", "--task-from", "none"],
        env,
        input=json.dumps(stdin_json),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == ""

    export = run_cli(["export"], env)
    assert export.returncode == 0, export.stderr
    turns = _turn_lines(export.stdout)
    assert len(turns) == 3, turns
    contents = {t["content"] for t in turns}
    assert "one more thing" in contents


def test_claude_code_hook_rejected_daemon_logs_nothing_but_exits_clean(start_daemon, stub, tmp_path):
    daemon = start_daemon(stub=stub, token="s3cret-token")
    env = daemon_client_env(daemon, token=None)

    transcript_path = tmp_path / "transcript.jsonl"
    _write_transcript(transcript_path, _BASE_TRANSCRIPT)

    stdin_json = {
        "session_id": "sess-rejected",
        "transcript_path": str(transcript_path),
        "cwd": str(tmp_path),
        "hook_event_name": "Stop",
    }

    result = run_cli(
        ["hook", "claude-code", "--no-improve", "--task-from", "none"],
        env,
        input=json.dumps(stdin_json),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == ""

    hook_log = daemon.home / "hook.log"
    assert hook_log.exists(), "hook.log must exist after a rejected-daemon run"
    log_text = hook_log.read_text()
    assert (
        "daemon requires MEMSTRAND_TOKEN which this hook does not have — "
        "turns NOT logged; they will be picked up on the next run once fixed"
    ) in log_text

    # No turns were logged -- verify with the token this time.
    good_env = daemon_client_env(daemon, token=True)
    export = run_cli(["export"], good_env)
    assert export.returncode == 0, export.stderr
    assert _turn_lines(export.stdout) == []


def test_claude_code_context_hook_prints_remembered_preference(start_daemon, stub, tmp_path):
    daemon = start_daemon(stub=stub)
    env = daemon_client_env(daemon)

    result = run_cli(
        ["remember", "always sign off messages with -The Memstrand Bot", "--category", "preference"], env
    )
    assert result.returncode == 0, result.stderr

    stdin_json = {
        "prompt": "how should I reply to this email?",
        "cwd": str(tmp_path),
        "hook_event_name": "UserPromptSubmit",
    }
    result = run_cli(
        ["hook", "claude-code-context", "--task-from", "none"],
        env,
        input=json.dumps(stdin_json),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "The Memstrand Bot" in result.stdout
