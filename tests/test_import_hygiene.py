"""Import hygiene: the CLI/client/hooks/mcp_server import paths must not
pull in `qdrant_client` -> `fastembed` -> `onnxruntime`.

Before this existed, a bare `import memstrand.cli` (which happens on
*every* CLI invocation, including Claude Code hooks and remote-mode
commands that only ever talk HTTP to a running daemon) transitively
imported `.api` -> `qdrant_client` -> fastembed -> onnxruntime: ~400ms of
extra import time, and on some Linux CI runners onnxruntime's
hardware-probe logger writes a warning
(`[W:onnxruntime:Default, device_discovery.cc:95 GetPciBusId] ...`)
straight to stderr -- which broke hook tests that assert a hook prints
*nothing*. See the module docstrings in `__init__.py`/`cli.py`/
`client.py`/`mcp_server.py` for the fix (lazy `.api` access).

Every check here runs in a fresh subprocess (`sys.executable -c ...`)
rather than importing inside the test process itself: pytest's own
collection, or an earlier test module (e.g. `test_cli.py` imports
`memstrand.api` directly), may have already pulled `memstrand.api` and
friends into *this* process's `sys.modules`, which would make any
in-process check meaningless/flaky depending on test order.
"""

from __future__ import annotations

import json
import subprocess
import sys

# Anything in this list getting pulled onto the hook/remote-mode path is a
# regression: these are exactly the modules `.api` (and, transitively,
# `qdrant_client`'s own fastembed integration) drag in.
_HEAVY = (
    "qdrant_client",
    "fastembed",
    "onnxruntime",
    "memstrand.api",
    "memstrand.stores.vector_qdrant",
)


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _modules_present_after(import_line: str) -> dict[str, bool]:
    """Run `import_line` in a fresh subprocess, then report which of
    `_HEAVY` ended up in `sys.modules`. Also asserts the import itself
    produced no stdout/stderr -- onnxruntime's device-discovery warning
    would otherwise show up right here."""
    code = (
        f"{import_line}\n"
        "import json, sys\n"
        f"print(json.dumps({{m: (m in sys.modules) for m in {_HEAVY!r}}}))\n"
    )
    result = _run(code)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stderr == "", f"unexpected stderr (e.g. an onnxruntime warning): {result.stderr!r}"
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_import_cli_does_not_load_heavy_deps():
    present = _modules_present_after("import memstrand.cli")
    assert not any(present.values()), present


def test_import_client_does_not_load_heavy_deps():
    present = _modules_present_after("import memstrand.client")
    assert not any(present.values()), present


def test_import_hooks_does_not_load_heavy_deps():
    present = _modules_present_after("import memstrand.hooks")
    assert not any(present.values()), present


def test_import_mcp_server_does_not_load_heavy_deps():
    # Importing mcp_server (as `memstrand mcp` and the daemon's module
    # import both do before main()/build_mcp() ever runs) must not force
    # `.api` to load -- only actually opening a local Memory should.
    present = _modules_present_after("import memstrand.mcp_server")
    assert not any(present.values()), present


def test_plain_import_memstrand_does_not_load_heavy_deps():
    # `import memstrand` alone (without touching `Memory`/`StoreLocked`)
    # must stay cheap -- see the `__getattr__` in `__init__.py`.
    present = _modules_present_after("import memstrand")
    assert not any(present.values()), present


def test_from_memstrand_import_memory_still_works():
    result = _run(
        "from memstrand import Memory, StoreLocked\n"
        "print(Memory.__module__, Memory.__name__)\n"
        "print(StoreLocked.__module__, StoreLocked.__name__)\n"
        "import sys\n"
        "print('memstrand.api' in sys.modules)\n"
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines[0] == "memstrand.api Memory"
    assert lines[1] == "memstrand.api StoreLocked"
    # Accessing the lazy attribute is exactly what's allowed to load `.api`.
    assert lines[2] == "True"


def test_memstrand_attribute_access_does_not_cache_across_unrelated_names():
    result = _run(
        "import memstrand\n"
        "try:\n"
        "    memstrand.not_a_real_attribute\n"
        "except AttributeError as exc:\n"
        "    print('ok', exc)\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("ok")


def test_memstrand_dir_lists_memory_and_store_locked():
    result = _run(
        "import memstrand\n"
        "print(sorted(n for n in dir(memstrand) if n in ('Memory', 'StoreLocked')))\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "['Memory', 'StoreLocked']"


def test_cli_help_does_not_load_onnxruntime():
    code = (
        "import sys\n"
        "sys.argv = ['memstrand', '--help']\n"
        "from memstrand.cli import main\n"
        "try:\n"
        "    main()\n"
        "except SystemExit:\n"
        "    pass\n"
        "import json\n"
        f"print(json.dumps({{m: (m in sys.modules) for m in {_HEAVY!r}}}), file=sys.stderr)\n"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr
    # argparse's own --help text goes to stdout; our marker line is the
    # only thing on stderr -- if onnxruntime warned, it would be here too.
    present = json.loads(result.stderr.strip().splitlines()[-1])
    assert not any(present.values()), present


def test_hook_print_config_does_not_load_onnxruntime():
    code = (
        "import sys\n"
        "sys.argv = ['memstrand', 'hook', 'print-config']\n"
        "from memstrand.cli import main\n"
        "rc = main()\n"
        "assert rc == 0, rc\n"
        "import json\n"
        f"print(json.dumps({{m: (m in sys.modules) for m in {_HEAVY!r}}}), file=sys.stderr)\n"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr
    present = json.loads(result.stderr.strip().splitlines()[-1])
    assert not any(present.values()), present


def test_cli_import_time_is_modest():
    # Lenient threshold: this guards against a regression that drags
    # qdrant_client/fastembed back onto the hot path (which cost ~400ms
    # before this fix went in), not a tight budget that could flake on a
    # slow or loaded CI runner.
    result = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", "import memstrand.cli"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    line = next(ln for ln in result.stderr.splitlines() if ln.strip().endswith("| memstrand.cli"))
    # `import time:   self us | cumulative us | name`
    cumulative_us = int(line.split("|")[1].strip())
    assert cumulative_us < 250_000, f"memstrand.cli import took {cumulative_us}us: {line!r}"
