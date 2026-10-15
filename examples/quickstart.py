"""memstrand quickstart: the 3-test acceptance rubric, against a real
persistent store (not `Memory.in_memory()` — see examples/second_brain.md
or the README's 30-second demo for the zero-setup version).

1. remember in one "session"
2. recall by meaning in another session (same task) — sessions never
   isolate semantic recall; that's the point of the semantic layer
3. connect facts across sessions — without leaking across tasks

Scoping rule: task_id / host are HARD filters. An unscoped query only
ever sees global atoms (no task/host) PLUS, with a scope set, its own
scope. A query scoped to task X sees task-X atoms plus globals, and
nothing from other tasks. Session is provenance, not a filter.

`Memory()` with no arguments is the persistent default: an embedded,
on-disk Qdrant plus SQLite under `$MEMSTRAND_HOME` (default `~/.memstrand`,
see `memstrand config`). The embedder (fastembed by default) downloads its
model from Hugging Face on first use — if you're offline and have never
run memstrand before, this script will fail on the first `remember()`
until you either get a connection once, or point MEMSTRAND_EMBED_BACKEND
at a local OpenAI-compatible embeddings endpoint (see
examples/second_brain.md). `remember()` is idempotent per (content, scope),
so re-running this script is safe.
"""

from memstrand import Category, Memory, Provenance

with Memory() as mem:
    # session 1: learn something
    mem.remember(
        "Chiran's mini PC is named cognita-pulse",
        category=Category.fact,
        provenance=Provenance.explicit_statement,
        confidence=0.95,
        session_id="s1",
        task_id="homelab",
    )

    # session 2 (new chat, same task): recall by meaning
    hits = mem.recall("what is the mini PC called?", session_id="s2", task_id="homelab")
    print("test 2 (semantic recall):", hits)
    assert any(
        "cognita-pulse" in (h["content"] or "") for h in hits
    ), "REGRESSION: same-task cross-session recall broke"

    # unrelated task must NOT see task-scoped memories
    mem.remember(
        "Playwright 1.63 installed for agent testing",
        category=Category.fact,
        task_id="agent-testing",
    )
    leaked = mem.recall("playwright", task_id="disk-cleanup")
    print("test 3 (no cross-task leak):", leaked)
    assert leaked == [], "SCOPE BUG: task-scoped memory leaked!"

    # unscoped queries never see task-scoped atoms (the real Playwright fix)
    leaked_unscoped = mem.recall("playwright")
    print("test 3b (unscoped query):", leaked_unscoped)
    assert leaked_unscoped == [], "SCOPE BUG: unscoped query saw task-scoped memory!"

    print("all three tests pass")
