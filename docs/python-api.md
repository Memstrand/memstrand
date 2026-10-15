# Python API

```python
from memstrand import Memory, Category, Provenance

mem = Memory()                 # persistent: on-disk Qdrant + SQLite under MEMSTRAND_HOME
mem = Memory.in_memory()       # ephemeral: temp dir, in-memory Qdrant — for demos/tests

# write path
mem.remember(content, category=Category.fact, provenance=Provenance.explicit_statement,
             confidence=0.8, session_id=None, task_id=None, host=None, ttl_days=None,
             source=None, source_ref=None, agent=None)              # -> MemoryAtom
mem.log_turn(session_id, role, content, task_id=None)                # raw turn, cheap
mem.update(atom_id, content=None, category=None, confidence=None)    # -> MemoryAtom (supersedes if content changed)
mem.forget(atom_id)                                                  # -> bool
mem.forget_source(source, task_id=..., host=...)                     # -> int; default = every scope, exact-scope match if given

# scoped read path
mem.recall(query, session_id=None, task_id=None, host=None, categories=None,
           limit=10, include_superseded=False, track=True,
           min_relevance=None)       # -> list[dict]; defaults to MEMSTRAND_MIN_RELEVANCE; neighbors are scope-filtered too
mem.get(atom_id, task_id=None, host=None, scoped=False)              # -> dict | None; scoped=False (default) is the owner-facing browse view (all scopes)
mem.list(category=None, task_id=None, host=None, limit=100, offset=0)  # -> list[dict]
mem.count()                                                           # -> int
mem.stats()                                                           # -> dict

# context injection / grounded answering
mem.context(query=None, task_id=None, host=None, limit=8, max_chars=4000)  # -> str (Markdown block)
mem.grounded_answer(question, **recall_kwargs)                       # -> str (needs a local LLM)

# sessions / distillation / consolidation
mem.improve(session_ids=None, background=False)                      # -> dict
mem.consolidate(now=None)                                            # -> dict

# export / import / ingest
mem.export(fmt="jsonl")                                              # -> str ("jsonl" or "markdown")
mem.import_jsonl(text)                                                # -> dict
mem.ingest(path, task_id=None, force=False)                           # -> dict

mem.close()  # or: with Memory() as mem: ...
```

`memstrand.client.get_memory()` returns a `RemoteMemory` (same method
surface, talks to `memstrand serve` over REST) if a daemon is reachable,
else falls back to a local `Memory()` — the embedded stores are
single-writer, so this is how the CLI and hooks avoid corrupting a daemon's
state.

## Typed schema

```python
from memstrand import Category, Provenance

list(Category)      # instruction, fact, decision, goal, commitment, preference,
                     # relationship, context, event, learning, observation,
                     # artifact, error  (13 total)

list(Provenance)     # explicit_statement, inferred, corrected, validated,
                     # observed, imported
```

See `src/memstrand/schema.py` for the full `MemoryAtom`/`MemoryEdge`/
`RelationType` definitions, and [docs/architecture.md](architecture.md) for
how reads, writes, and consolidation use them.

## Swapping backends

`Memory(...)` accepts explicit `embedder=`, `vector_store=`, `graph_store=`,
and `session_store=` keyword arguments, which is how the test suite runs
everything in-process with a deterministic `HashEmbedder` and
`QdrantClient(":memory:")` instead of downloading an embedding model or
touching disk — see [CONTRIBUTING.md](../CONTRIBUTING.md) for the pattern
and how to add a new store backend.
