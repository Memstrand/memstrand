# memstrand

[![PyPI](https://img.shields.io/pypi/v/memstrand)](https://pypi.org/project/memstrand/)
[![Python](https://img.shields.io/pypi/pyversions/memstrand)](https://pypi.org/project/memstrand/)
[![License: MIT](https://img.shields.io/github/license/chiranjeevirajukr/memstrand)](LICENSE)

**Local-first typed memory for AI agents. Zero API keys.**

Two strands, like DNA: one strand holds **facts** (vectors), the other holds **relationships** (graph). Every write joins them base-pair by base-pair — a memory is never stored without its connections, and never retrieved without its scope.

## 30-second demo

No Docker. No server. No LLM. `pip install` is the entire setup:

```python
from memstrand import Category, Memory, Provenance

mem = Memory.in_memory()

# A fact scoped to one task...
mem.remember(
    "Playwright installed as an agent-testing task dependency",
    category=Category.fact,
    provenance=Provenance.explicit_statement,
    task_id="playwright-test",
)

# ...and a global preference
mem.remember(
    "Chiran prefers a natural, unstylized look in AI portraits",
    category=Category.preference,
    provenance=Provenance.explicit_statement,
    confidence=0.95,
)

# An unrelated session asks about testing setup.
# The Playwright fact does NOT leak across the task boundary:
for hit in mem.recall("testing setup"):
    print(hit["category"], "-", hit["content"])
# preference - Chiran prefers a natural, unstylized look in AI portraits

# Same query WITH the task scope sees it:
for hit in mem.recall("testing setup", task_id="playwright-test"):
    print(hit["category"], "-", hit["content"])
# fact - Playwright installed as an agent-testing task dependency
```

Typed. Scoped. Provenance on every atom. The hot path (`remember`/`recall`) never touches an LLM — embeddings are local ONNX, the graph is embedded Kuzu, sessions are SQLite.

## Why

Flat vector memory has three failure modes, and every agent hits all three:

1. **No types** — a preference, a decision, and a guess all look the same to cosine similarity.
2. **No scope** — a fact learned while testing Playwright leaks into an unrelated Mac disk-cleanup session. (This actually happened. It's why this exists.)
3. **No provenance** — nobody knows whether a memory was *said*, *inferred*, or *corrected*, or how confident the agent was.

memstrand fixes all three: every memory is **typed** (13 categories), **scoped** (session/task/host), and **carries provenance** (how it was learned, confidence, expiry).

## Persistent setup

`in_memory()` is ephemeral — data dies with the process. For real use:

```bash
docker compose up -d          # Qdrant (vectors); Kuzu + SQLite are embedded, zero setup
pip install memstrand
```

```python
mem = Memory()  # Docker Qdrant + on-disk Kuzu/SQLite
```

For the session-distillation pass (`mem.improve()`), point at any OpenAI-compatible local endpoint:

```bash
export MEMSTRAND_LLM_BASE_URL=http://localhost:8081/v1
```

## Architecture

Seven layers, deterministic write routing, scoped retrieval:

| Layer | What lives there | Scope |
|---|---|---|
| L0 working | live transcript + task/host scope; outranks everything | task |
| L1 session cache | fast raw Q&A, no extraction, TTL'd | session |
| L2 episodic | distilled session learnings | task + host |
| L3 semantic | typed facts, vector-embedded | global |
| L4 consolidated | curated cross-session synthesis | global |
| L5 graph | typed edges over everything above | global |
| L6 procedural | action sequences, workflow patterns, lessons | global |

**Write path** (the session-commit loop): during a session, writes land in L1 with zero extraction cost. At session end (or pre-compact hook), an async distillation pass extracts Memanto-typed atoms via your local LLM, dedups by content hash, checks for conflicts — contradictions become `contradicts` **edges**, never deletions — and promotes into L2/L3/L5.

**Read path:** scope filter *first* (session/task/host is a hard filter, not a ranking signal), then tiered abstracts before full content, graph traversal for multi-hop, and a `grounded_answer()` primitive that answers only from stored memory.

**Memory dynamics** (stolen fair and square, see below): per-fact FSRS strength with a review queue, Hebbian edge weights (use strengthens, disuse decays), and replay consolidation that prunes weak *edges* but never *facts*.

## How it compares

| | memstrand | Mem0 | Cognee | raw vector RAG |
|---|---|---|---|---|
| Local-first, no API keys | ✅ | OSS self-host option | ✅ | ✅ |
| Typed memory schema | ✅ 13 categories | ✅ notes | ✅ ontology | ❌ |
| Scope enforcement (hard filter) | ✅ | ❌ | ❌ | ❌ |
| Graph + vector joint writes | ✅ | ✅ | ✅ | ❌ |
| Contradictions as edges, not deletes | ✅ | ❌ | ✅ | ❌ |

## API

```python
mem.remember(content, category=..., provenance=..., confidence=..., ttl_days=...)
mem.recall(query, session_id=..., task_id=..., limit=...)   # scope-aware
mem.improve(session_ids=[...])    # distill sessions → permanent memory (needs local LLM)
mem.forget(memory_id)             # provenance-chain cleanup, never silent
mem.grounded_answer(question)     # answer ONLY from stored memories
```

MCP server included: `python -m memstrand.mcp_server` exposes the same surface to Claude Code, Cursor, and friends over stdio.

## Acceptance test

Any memory system should pass these three before you trust it (rubric adapted from a Cognee ad, because it was right):

1. **Remember me in a brand-new chat** without re-pasting anything.
2. **Search by meaning**, not keywords.
3. **Connect facts across sessions** — and *don't* leak facts across unrelated tasks (the Playwright test).

## Acknowledgments

Standing on shoulders, with credit:

- **Memanto** (moorcheh-ai/memanto) — the 13-category typed schema, provenance, and confidence model. Their product is cloud-only; this is the schema reimplemented local-first.
- **Cognee** (topoteretes/cognee) — the pipeline patterns: fast session cache → distillation, joint graph+vector writes, contradictions as additive edges.
- **ZenBrain** (zensation-ai/zenbrain) — per-fact FSRS, deterministic write routing, replay consolidation, Hebbian edges.
- **Mem0** — the ADD/UPDATE/DELETE/NONE extraction resolution pattern.

## Roadmap

- [x] Schema + store interfaces + scaffold
- [x] Session-commit distillation loop (extract → dedup → judge → promote)
- [x] Scoped retrieval (task/host hard filters; unscoped queries see globals only)
- [x] Graph neighbor hydration + Hebbian strengthen/decay
- [ ] FSRS + replay consolidation pass (math is in `pipeline/consolidate.py`, not yet scheduled)
- [x] MCP server tools (remember/recall/forget)
- [ ] `memstrand import` — adopt an existing Markdown memory file
- [x] Benchmarks: the 3-test rubric, automated (`tests/test_memory_flow.py`)

## License

MIT — see [LICENSE](LICENSE).
