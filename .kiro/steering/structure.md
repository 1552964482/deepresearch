# Project Structure

```
src/dr_agent/
├── llm/         MimoPool, JudgeClient, token_bucket, errors
├── schemas/     Pydantic models (task / report / attack / patch)
├── orchestrator/   state_machine, envelope, runner, red_blue_loop
├── agents/      base, planner, searcher, reader, writer, red, blue
├── memory/      embedder, store, compress
├── tools/       search, search_cache, fetcher
├── eval/        bench, rule_metrics, judge_metrics, stats, compare, runner
└── utils/       logging, trace
```

## Module boundaries to keep clean

- **`agents/` only depends on `llm/`, `schemas/`, `memory/`, `tools/`,
  `orchestrator/`** — never on `eval/`. Eval is a consumer of agents,
  never the other way around.
- **`memory/` and `tools/` must not import each other.** They're peer
  resources; the orchestrator wires them together.
- **`schemas/` has no internal imports** beyond stdlib + pydantic. It is
  the leaf of the dependency tree.
- **`orchestrator/runner.py` is the only public entry point** for an
  end-to-end research pipeline. CLI and Streamlit both call into it.

## Conventions

- All async; no sync I/O on the hot path. CPU-heavy work (TextRank,
  numpy cosine) goes through `asyncio.to_thread` only when necessary.
- Public Agent classes inherit `AbstractAgent[InT, OutT]` and return
  `ResultEnvelope[OutT]`. Exceptions never propagate across DAG branches.
- LLM calls go through `MimoPool` (main flow) or `JudgeClient` (eval).
  No direct `httpx.AsyncClient` to OpenAI-style endpoints elsewhere.
- Logging uses `loguru` and the `_mask_filter` to redact `tp-*` / `sk-*`
  tokens. Never `print()` API keys.
- Persisted artifacts go under `experiments/results/<TS>-*` (eval) or
  `reports/` (single runs). Both directories are gitignored.

## Test placement

- One `tests/test_<module>.py` per public module under test.
- Unit tests must be offline: mock `httpx.AsyncClient` for pool tests,
  use a deterministic fake embedder for compress / memory tests.
- E2E or integration tests that hit real mimo/Tavily are **not** in
  `tests/`; they live in `experiments/` as scripts.
