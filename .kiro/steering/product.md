# Product Context

DeepResearch-MultiAgent is a multi-agent system that produces grounded
research reports from a single user query. It is a portfolio / résumé
project, not a production system; readability of the codebase and
defensibility of the experimental claims take priority over scale.

## Audience

- **Recruiters / interviewers** scanning the GitHub README and benchmark
  numbers — every claim in `README.md` and `docs/ablation.md` must be
  reproducible from `experiments/results/`.
- **Hiring managers / engineers** opening the source — favor explicit
  module boundaries and docstrings over clever abstractions.

## Backend choices (current)

- **Primary LLM**: `mimo-v2.5-pro` (xiaomi). Four API keys, 100 RPM each,
  safe concurrency 4 per key → global concurrency 16.
- **Judge LLM**: `gpt-5.4` via `aveve.xyz` (independent model family).
  When the endpoint is unreachable, automatically fall back to mimo and
  label every score with `self_bias_risk=True` and
  `backend="mimo-fallback"`.
- **Embedder**: local `BAAI/bge-small-zh-v1.5` (95 MB, 512-D, normalized).
- **Web search**: Tavily (free tier 1000/month) with a SQLite cache.
