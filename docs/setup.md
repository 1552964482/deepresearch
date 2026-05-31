# Setup & Reproducibility Guide

## Prerequisites

- Python ≥ 3.11 (tested on Anaconda Python 3.12 + Windows)
- ~1 GB free disk for embedder + caches
- Internet for first run (Hugging Face downloads `bge-small-zh-v1.5`,
  Tavily search calls, mimo / aveve endpoints)
- Optional: NVIDIA GPU (CUDA-enabled `torch`) for ~6× faster embedding

## Install

```bash
git clone https://github.com/1552964482/deepresearch.git
cd deepresearch

pip install -e .
pip install \
    sentence-transformers tavily-python trafilatura pymupdf \
    scikit-learn jieba sumy networkx tiktoken
pip install streamlit          # optional, demo UI
pip install pytest pytest-asyncio respx   # tests
```

If `import jieba` fails with `dict.txt missing`, force-reinstall:

```bash
pip install --force-reinstall --no-deps jieba
```

## Configure

Copy `.env.example` to `.env` and fill in your keys. The runtime needs:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY1` … `OPENAI_API_KEY4` | mimo keys (load-balanced) |
| `OPENAI_BASE_URL` | mimo OpenAI-compatible endpoint |
| `OPENAI_MODEL` | mimo model name (e.g. `mimo-v2.5-pro`) |
| `JUDGE_API_KEY` / `JUDGE_BASE_URL` / `JUDGE_MODEL` | independent judge endpoint |
| `TAVILY_API_KEY` | web search |

`.env` is gitignored — never commit it.

## Run

### Single research report

```bash
# Phase-1 hello-world (no retrieval)
dr-agent run "什么是 GRPO 算法及其相对 PPO 的核心改进" --no-grounded

# Full pipeline (Phase 2)
dr-agent run "什么是 GRPO 算法及其相对 PPO 的核心改进"

# Full pipeline + K=2 Red-Blue review
dr-agent run "什么是 GRPO 算法及其相对 PPO 的核心改进" --review 2
```

Reports land under `reports/<timestamp>-<slug>.md`.

### Benchmark evaluation

```bash
# Single config (35 questions, ~10 min)
dr-agent eval --bench researchbench --mode pipeline --review 0

# Full ablation (3 configs × 35 = 105 tasks, ~70 min)
python experiments/run_ablation.py --concurrency 3 --n-judge 2

# Cheap rerun: only Red-Blue, reusing saved r0 reports (zero Tavily)
python experiments/rerun_red_blue.py \
  --src experiments/results/<TS>-ablation/pipeline-r0/reports \
  --review 2 \
  --compare-with experiments/results/<TS>-ablation/pipeline-r0/per_question.csv
```

Each run writes a timestamped folder under `experiments/results/`:

```
<TS>-ablation/
├── ablation-summary.md
├── compare-pipeline-r0-vs-baseline.md
├── compare-pipeline-r2-vs-pipeline-r0.md
├── compare-pipeline-r2-vs-baseline.md
└── <config>/
    ├── per_question.csv
    ├── summary.md
    ├── summary.json
    └── reports/<qid>-mimo-<mode>.md
```

### Streamlit demo

```bash
streamlit run ui/app.py
```

Two views:

- **Live Run** — submit a query, watch state transitions, Red-Blue
  rounds, pool stats, and the final markdown.
- **Eval Browser** — pick any `experiments/results/*-ablation` folder,
  see summary tables and Cohen's d comparisons.

## Tests

```bash
pytest tests/                          # 65 unit tests, fully offline
pytest tests/ -k state_machine -v      # one module
pytest tests/ --cov=src/dr_agent       # with coverage
```

## Common pitfalls

- **OpenMP error on Windows + Anaconda**: set
  `KMP_DUPLICATE_LIB_OK=TRUE` before any torch import. The CLI / experiments scripts already do; if you write a new entry point, set it
  there too.
- **Tavily 429 / quota exceeded**: free tier is 1000 searches/month.
  Use `experiments/rerun_red_blue.py` or set
  `TAVILY_API_KEY` to a paid account.
- **Judge endpoint 502**: the JudgeClient automatically falls back to
  mimo and tags every score with `self_bias_risk=True`. The eval
  summary headers will note this.
- **`Cannot copy out of meta tensor`** during embedder load: triggered
  when `SentenceTransformer` is first invoked from a worker thread.
  Always call `embedder.warmup()` from the main thread before launching
  background tasks (CLI and runner already do).
