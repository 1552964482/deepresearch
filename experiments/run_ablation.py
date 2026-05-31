"""End-to-end ablation driver.

Runs three configurations on ResearchBench and writes paired Cohen's d
comparison tables:

  A. baseline        — single-prompt mimo (no retrieval)
  B. pipeline-r0     — full DAG (Planner -> Search -> Read -> Compress -> Write)
  C. pipeline-r2     — full DAG + K=2 Red-Blue adversarial review

Output structure under ``experiments/results/<TIMESTAMP>-ablation/``:

    baseline/per_question.csv, summary.{md,json}, reports/*.md
    pipeline-r0/per_question.csv, summary.{md,json}, reports/*.md
    pipeline-r2/per_question.csv, summary.{md,json}, reports/*.md
    compare-pipeline-r0-vs-baseline.md
    compare-pipeline-r2-vs-pipeline-r0.md
    compare-pipeline-r2-vs-baseline.md
    ablation-summary.md
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# OpenMP workaround must precede torch import (transitively via embedder).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from loguru import logger

from dr_agent.config import get_settings
from dr_agent.eval.bench import load_researchbench
from dr_agent.eval.compare import compare, write_comparison_md
from dr_agent.eval.runner import (
    EvalRunner,
    write_csv,
    write_summary_json,
    write_summary_md,
)
from dr_agent.llm.judge import JudgeClient
from dr_agent.llm.pool import MimoPool
from dr_agent.memory.embedder import Embedder
from dr_agent.tools.fetcher import Fetcher
from dr_agent.tools.search import WebSearcher
from dr_agent.utils.logging import setup_logging


@dataclass
class ConfigSpec:
    label: str
    mode: str
    review_rounds: int


CONFIGS = [
    ConfigSpec("baseline", "baseline", 0),
    ConfigSpec("pipeline-r0", "pipeline", 0),
    ConfigSpec("pipeline-r2", "pipeline", 2),
]


async def run_one_config(
    cfg: ConfigSpec,
    *,
    pool: MimoPool,
    judge: JudgeClient,
    embedder: Embedder,
    web_searcher: WebSearcher,
    fetcher: Fetcher,
    bench,
    out_root: Path,
    n_judge_samples: int,
    concurrency: int,
) -> Path:
    cfg_dir = out_root / cfg.label
    cfg_dir.mkdir(parents=True, exist_ok=True)
    report_dir = cfg_dir / "reports"
    report_dir.mkdir(exist_ok=True)

    runner = EvalRunner(
        pool=pool,
        judge=judge,
        embedder=embedder,
        web_searcher=web_searcher,
        fetcher=fetcher,
        memory_db=cfg_dir / "memory.db",
        report_dir=report_dir,
        backend_name="mimo",
        mode=cfg.mode,  # type: ignore[arg-type]
        review_rounds=cfg.review_rounds,
        n_judge_samples=n_judge_samples,
    )
    t0 = time.monotonic()
    try:
        results, summary = await runner.run_all(bench, concurrency=concurrency)
    finally:
        runner.close()
    elapsed = time.monotonic() - t0
    logger.info(
        "config={} done in {:.1f}s ({} samples, {} failed)",
        cfg.label,
        elapsed,
        summary.n_samples,
        summary.n_failed,
    )

    write_csv(results, cfg_dir / "per_question.csv")
    write_summary_md(summary, cfg_dir / "summary.md")
    write_summary_json(summary, cfg_dir / "summary.json")
    return cfg_dir / "per_question.csv"


def write_ablation_summary(out_root: Path, csv_paths: dict[str, Path]) -> None:
    """Concise overview across all three configurations."""
    lines: list[str] = ["# Ablation Summary (ResearchBench)\n"]
    lines.append("| config | n | factual_acc | hallu | cite_cov | judge_overall |")
    lines.append("|---|---|---|---|---|---|")
    for label in ("baseline", "pipeline-r0", "pipeline-r2"):
        summary_path = out_root / label / "summary.json"
        if not summary_path.exists():
            lines.append(f"| {label} | n/a | — | — | — | — |")
            continue
        import json

        s = json.loads(summary_path.read_text(encoding="utf-8"))
        m = s["metrics_mean"]
        n = s["n_samples"]
        j = s.get("judge_overall_mean")
        lines.append(
            f"| {label} | {n} | {m['factual_accuracy']:.3f} | "
            f"{m['hallucination_rate']:.3f} | {m['citation_coverage']:.3f} | "
            + (f"{j:.3f}" if j is not None else "n/a")
            + " |"
        )
    lines.append("")
    lines.append("See `compare-*.md` for per-metric Cohen's d effect sizes.\n")
    (out_root / "ablation-summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


async def main_async(args: argparse.Namespace) -> None:
    setup_logging()
    settings = get_settings()
    bench = load_researchbench()
    if args.limit:
        bench = bench.filter(limit=args.limit)
    if args.ids:
        bench = bench.filter(ids=[i.strip() for i in args.ids.split(",")])
    if args.domain:
        bench = bench.filter(domain=args.domain)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.out) / f"{ts}-ablation"
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info("ablation root: {}", out_root)
    logger.info(
        "running {} configs × {} questions = {} tasks",
        len(CONFIGS),
        len(bench.questions),
        len(CONFIGS) * len(bench.questions),
    )

    embedder = Embedder()
    embedder.warmup()

    from dr_agent.tools.search_cache import SearchCache
    search_cache = SearchCache(Path(".cache/tavily.db"))

    csv_paths: dict[str, Path] = {}
    async with (
        MimoPool(settings.mimo) as pool,
        Fetcher() as fetcher,
    ):
        judge = JudgeClient(
            settings.judge,
            fallback_pool=pool,
            fallback_model=settings.mimo.model,
        )
        try:
            web_searcher = WebSearcher(cache=search_cache)
            for cfg in CONFIGS:
                if args.skip and cfg.label in args.skip.split(","):
                    logger.info("skipping config={} (per --skip)", cfg.label)
                    continue
                logger.info("=== running config: {} ===", cfg.label)
                csv_path = await run_one_config(
                    cfg,
                    pool=pool,
                    judge=judge,
                    embedder=embedder,
                    web_searcher=web_searcher,
                    fetcher=fetcher,
                    bench=bench,
                    out_root=out_root,
                    n_judge_samples=args.n_judge,
                    concurrency=args.concurrency,
                )
                csv_paths[cfg.label] = csv_path
        finally:
            await judge.aclose()

    logger.info("tavily-cache stats: {}", search_cache.stats())
    search_cache.close()

    # Pairwise comparisons.
    pairs = [
        ("baseline", "pipeline-r0"),
        ("pipeline-r0", "pipeline-r2"),
        ("baseline", "pipeline-r2"),
    ]
    for a, b in pairs:
        if a not in csv_paths or b not in csv_paths:
            continue
        rep = compare(
            csv_paths[a], csv_paths[b], label_a=a, label_b=b
        )
        write_comparison_md(rep, out_root / f"compare-{b}-vs-{a}.md")
        logger.info("wrote compare-{}-vs-{}.md", b, a)

    write_ablation_summary(out_root, csv_paths)
    logger.info("done. summary: {}", out_root / "ablation-summary.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 3-config ablation.")
    parser.add_argument(
        "--out", default="experiments/results", help="output root"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="cap number of questions per config"
    )
    parser.add_argument(
        "--ids", type=str, default=None, help="comma-separated question ids"
    )
    parser.add_argument(
        "--domain", type=str, default=None, help="filter to a single domain"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="parallel research-tasks; each one already uses 16 inner LLM concurrency",
    )
    parser.add_argument(
        "--n-judge", type=int, default=2, help="judge self-consistency samples"
    )
    parser.add_argument(
        "--skip", type=str, default="",
        help="comma-separated config labels to skip (baseline,pipeline-r0,pipeline-r2)",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
