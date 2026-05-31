"""dr-agent CLI."""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path

# Anaconda + torch OpenMP-conflict workaround. Must be set BEFORE torch import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import typer
from rich.console import Console
from rich.table import Table

from dr_agent.config import get_settings
from dr_agent.llm.judge import JudgeClient
from dr_agent.llm.pool import MimoPool
from dr_agent.orchestrator.runner import run_grounded, run_minimal
from dr_agent.orchestrator.state_machine import StateMachine
from dr_agent.utils.logging import setup_logging

app = typer.Typer(
    add_completion=False,
    help="DeepResearch-MultiAgent CLI",
    no_args_is_help=True,
    pretty_exceptions_enable=False,  # don't expand huge dataclass tracebacks
)
console = Console()


def _slugify(s: str, n: int = 40) -> str:
    s = re.sub(r"\s+", "-", s.strip())
    s = re.sub(r"[^\w\u4e00-\u9fa5\-]", "", s)
    return s[:n] or "report"


@app.command("run")
def cmd_run(
    query: str = typer.Argument(..., help="Research query"),
    output_dir: Path = typer.Option(
        Path("reports"), "--out", "-o", help="Output directory"
    ),
    grounded: bool = typer.Option(
        True,
        "--grounded/--no-grounded",
        help="Enable web retrieval + memory + compression (Phase 2). "
        "Use --no-grounded for the Phase-1 knowledge-only path.",
    ),
    review: int = typer.Option(
        0,
        "--review",
        help="Number of adversarial review rounds (Phase 3). "
        "0 = disabled, 2 = recommended.",
    ),
    reviewer: str = typer.Option(
        "red",
        "--reviewer",
        help="Reviewer mode: 'red' (single 4-dim agent) or "
        "'multi' (3 persona critics + consensus merge).",
    ),
    depth: int = typer.Option(
        0,
        "--depth",
        help="Recursive deep-dive depth (0 = single layer / disabled). "
        "Each level follows up on knowledge gaps per SubTask.",
    ),
    breadth: int = typer.Option(
        2,
        "--breadth",
        help="Max follow-up sub-questions per SubTask per deep-dive level.",
    ),
    deepdive_budget: int = typer.Option(
        24,
        "--deepdive-budget",
        help="Global hard cap on extra Tavily searches during deep-dive.",
    ),
    in_loop_judge: bool = typer.Option(
        False,
        "--in-loop-judge",
        help="Enable Judge-based quality rollback inside the review loop "
        "(burns judge tokens; off by default).",
    ),
    max_results: int = typer.Option(
        5, "--max-results", help="Tavily results per subtask query"
    ),
    db_path: Path = typer.Option(
        Path(".cache/memory.db"), "--db", help="SQLite memory store path"
    ),
) -> None:
    """Run the research pipeline end-to-end."""
    setup_logging()
    settings = get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)

    async def _go() -> None:
        async with MimoPool(settings.mimo) as pool:
            if grounded:
                from dr_agent.memory.embedder import Embedder
                from dr_agent.memory.store import MemoryStore
                from dr_agent.tools.fetcher import Fetcher
                from dr_agent.tools.search import WebSearcher
                from dr_agent.tools.search_cache import SearchCache

                embedder = Embedder()
                embedder.warmup()
                memory = MemoryStore(db_path, embedder)
                judge_client: JudgeClient | None = None
                rb_result = None
                search_cache = SearchCache(Path(".cache/tavily.db"))
                try:
                    async with Fetcher() as fetcher:
                        searcher = WebSearcher(cache=search_cache)
                        if review > 0 and in_loop_judge:
                            judge_client = JudgeClient(settings.judge)
                        report, sm, rb_result = await run_grounded(
                            query,
                            pool,
                            embedder=embedder,
                            memory=memory,
                            web_searcher=searcher,
                            fetcher=fetcher,
                            config=settings.orch,
                            max_results_per_query=max_results,
                            review_rounds=review,
                            judge=judge_client,
                            in_loop_judge=in_loop_judge,
                            reviewer=reviewer,
                            depth=depth,
                            breadth=breadth,
                            deepdive_max_searches=deepdive_budget,
                        )
                finally:
                    if judge_client is not None:
                        await judge_client.aclose()
                console.print(
                    f"[dim]memory items after run: {memory.stats()['items']}[/dim]"
                )
                console.print(
                    f"[dim]tavily-cache: {search_cache.stats()}[/dim]"
                )
                memory.close()
                search_cache.close()
            else:
                report, sm = await run_minimal(query, pool, config=settings.orch)
                rb_result = None
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = _slugify(query, 40)
        out_path = output_dir / f"{ts}-{slug}.md"
        out_path.write_text(report.to_markdown(), encoding="utf-8")
        console.print(f"\n[bold green][OK] Report saved:[/] {out_path}")
        console.print(
            "[dim]state-history: "
            + " -> ".join(
                [sm.history[0].from_state.value]
                + [r.to_state.value for r in sm.history]
            )
            + "[/dim]"
        )
        if rb_result is not None and rb_result.rounds:
            console.print("\n[bold]Red-Blue rounds:[/]")
            for rs in rb_result.rounds:
                console.print(
                    f"  round {rs.round_idx}: attacks={rs.n_attacks} "
                    f"(F={rs.n_attacks_factual}, L={rs.n_attacks_logic}, "
                    f"C={rs.n_attacks_citation}), patches_accepted="
                    f"{rs.n_patches_accepted}, parse={rs.parse_strategy}"
                )
            if rb_result.rolled_back:
                console.print("[yellow]rolled back due to quality drop[/]")
            console.print(f"[dim]stop_reason: {rb_result.stop_reason}[/]")
        console.print("\n[bold]Pool stats:[/]")
        console.print_json(json.dumps(pool.stats(), ensure_ascii=False))

    asyncio.run(_go())


@app.command("state-graph")
def cmd_state_graph() -> None:
    """Print the state-machine as a mermaid diagram."""
    typer.echo(StateMachine.export_mermaid())


@app.command("pool-stats")
def cmd_pool_stats() -> None:
    """Show MimoPool config (live stats require a running pool)."""
    setup_logging()
    settings = get_settings()
    table = Table(title="MimoPool Configuration")
    table.add_column("field")
    table.add_column("value")
    table.add_row("model", settings.mimo.model)
    table.add_row("base_url", settings.mimo.base_url)
    table.add_row("num_keys", str(len(settings.mimo.api_keys)))
    table.add_row("rpm_per_key", str(settings.mimo.rpm_per_key))
    table.add_row("safe_concurrency_per_key", str(settings.mimo.safe_concurrency_per_key))
    table.add_row("total_concurrency", str(settings.mimo.total_concurrency))
    console.print(table)


@app.command("smoke-judge")
def cmd_smoke_judge(
    question: str = typer.Argument(..., help="Question for the judge"),
    report_path: Path = typer.Argument(..., help="Path to a markdown report"),
    n: int = typer.Option(2, "--n", help="Judge sample count"),
) -> None:
    """Run the JudgeClient against a saved report (smoke test)."""
    setup_logging()
    settings = get_settings()
    report_text = report_path.read_text(encoding="utf-8")

    async def _go() -> None:
        async with JudgeClient(settings.judge) as judge:
            score = await judge.score(
                question=question, report=report_text, n_samples=n
            )
        console.print_json(score.model_dump_json(indent=2))

    asyncio.run(_go())


@app.command("eval")
def cmd_eval(
    bench: str = typer.Option(
        "researchbench", "--bench", help="Benchmark name (researchbench)"
    ),
    mode: str = typer.Option(
        "pipeline", "--mode",
        help="pipeline = full DAG; baseline = single-prompt knowledge-only",
    ),
    review: int = typer.Option(
        0, "--review", help="Red-Blue rounds in pipeline mode (0 disables)"
    ),
    backend: str = typer.Option(
        "mimo", "--backend",
        help="LLM backend label (used in CSV/output naming, currently only "
        "'mimo' is wired since other backends require external endpoints)",
    ),
    domain: str | None = typer.Option(
        None, "--domain", help="Filter to a single domain"
    ),
    ids: str | None = typer.Option(
        None, "--ids",
        help="Comma-separated list of question IDs to evaluate (e.g. rb-001,rb-005)",
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Cap on number of questions"
    ),
    concurrency: int = typer.Option(
        2, "--concurrency",
        help="Parallel research-task concurrency (each task already uses 16 inner LLM concurrency)",
    ),
    n_judge_samples: int = typer.Option(
        3, "--n-judge", help="Judge self-consistency sample count"
    ),
    out_dir: Path = typer.Option(
        Path("experiments/results"), "--out", help="Results root dir"
    ),
) -> None:
    """Run benchmark evaluation: rule metrics + LLM-as-Judge + bootstrap CI."""
    from dr_agent.eval.bench import load_researchbench
    from dr_agent.eval.runner import (
        EvalRunner,
        write_csv,
        write_summary_json,
        write_summary_md,
    )
    from dr_agent.memory.embedder import Embedder
    from dr_agent.tools.fetcher import Fetcher
    from dr_agent.tools.search import WebSearcher

    setup_logging()
    settings = get_settings()

    if bench != "researchbench":
        raise typer.BadParameter("only 'researchbench' is bundled in this version")

    bench_obj = load_researchbench()
    bench_obj = bench_obj.filter(
        domain=domain,
        ids=[i.strip() for i in ids.split(",")] if ids else None,
        limit=limit,
    )
    console.print(
        f"[bold]ResearchBench loaded:[/] {len(bench_obj.questions)} questions "
        f"(mode={mode}, backend={backend}, review={review})"
    )

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = out_dir / f"{ts}-{backend}-{mode}-r{review}"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_dir = run_dir / "reports"
    report_dir.mkdir(exist_ok=True)

    async def _go() -> None:
        embedder = Embedder()
        embedder.warmup()
        from dr_agent.tools.search_cache import SearchCache
        search_cache = SearchCache(Path(".cache/tavily.db"))
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
                runner = EvalRunner(
                    pool=pool,
                    judge=judge,
                    embedder=embedder,
                    web_searcher=web_searcher,
                    fetcher=fetcher,
                    memory_db=run_dir / "memory.db",
                    report_dir=report_dir,
                    backend_name=backend,
                    mode=mode,  # type: ignore[arg-type]
                    review_rounds=review,
                    n_judge_samples=n_judge_samples,
                )
                try:
                    results, summary = await runner.run_all(
                        bench_obj, concurrency=concurrency
                    )
                finally:
                    runner.close()
            finally:
                await judge.aclose()
        console.print(f"[dim]tavily-cache: {search_cache.stats()}[/dim]")
        search_cache.close()

        write_csv(results, run_dir / "per_question.csv")
        write_summary_md(summary, run_dir / "summary.md")
        write_summary_json(summary, run_dir / "summary.json")

        console.print(f"\n[bold green][OK] Eval saved:[/] {run_dir}")
        console.print(f"  - per-question CSV: per_question.csv ({len(results)} rows)")
        console.print(
            f"  - summary: factual_acc={summary.metrics_mean['factual_accuracy']:.3f} "
            f"hallu={summary.metrics_mean['hallucination_rate']:.3f} "
            f"cite_cov={summary.metrics_mean['citation_coverage']:.3f}"
            + (
                f" judge={summary.judge_overall_mean:.3f}"
                if summary.judge_overall_mean is not None
                else ""
            )
        )

    asyncio.run(_go())


@app.command("compare")
def cmd_compare(
    runs: list[str] = typer.Argument(
        ..., help="Pairs like 'path/to/run:Label'. First run is the baseline."
    ),
    out: Path = typer.Option(
        Path("experiments/ablation.md"), "--out", help="Markdown output path"
    ),
) -> None:
    """Produce an ablation comparison table from multiple eval runs."""
    from dr_agent.eval.compare import compare as _compare

    parsed: list[tuple[Path, str]] = []
    for r in runs:
        if ":" in r:
            path_s, label = r.rsplit(":", 1)
        else:
            path_s, label = r, Path(r).name
        p = Path(path_s)
        if p.is_dir():
            p = p / "per_question.csv"
        if not p.exists():
            raise typer.BadParameter(f"per_question.csv not found at {p}")
        parsed.append((p, label))
    md = _compare(parsed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    console.print(f"[bold green][OK] Comparison saved:[/] {out}")
    console.print(md)


if __name__ == "__main__":
    app()
