"""Re-run Red-Blue review on existing pipeline-r0 reports.

This script avoids further Tavily quota usage by reusing the markdown
reports produced during a previous pipeline-r0 ablation run. Each report
is re-loaded into a ``ResearchReport`` (best-effort markdown parsing),
fed through the Red-Blue loop (K=2), then re-evaluated with rule metrics
and the LLM-as-Judge.

Use this to validate prompt / invariant changes in the Red-Blue loop
without re-running Planner/Searcher/Reader.

Usage:
    python experiments/rerun_red_blue.py \\
      --src experiments/results/<TS>-ablation/pipeline-r0/reports \\
      --bench-ids-from experiments/results/<TS>-ablation/pipeline-r0/per_question.csv \\
      --review 2 \\
      --concurrency 3
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from loguru import logger

from dr_agent.agents.base import AgentContext
from dr_agent.config import get_settings
from dr_agent.eval.bench import BenchQuestion, load_researchbench
from dr_agent.eval.compare import compare, write_comparison_md
from dr_agent.eval.rule_metrics import compute_rule_metrics
from dr_agent.eval.runner import (
    SampleResult,
    write_csv,
    write_summary_json,
    write_summary_md,
)
from dr_agent.eval.stats import bootstrap_ci
from dr_agent.eval.runner import EvalSummary
from dr_agent.llm.judge import JudgeClient, JudgeScore
from dr_agent.llm.pool import MimoPool
from dr_agent.memory.embedder import Embedder
from dr_agent.orchestrator.red_blue_loop import run_red_blue_loop
from dr_agent.orchestrator.state_machine import StateMachine
from dr_agent.schemas.report import Citation, ResearchReport, Section
from dr_agent.utils.logging import setup_logging


# ---------- markdown -> ResearchReport ----------


_HEAD2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_REF_LINE_RE = re.compile(
    r"^- \[(?P<title>[^\]]+)\]\((?P<url>[^)]+)\)(?:\s+—\s+(?P<snippet>.*))?$"
)


def _parse_markdown_report(md: str, *, task_id: str, query: str) -> ResearchReport:
    """Best-effort reconstruction of ResearchReport from its markdown form.

    Rather than try to perfectly recover the original section boundaries
    (LLM-written reports often embed nested ``##`` headings copied from
    source pages), we collapse the entire body into a single Section.
    Red-Blue only requires section_id stability for patch routing, so a
    single section is fine for rerun purposes.
    """
    title = "Recovered report"
    title_m = re.search(r"^#\s+(.+?)\s*$", md, re.MULTILINE)
    if title_m:
        title = title_m.group(1).strip()

    # Strip the leading "# Title" + meta blockquote lines
    lines = md.splitlines()
    body_lines: list[str] = []
    skip_meta = True
    for line in lines:
        stripped = line.strip()
        if skip_meta:
            if stripped.startswith("# "):
                continue
            if stripped.startswith(">"):
                continue
            if not stripped:
                continue
            skip_meta = False
        body_lines.append(line)
    body_text = "\n".join(body_lines)

    # Split off "## All References" (case-insensitive) into citations.
    citations: list[Citation] = []
    refs_split = re.split(
        r"^##\s+(?:All\s+References|References)\s*$",
        body_text,
        maxsplit=1,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    main_body = refs_split[0].strip()
    if len(refs_split) > 1:
        for line in refs_split[1].splitlines():
            line = line.strip()
            rm = _REF_LINE_RE.match(line)
            if rm:
                citations.append(
                    Citation(
                        id=f"c-{len(citations)}",
                        title=rm.group("title"),
                        url=rm.group("url"),
                        snippet=(rm.group("snippet") or "").strip(),
                    )
                )

    # Optionally pull "## Summary" text out as report.summary, then drop it
    summary = ""
    sm = re.match(
        r"^##\s+Summary\s*\n+(.+?)(?=\n##\s+|\Z)",
        main_body,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if sm:
        summary = sm.group(1).strip()
        main_body = main_body[sm.end():].lstrip()

    sections = [
        Section(
            heading="Reconstructed body",
            body=main_body.strip(),
            subtask_id="sec-all",
        )
    ] if main_body.strip() else []

    return ResearchReport(
        task_id=task_id,
        user_query=query,
        title=title,
        summary=summary,
        sections=sections,
        citations=citations,
    )


# ---------- one rerun ----------


@dataclass
class _Inputs:
    qid: str
    domain: str
    question: str
    md_path: Path


def _load_inputs(src_reports_dir: Path, bench_questions: list[BenchQuestion]) -> list[_Inputs]:
    """Pair each bench question with its r0 markdown report.

    Filenames are expected to follow the pattern
    ``<qid>-mimo-pipeline.md`` (the EvalRunner default).
    """
    by_id = {q.id: q for q in bench_questions}
    out: list[_Inputs] = []
    for path in sorted(src_reports_dir.glob("*.md")):
        m = re.match(r"^(rb-\d{3})-", path.name)
        if not m:
            continue
        qid = m.group(1)
        if qid not in by_id:
            continue
        q = by_id[qid]
        out.append(
            _Inputs(qid=qid, domain=q.domain, question=q.question, md_path=path)
        )
    return out


async def _process_one(
    inp: _Inputs,
    *,
    pool: MimoPool,
    judge: JudgeClient,
    embedder: Embedder,
    review_rounds: int,
    out_report_dir: Path,
    n_judge_samples: int,
    reviewer: str = "red",
) -> SampleResult:
    md = inp.md_path.read_text(encoding="utf-8")
    base_report = _parse_markdown_report(md, task_id=f"rerun-{inp.qid}", query=inp.question)

    sm = StateMachine()
    sm.fire("start")
    sm.fire("plan_skip_search")  # PLANNING -> WRITING (we already have a draft)

    ctx = AgentContext(pool=pool)
    t0 = time.monotonic()
    rb = await run_red_blue_loop(
        base_report,
        ctx,
        sm=sm,
        max_rounds=review_rounds,
        reviewer=reviewer,
        embedder=embedder,
    )
    elapsed = time.monotonic() - t0
    final_report = rb.final_report

    # Persist
    out_path = out_report_dir / f"{inp.qid}-mimo-pipeline-r{review_rounds}.md"
    out_path.write_text(final_report.to_markdown(), encoding="utf-8")
    final_md = final_report.to_markdown()

    # Find the reference facts and forbidden claims.
    bench = load_researchbench()
    q = next(x for x in bench.questions if x.id == inp.qid)
    rm = compute_rule_metrics(
        final_md,
        reference_facts=q.reference_facts,
        forbidden_claims=q.forbidden_claims,
        embedder=embedder,
    )

    judge_score: JudgeScore | None = None
    try:
        judge_score = await judge.score(
            question=inp.question, report=final_md, n_samples=n_judge_samples
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("judge failed for {}: {}", inp.qid, e)

    n_rounds = len(rb.rounds)
    n_attacks_total = sum(r.n_attacks for r in rb.rounds)
    n_patches_total = sum(r.n_patches_accepted for r in rb.rounds)
    logger.info(
        "{} rerun: {} rounds, {} attacks, {} patches accepted, stop={}",
        inp.qid,
        n_rounds,
        n_attacks_total,
        n_patches_total,
        rb.stop_reason,
    )

    return SampleResult(
        question_id=inp.qid,
        domain=inp.domain,
        question=inp.question,
        backend="mimo",
        mode="pipeline-rerun",
        report_path=str(out_path),
        rule_metrics=rm,
        judge_score=judge_score,
        elapsed_s=elapsed,
    )


def _summarize(samples: list[SampleResult]) -> EvalSummary:
    ok = [s for s in samples if s.error is None]
    f_accs = [s.rule_metrics.factual_accuracy for s in ok]
    hallus = [s.rule_metrics.hallucination_rate for s in ok]
    cites = [s.rule_metrics.citation_coverage for s in ok]
    j_overall = [s.judge_score.overall for s in ok if s.judge_score is not None]

    ci_facc = bootstrap_ci(f_accs)
    ci_hall = bootstrap_ci(hallus)
    ci_cite = bootstrap_ci(cites)

    judge_backends: dict[str, int] = {}
    n_self_bias_risk = 0
    by_domain: dict[str, list[SampleResult]] = {}
    for s in ok:
        if s.judge_score is not None:
            judge_backends[s.judge_score.backend] = (
                judge_backends.get(s.judge_score.backend, 0) + 1
            )
            if s.judge_score.self_bias_risk:
                n_self_bias_risk += 1
        by_domain.setdefault(s.domain, []).append(s)

    domain_means: dict[str, dict[str, float]] = {}
    for dom, group in by_domain.items():
        n = len(group)
        domain_means[dom] = {
            "n": n,
            "factual_accuracy": sum(g.rule_metrics.factual_accuracy for g in group) / n,
            "hallucination_rate": sum(g.rule_metrics.hallucination_rate for g in group) / n,
            "citation_coverage": sum(g.rule_metrics.citation_coverage for g in group) / n,
            "judge_overall": (
                sum(g.judge_score.overall for g in group if g.judge_score) /
                max(sum(1 for g in group if g.judge_score), 1)
            ) if any(g.judge_score for g in group) else float("nan"),
        }

    return EvalSummary(
        backend="mimo",
        mode="pipeline-rerun",
        n_samples=len(ok),
        n_failed=len(samples) - len(ok),
        metrics_mean={
            "factual_accuracy": ci_facc.point,
            "hallucination_rate": ci_hall.point,
            "citation_coverage": ci_cite.point,
        },
        metrics_ci95={
            "factual_accuracy": (ci_facc.low, ci_facc.high),
            "hallucination_rate": (ci_hall.low, ci_hall.high),
            "citation_coverage": (ci_cite.low, ci_cite.high),
        },
        judge_overall_mean=(sum(j_overall) / len(j_overall)) if j_overall else None,
        judge_overall_ci95=(
            (lambda c: (c.low, c.high))(bootstrap_ci(j_overall)) if j_overall else None
        ),
        by_domain=domain_means,
        judge_backends=judge_backends,
        n_self_bias_risk=n_self_bias_risk,
    )


# ---------- main ----------


async def main_async(args: argparse.Namespace) -> None:
    setup_logging()
    settings = get_settings()
    src_reports = Path(args.src)
    if not src_reports.exists():
        raise SystemExit(f"src not found: {src_reports}")

    bench = load_researchbench()
    inputs = _load_inputs(src_reports, bench.questions)
    if args.limit:
        inputs = inputs[: args.limit]
    logger.info("rerun inputs: {} reports under {}", len(inputs), src_reports)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.out) / f"{ts}-rerun-redblue-r{args.review}"
    out_root.mkdir(parents=True, exist_ok=True)
    report_dir = out_root / "reports"
    report_dir.mkdir(exist_ok=True)
    logger.info("rerun output: {}", out_root)

    embedder = Embedder()
    embedder.warmup()

    sem = asyncio.Semaphore(args.concurrency)
    samples: list[SampleResult] = []

    async with MimoPool(settings.mimo) as pool:
        judge = JudgeClient(
            settings.judge,
            fallback_pool=pool,
            fallback_model=settings.mimo.model,
        )
        try:
            async def one(inp: _Inputs) -> None:
                async with sem:
                    s = await _process_one(
                        inp,
                        pool=pool,
                        judge=judge,
                        embedder=embedder,
                        review_rounds=args.review,
                        out_report_dir=report_dir,
                        n_judge_samples=args.n_judge,
                        reviewer=args.reviewer,
                    )
                    samples.append(s)

            await asyncio.gather(*(one(i) for i in inputs))
        finally:
            await judge.aclose()

    summary = _summarize(samples)
    write_csv(samples, out_root / "per_question.csv")
    write_summary_md(summary, out_root / "summary.md")
    write_summary_json(summary, out_root / "summary.json")
    logger.info("done. summary -> {}", out_root / "summary.md")

    # If a baseline csv was provided, also write a comparison
    if args.compare_with:
        cmp_path = Path(args.compare_with)
        rep = compare(
            cmp_path,
            out_root / "per_question.csv",
            label_a="pipeline-r0 (orig)",
            label_b="pipeline-r2 (rerun-v2)",
        )
        write_comparison_md(rep, out_root / "compare-vs-r0.md")
        logger.info("comparison written: {}", out_root / "compare-vs-r0.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run only the Red-Blue review against saved pipeline-r0 reports."
    )
    parser.add_argument(
        "--src", required=True,
        help="Directory containing rb-NNN-mimo-pipeline.md files.",
    )
    parser.add_argument(
        "--out", default="experiments/results", help="Output root."
    )
    parser.add_argument(
        "--review", type=int, default=2, help="Red-Blue rounds (default: 2)"
    )
    parser.add_argument(
        "--reviewer", default="red", choices=["red", "multi"],
        help="Reviewer mode: 'red' single agent or 'multi' persona critics.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=3,
        help="Concurrent reruns (each uses 16 inner LLM concurrency).",
    )
    parser.add_argument(
        "--n-judge", type=int, default=2, help="Judge self-consistency samples"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap rerun count for a smoke test.",
    )
    parser.add_argument(
        "--compare-with", default=None,
        help="Path to a per_question.csv (e.g. pipeline-r0) for paired Cohen's d.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
