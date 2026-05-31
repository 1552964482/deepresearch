"""Evaluation runner.

For each ``BenchQuestion`` the runner:
  1. Generates a report by either:
     * running the full DeepResearch pipeline on a configured backend
       (``mode='pipeline'``), or
     * single-prompt baseline (``mode='baseline'``) — useful for
       "dumb but fast" comparison points.
  2. Computes rule metrics (factual accuracy / hallucination /
     citation coverage).
  3. Calls the independent JudgeClient for a 5-dimension score.
  4. Persists per-question CSV + a Markdown summary including
     bootstrap 95% CI.

Two reports compared with the same metrics yield Cohen's d for the
ablation table.
"""

from __future__ import annotations

import asyncio
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from loguru import logger

from dr_agent.eval.bench import Bench, BenchQuestion
from dr_agent.eval.rule_metrics import RuleMetrics, compute_rule_metrics
from dr_agent.eval.stats import bootstrap_ci
from dr_agent.llm.judge import JudgeClient, JudgeScore
from dr_agent.llm.pool import MimoPool
from dr_agent.memory.compress import Compressor
from dr_agent.memory.embedder import Embedder
from dr_agent.memory.store import MemoryStore
from dr_agent.orchestrator.runner import run_grounded
from dr_agent.schemas.report import ResearchReport
from dr_agent.tools.fetcher import Fetcher
from dr_agent.tools.search import WebSearcher


@dataclass
class SampleResult:
    question_id: str
    domain: str
    question: str
    backend: str
    mode: str
    report_path: str
    rule_metrics: RuleMetrics
    judge_score: JudgeScore | None
    elapsed_s: float
    error: str | None = None


@dataclass
class EvalSummary:
    backend: str
    mode: str
    n_samples: int
    n_failed: int
    metrics_mean: dict[str, float]
    metrics_ci95: dict[str, tuple[float, float]]
    judge_overall_mean: float | None
    judge_overall_ci95: tuple[float, float] | None
    by_domain: dict[str, dict[str, float]] = field(default_factory=dict)
    judge_backends: dict[str, int] = field(default_factory=dict)
    n_self_bias_risk: int = 0


# ---------- baseline backend ----------


_BASELINE_SYS = """你是一名研究报告作者。请基于通用知识，对用户给出的研究问题撰写一份完整但精炼的研究报告。

要求：
- 中文输出（除非问题本身是英文）
- 用 markdown 标题组织（## 一级，### 二级）
- 给出 2-4 个核心要点 / 章节
- 标注关键事实的年份或来源（如「据 2023 年 XX 论文」）
- 不要捏造引用编号 [1][2]
- 长度严格控制在 600-900 字"""


async def _baseline_report(
    pool: MimoPool, question: BenchQuestion
) -> ResearchReport:
    messages = [
        {"role": "system", "content": _BASELINE_SYS},
        {"role": "user", "content": question.question},
    ]
    res = await pool.chat(messages, temperature=0.4, max_tokens=1200)
    return ResearchReport(
        task_id=f"baseline-{question.id}",
        user_query=question.question,
        title=f"Baseline: {question.question[:40]}",
        summary="",
        sections=[],
        citations=[],
    ).model_copy(update={"summary": res.content})


# ---------- pipeline backend ----------


async def _pipeline_report(
    pool: MimoPool,
    question: BenchQuestion,
    *,
    embedder: Embedder,
    memory: MemoryStore,
    web_searcher: WebSearcher,
    fetcher: Fetcher,
    review_rounds: int,
) -> ResearchReport:
    report, _sm, _rb = await run_grounded(
        question.question,
        pool,
        embedder=embedder,
        memory=memory,
        web_searcher=web_searcher,
        fetcher=fetcher,
        review_rounds=review_rounds,
    )
    return report


# ---------- main runner ----------


def _domain_means(samples: list[SampleResult]) -> dict[str, dict[str, float]]:
    by_domain: dict[str, list[SampleResult]] = {}
    for s in samples:
        by_domain.setdefault(s.domain, []).append(s)
    out: dict[str, dict[str, float]] = {}
    for dom, group in by_domain.items():
        n = len(group)
        if n == 0:
            continue
        f_acc = sum(g.rule_metrics.factual_accuracy for g in group) / n
        hallu = sum(g.rule_metrics.hallucination_rate for g in group) / n
        c_cov = sum(g.rule_metrics.citation_coverage for g in group) / n
        j_overall = [g.judge_score.overall for g in group if g.judge_score]
        out[dom] = {
            "n": n,
            "factual_accuracy": f_acc,
            "hallucination_rate": hallu,
            "citation_coverage": c_cov,
            "judge_overall": sum(j_overall) / len(j_overall) if j_overall else float("nan"),
        }
    return out


class EvalRunner:
    """Top-level orchestrator for benchmark evaluation."""

    def __init__(
        self,
        *,
        pool: MimoPool,
        judge: JudgeClient,
        embedder: Embedder,
        web_searcher: WebSearcher,
        fetcher: Fetcher,
        memory_db: Path,
        report_dir: Path,
        backend_name: str,
        mode: Literal["pipeline", "baseline"],
        review_rounds: int = 0,
        n_judge_samples: int = 3,
    ) -> None:
        self.pool = pool
        self.judge = judge
        self.embedder = embedder
        self.web_searcher = web_searcher
        self.fetcher = fetcher
        self.memory_db = memory_db
        self.report_dir = report_dir
        self.backend_name = backend_name
        self.mode = mode
        self.review_rounds = review_rounds
        self.n_judge_samples = n_judge_samples
        self._memory: MemoryStore | None = None

    def _ensure_memory(self) -> MemoryStore:
        if self._memory is None:
            self._memory = MemoryStore(self.memory_db, self.embedder)
        return self._memory

    async def run_one(self, q: BenchQuestion) -> SampleResult:
        t0 = time.monotonic()
        try:
            if self.mode == "pipeline":
                report = await _pipeline_report(
                    self.pool,
                    q,
                    embedder=self.embedder,
                    memory=self._ensure_memory(),
                    web_searcher=self.web_searcher,
                    fetcher=self.fetcher,
                    review_rounds=self.review_rounds,
                )
            else:
                report = await _baseline_report(self.pool, q)
            elapsed = time.monotonic() - t0
        except Exception as e:  # noqa: BLE001
            logger.exception("question {} failed", q.id)
            return SampleResult(
                question_id=q.id,
                domain=q.domain,
                question=q.question,
                backend=self.backend_name,
                mode=self.mode,
                report_path="",
                rule_metrics=RuleMetrics(
                    factual_accuracy=0.0,
                    hallucination_rate=0.0,
                    citation_coverage=0.0,
                    n_facts_total=len(q.reference_facts),
                    n_facts_hit=0,
                    n_sentences_total=0,
                    n_sentences_with_citation=0,
                    n_forbidden_total=len(q.forbidden_claims),
                    n_forbidden_hit=0,
                ),
                judge_score=None,
                elapsed_s=time.monotonic() - t0,
                error=repr(e)[:300],
            )

        # Persist report
        rpath = self.report_dir / f"{q.id}-{self.backend_name}-{self.mode}.md"
        rpath.write_text(report.to_markdown(), encoding="utf-8")

        report_md = report.to_markdown()

        # Rule metrics (sync, fast)
        rm = compute_rule_metrics(
            report_md,
            reference_facts=q.reference_facts,
            forbidden_claims=q.forbidden_claims,
            embedder=self.embedder,
        )

        # Judge (independent backend, n_samples self-consistency)
        judge_score: JudgeScore | None = None
        try:
            judge_score = await self.judge.score(
                question=q.question, report=report_md, n_samples=self.n_judge_samples
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("judge failed for {}: {}", q.id, e)

        return SampleResult(
            question_id=q.id,
            domain=q.domain,
            question=q.question,
            backend=self.backend_name,
            mode=self.mode,
            report_path=str(rpath),
            rule_metrics=rm,
            judge_score=judge_score,
            elapsed_s=elapsed,
        )

    async def run_all(
        self, bench: Bench, *, concurrency: int = 2
    ) -> tuple[list[SampleResult], EvalSummary]:
        sem = asyncio.Semaphore(concurrency)

        async def _wrapped(q: BenchQuestion) -> SampleResult:
            async with sem:
                logger.info("[{}/{}] {} domain={}", q.id, self.mode, q.id, q.domain)
                return await self.run_one(q)

        results = await asyncio.gather(*(_wrapped(q) for q in bench.questions))
        return results, self._summarize(results)

    def _summarize(self, samples: list[SampleResult]) -> EvalSummary:
        ok = [s for s in samples if s.error is None]
        n_failed = len(samples) - len(ok)
        f_accs = [s.rule_metrics.factual_accuracy for s in ok]
        hallus = [s.rule_metrics.hallucination_rate for s in ok]
        cites = [s.rule_metrics.citation_coverage for s in ok]
        j_overall = [s.judge_score.overall for s in ok if s.judge_score is not None]

        ci_facc = bootstrap_ci(f_accs)
        ci_hall = bootstrap_ci(hallus)
        ci_cite = bootstrap_ci(cites)

        judge_backends: dict[str, int] = {}
        n_self_bias_risk = 0
        for s in ok:
            if s.judge_score is None:
                continue
            judge_backends[s.judge_score.backend] = (
                judge_backends.get(s.judge_score.backend, 0) + 1
            )
            if s.judge_score.self_bias_risk:
                n_self_bias_risk += 1

        return EvalSummary(
            backend=self.backend_name,
            mode=self.mode,
            n_samples=len(ok),
            n_failed=n_failed,
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
                (lambda c: (c.low, c.high))(bootstrap_ci(j_overall))
                if j_overall
                else None
            ),
            by_domain=_domain_means(ok),
            judge_backends=judge_backends,
            n_self_bias_risk=n_self_bias_risk,
        )

    def close(self) -> None:
        if self._memory is not None:
            self._memory.close()


# ---------- output helpers ----------


def write_csv(samples: list[SampleResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id", "domain", "backend", "mode",
                "factual_accuracy", "hallucination_rate", "citation_coverage",
                "n_facts_total", "n_facts_hit", "n_sentences_total",
                "judge_backend", "judge_self_bias_risk",
                "judge_overall", "judge_accuracy", "judge_completeness",
                "judge_logic", "judge_citation", "judge_readability",
                "elapsed_s", "error",
            ]
        )
        for s in samples:
            js = s.judge_score
            w.writerow(
                [
                    s.question_id, s.domain, s.backend, s.mode,
                    f"{s.rule_metrics.factual_accuracy:.4f}",
                    f"{s.rule_metrics.hallucination_rate:.4f}",
                    f"{s.rule_metrics.citation_coverage:.4f}",
                    s.rule_metrics.n_facts_total,
                    s.rule_metrics.n_facts_hit,
                    s.rule_metrics.n_sentences_total,
                    js.backend if js else "",
                    int(js.self_bias_risk) if js else "",
                    f"{js.overall:.3f}" if js else "",
                    f"{js.accuracy:.3f}" if js else "",
                    f"{js.completeness:.3f}" if js else "",
                    f"{js.logic:.3f}" if js else "",
                    f"{js.citation_quality:.3f}" if js else "",
                    f"{js.readability:.3f}" if js else "",
                    f"{s.elapsed_s:.2f}",
                    s.error or "",
                ]
            )


def write_summary_md(summary: EvalSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# Eval Summary — backend={summary.backend} mode={summary.mode}\n")
    lines.append(f"- samples evaluated: {summary.n_samples} ({summary.n_failed} failed)")
    if summary.judge_backends:
        lines.append(
            "- judge backends seen: "
            + ", ".join(f"{k}×{v}" for k, v in sorted(summary.judge_backends.items()))
        )
        if summary.n_self_bias_risk:
            lines.append(
                f"- ⚠️  self-bias risk: {summary.n_self_bias_risk} samples scored via mimo fallback"
            )
    lines.append("")
    lines.append("## Aggregate metrics (95% CI via Bootstrap, BCa)\n")
    lines.append("| metric | mean | 95% CI |")
    lines.append("|---|---|---|")
    for k, mean in summary.metrics_mean.items():
        lo, hi = summary.metrics_ci95[k]
        lines.append(f"| {k} | {mean:.3f} | [{lo:.3f}, {hi:.3f}] |")
    if summary.judge_overall_mean is not None:
        jlo, jhi = summary.judge_overall_ci95 or (float("nan"), float("nan"))
        lines.append(
            f"| judge_overall (1-5) | {summary.judge_overall_mean:.3f} | [{jlo:.3f}, {jhi:.3f}] |"
        )

    lines.append("")
    lines.append("## By-domain breakdown\n")
    lines.append("| domain | n | factual_acc | hallu | cite_cov | judge_overall |")
    lines.append("|---|---|---|---|---|---|")
    for dom, m in sorted(summary.by_domain.items()):
        lines.append(
            f"| {dom} | {int(m['n'])} | {m['factual_accuracy']:.3f} | "
            f"{m['hallucination_rate']:.3f} | {m['citation_coverage']:.3f} | "
            + (
                f"{m['judge_overall']:.3f}"
                if not (m['judge_overall'] != m['judge_overall'])  # not NaN
                else "n/a"
            )
            + " |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_json(summary: EvalSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "backend": summary.backend,
        "mode": summary.mode,
        "n_samples": summary.n_samples,
        "n_failed": summary.n_failed,
        "metrics_mean": summary.metrics_mean,
        "metrics_ci95": {k: list(v) for k, v in summary.metrics_ci95.items()},
        "judge_overall_mean": summary.judge_overall_mean,
        "judge_overall_ci95": (
            list(summary.judge_overall_ci95) if summary.judge_overall_ci95 else None
        ),
        "by_domain": summary.by_domain,
        "judge_backends": summary.judge_backends,
        "n_self_bias_risk": summary.n_self_bias_risk,
    }
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
