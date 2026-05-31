"""Cross-config comparison: Cohen's d effect sizes between two eval runs.

Two runs are compared sample-by-sample (joined on ``question_id``) so the
effect size is computed on **paired** observations of the same questions.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from dr_agent.eval.stats import bootstrap_ci, cohens_d


@dataclass
class PairedRow:
    question_id: str
    domain: str
    metric_a: float
    metric_b: float


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _maybe_float(s: str) -> float:
    try:
        return float(s) if s.strip() else float("nan")
    except ValueError:
        return float("nan")


@dataclass
class MetricComparison:
    metric: str
    n_paired: int
    mean_a: float
    mean_b: float
    delta: float
    cohens_d: float
    ci_a: tuple[float, float]
    ci_b: tuple[float, float]


@dataclass
class ComparisonReport:
    label_a: str
    label_b: str
    metric_results: list[MetricComparison]


METRICS = (
    "factual_accuracy",
    "hallucination_rate",
    "citation_coverage",
    "judge_overall",
)


def compare(
    csv_a: Path,
    csv_b: Path,
    *,
    label_a: str | None = None,
    label_b: str | None = None,
) -> ComparisonReport:
    rows_a = _load_csv(csv_a)
    rows_b = _load_csv(csv_b)
    by_id_a = {r["id"]: r for r in rows_a}
    by_id_b = {r["id"]: r for r in rows_b}
    common_ids = sorted(set(by_id_a) & set(by_id_b))

    results: list[MetricComparison] = []
    for metric in METRICS:
        paired: list[PairedRow] = []
        for qid in common_ids:
            a = _maybe_float(by_id_a[qid].get(metric, ""))
            b = _maybe_float(by_id_b[qid].get(metric, ""))
            if a != a or b != b:  # any NaN -> skip
                continue
            paired.append(
                PairedRow(
                    question_id=qid,
                    domain=by_id_a[qid].get("domain", ""),
                    metric_a=a,
                    metric_b=b,
                )
            )
        if not paired:
            results.append(
                MetricComparison(
                    metric=metric,
                    n_paired=0,
                    mean_a=float("nan"),
                    mean_b=float("nan"),
                    delta=float("nan"),
                    cohens_d=float("nan"),
                    ci_a=(float("nan"), float("nan")),
                    ci_b=(float("nan"), float("nan")),
                )
            )
            continue

        a_vals = [p.metric_a for p in paired]
        b_vals = [p.metric_b for p in paired]
        mean_a = sum(a_vals) / len(a_vals)
        mean_b = sum(b_vals) / len(b_vals)
        ci_a = bootstrap_ci(a_vals)
        ci_b = bootstrap_ci(b_vals)
        # Effect size of B vs A: positive d means b > a on average.
        d = cohens_d(b_vals, a_vals)
        results.append(
            MetricComparison(
                metric=metric,
                n_paired=len(paired),
                mean_a=mean_a,
                mean_b=mean_b,
                delta=mean_b - mean_a,
                cohens_d=d,
                ci_a=(ci_a.low, ci_a.high),
                ci_b=(ci_b.low, ci_b.high),
            )
        )

    return ComparisonReport(
        label_a=label_a or csv_a.stem,
        label_b=label_b or csv_b.stem,
        metric_results=results,
    )


def write_comparison_md(report: ComparisonReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# Comparison: **{report.label_b}** vs **{report.label_a}**\n")
    lines.append(
        "Cohen's d is computed on paired per-question observations. "
        "Positive d means **B > A** on the metric. Conventional thresholds: "
        "|d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large.\n"
    )
    lines.append("| metric | n | mean(A) | mean(B) | Δ | Cohen's d |")
    lines.append("|---|---|---|---|---|---|")
    for r in report.metric_results:
        if r.n_paired == 0:
            lines.append(f"| {r.metric} | 0 | n/a | n/a | n/a | n/a |")
            continue
        # Hallucination is "lower is better"; flip sign for readability comment.
        improvement_marker = ""
        if r.metric == "hallucination_rate":
            improvement_marker = " ⬇ better" if r.delta < 0 else " ⬆ worse"
        else:
            improvement_marker = " ⬆ better" if r.delta > 0 else " ⬇ worse"
        lines.append(
            f"| {r.metric} | {r.n_paired} | {r.mean_a:.3f} | {r.mean_b:.3f} | "
            f"{r.delta:+.3f}{improvement_marker} | {r.cohens_d:+.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
