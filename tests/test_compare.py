"""Tests for the cross-config comparison utility (eval/compare.py)."""

from __future__ import annotations

import csv
from pathlib import Path

from dr_agent.eval.compare import compare


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "domain", "factual_accuracy", "hallucination_rate",
              "citation_coverage", "judge_overall"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_compare_pairs_only_common_ids(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write(
        a,
        [
            {"id": "q1", "domain": "x", "factual_accuracy": "0.5",
             "hallucination_rate": "0.1", "citation_coverage": "0.0",
             "judge_overall": "3.5"},
            {"id": "q2", "domain": "x", "factual_accuracy": "0.6",
             "hallucination_rate": "0.2", "citation_coverage": "0.1",
             "judge_overall": "3.7"},
            {"id": "q3", "domain": "x", "factual_accuracy": "0.7",
             "hallucination_rate": "0.0", "citation_coverage": "0.2",
             "judge_overall": "4.0"},
        ],
    )
    _write(
        b,
        [
            {"id": "q1", "domain": "x", "factual_accuracy": "0.7",
             "hallucination_rate": "0.05", "citation_coverage": "0.3",
             "judge_overall": "4.0"},
            {"id": "q2", "domain": "x", "factual_accuracy": "0.8",
             "hallucination_rate": "0.10", "citation_coverage": "0.4",
             "judge_overall": "4.2"},
            {"id": "q4", "domain": "x", "factual_accuracy": "0.9",
             "hallucination_rate": "0.00", "citation_coverage": "0.5",
             "judge_overall": "4.5"},
        ],
    )
    rep = compare(a, b, label_a="A", label_b="B")
    facc = next(r for r in rep.metric_results if r.metric == "factual_accuracy")
    # Only q1 and q2 are common.
    assert facc.n_paired == 2
    assert facc.mean_a == 0.55
    assert facc.mean_b == 0.75
    # Cohen's d should be positive (B > A on mean)
    assert facc.cohens_d > 0


def test_compare_returns_zero_d_for_identical_rows(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    rows = [
        {"id": f"q{i}", "domain": "x", "factual_accuracy": "0.5",
         "hallucination_rate": "0.1", "citation_coverage": "0.2",
         "judge_overall": "4.0"}
        for i in range(5)
    ]
    _write(a, rows)
    _write(b, rows)
    rep = compare(a, b)
    for r in rep.metric_results:
        assert r.delta == 0.0
        # cohens_d is NaN when std is 0 — consistent with implementation.
        assert r.cohens_d != r.cohens_d or r.cohens_d == 0.0
