"""Tests for ResearchBench loader."""

from __future__ import annotations

from dr_agent.eval.bench import load_researchbench


def test_researchbench_has_35_questions() -> None:
    bench = load_researchbench()
    assert len(bench.questions) == 35


def test_researchbench_has_11_domains() -> None:
    bench = load_researchbench()
    domains = {q.domain for q in bench.questions}
    assert len(domains) == 11


def test_each_question_has_reference_facts() -> None:
    bench = load_researchbench()
    for q in bench.questions:
        assert q.id.startswith("rb-")
        assert q.question
        assert q.reference_facts, f"{q.id} missing reference_facts"
        assert q.scoring_rubric, f"{q.id} missing scoring_rubric"


def test_filter_by_domain() -> None:
    bench = load_researchbench()
    sub = bench.filter(domain="ai-ml-fundamentals")
    assert len(sub.questions) == 3
    assert all(q.domain == "ai-ml-fundamentals" for q in sub.questions)


def test_filter_by_ids() -> None:
    bench = load_researchbench()
    sub = bench.filter(ids=["rb-001", "rb-014"])
    assert {q.id for q in sub.questions} == {"rb-001", "rb-014"}


def test_filter_by_limit() -> None:
    bench = load_researchbench()
    sub = bench.filter(limit=5)
    assert len(sub.questions) == 5
