"""Rule-based metrics for research-report evaluation.

Three metrics, all computed from the report markdown plus the gold
``BenchQuestion``:

1. **factual_accuracy** — fraction of ``reference_facts`` that the report
   covers. A fact is "covered" if it has cosine similarity ≥
   ``hit_threshold`` to ANY sentence in the report (using bge-small-zh).

2. **hallucination_rate** — fraction of report sentences that
   semantically match any ``forbidden_claim`` (cosine ≥
   ``forbidden_threshold``). Lower is better.

3. **citation_coverage** — fraction of factual sentences (>= 12 chars,
   non-bullet, declarative) that contain at least one citation marker
   such as ``[1]`` or a markdown link. Higher is better.

The thresholds are intentionally conservative; tweaking them for the
ablation study is fine but should be documented in the experiment script.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from dr_agent.memory.embedder import Embedder

_CITATION_RE = re.compile(r"(\[\s*\d+\s*\])|(\[[^\]]+\]\([^)]+\))")
# Sentence splitter: same heuristic as compress.py
_SENT_RE = re.compile(r"(?<=[。！？!?；;])|\n+|(?<=[.!?])\s+")


@dataclass
class RuleMetrics:
    factual_accuracy: float
    hallucination_rate: float
    citation_coverage: float
    n_facts_total: int
    n_facts_hit: int
    n_sentences_total: int
    n_sentences_with_citation: int
    n_forbidden_total: int
    n_forbidden_hit: int


def _split_sentences(text: str, min_chars: int = 12) -> list[str]:
    parts = [p.strip() for p in _SENT_RE.split(text) if p and p.strip()]
    return [p for p in parts if len(p) >= min_chars]


def _strip_markdown(text: str) -> str:
    """Strip headings / fences / quote markers but keep prose intact."""
    cleaned = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("```", "---", ">", "|")):
            continue
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
        cleaned.append(stripped)
    return "\n".join(cleaned)


def compute_rule_metrics(
    report_md: str,
    *,
    reference_facts: list[str],
    forbidden_claims: list[str],
    embedder: Embedder,
    hit_threshold: float = 0.55,
    forbidden_threshold: float = 0.65,
) -> RuleMetrics:
    """Compute (factual_accuracy, hallucination_rate, citation_coverage)."""
    text = _strip_markdown(report_md)
    sentences = _split_sentences(text)

    # --- 1. factual_accuracy ---
    if reference_facts and sentences:
        sent_vecs = embedder.encode(sentences)
        fact_vecs = embedder.encode(reference_facts)
        # similarity matrix: facts (F, D) @ sents (D, N) -> (F, N)
        sim = fact_vecs @ sent_vecs.T
        max_per_fact = sim.max(axis=1)
        n_hit = int((max_per_fact >= hit_threshold).sum())
        factual_acc = n_hit / len(reference_facts)
    elif reference_facts:
        n_hit = 0
        factual_acc = 0.0
    else:
        n_hit = 0
        factual_acc = float("nan")

    # --- 2. hallucination_rate ---
    if forbidden_claims and sentences:
        sent_vecs2 = embedder.encode(sentences)
        forb_vecs = embedder.encode(forbidden_claims)
        sim2 = sent_vecs2 @ forb_vecs.T  # (N, F2)
        max_per_sent = sim2.max(axis=1) if sim2.size else np.zeros(len(sentences))
        n_forb = int((max_per_sent >= forbidden_threshold).sum())
        hallu = n_forb / max(len(sentences), 1)
    else:
        n_forb = 0
        hallu = 0.0

    # --- 3. citation_coverage ---
    n_with_cite = sum(1 for s in sentences if _CITATION_RE.search(s))
    cite_cov = n_with_cite / max(len(sentences), 1)

    return RuleMetrics(
        factual_accuracy=factual_acc,
        hallucination_rate=hallu,
        citation_coverage=cite_cov,
        n_facts_total=len(reference_facts),
        n_facts_hit=n_hit,
        n_sentences_total=len(sentences),
        n_sentences_with_citation=n_with_cite,
        n_forbidden_total=len(forbidden_claims),
        n_forbidden_hit=n_forb,
    )
