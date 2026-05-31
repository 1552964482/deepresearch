"""ResearchBench loader."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchQuestion:
    id: str
    domain: str
    language: str
    question: str
    reference_facts: list[str] = field(default_factory=list)
    forbidden_claims: list[str] = field(default_factory=list)
    scoring_rubric: list[str] = field(default_factory=list)


@dataclass
class Bench:
    name: str
    path: Path
    questions: list[BenchQuestion]

    def by_domain(self) -> dict[str, list[BenchQuestion]]:
        out: dict[str, list[BenchQuestion]] = {}
        for q in self.questions:
            out.setdefault(q.domain, []).append(q)
        return out

    def filter(
        self,
        *,
        domain: str | None = None,
        ids: list[str] | None = None,
        limit: int | None = None,
    ) -> "Bench":
        qs = self.questions
        if domain:
            qs = [q for q in qs if q.domain == domain]
        if ids:
            keep = set(ids)
            qs = [q for q in qs if q.id in keep]
        if limit is not None:
            qs = qs[:limit]
        return Bench(name=self.name, path=self.path, questions=qs)


def _load_jsonl(path: Path) -> list[BenchQuestion]:
    out: list[BenchQuestion] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{path}:{i} malformed JSON: {e}") from e
            out.append(
                BenchQuestion(
                    id=obj["id"],
                    domain=obj["domain"],
                    language=obj.get("language", "zh"),
                    question=obj["question"],
                    reference_facts=list(obj.get("reference_facts", [])),
                    forbidden_claims=list(obj.get("forbidden_claims", [])),
                    scoring_rubric=list(obj.get("scoring_rubric", [])),
                )
            )
    return out


def load_researchbench(root: Path | None = None) -> Bench:
    """Load the bundled ResearchBench from ``benchmarks/researchbench/``."""
    if root is None:
        # default: resolve relative to repo root via this file's location
        # src/dr_agent/eval/bench.py -> repo root is parents[3]
        root = Path(__file__).resolve().parents[3] / "benchmarks" / "researchbench"
    path = Path(root) / "questions.jsonl"
    questions = _load_jsonl(path)
    return Bench(name="researchbench", path=path, questions=questions)
