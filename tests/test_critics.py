"""Tests for the Multi-Critic consensus merge (offline, fake embedder)."""

from __future__ import annotations

import numpy as np

from dr_agent.agents.critics import MultiCritic
from dr_agent.schemas.attack import Attack, AttackType


class _FakeEmbedder:
    """Maps each unique span string to a fixed orthogonal-ish vector, except
    spans sharing a designated prefix get near-identical vectors so the
    consensus merge can cluster them."""

    dim = 8

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vecs = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            if t.startswith("DUP::"):
                v[0] = 1.0  # all duplicates collapse onto axis 0
            else:
                # deterministic off-axis direction per text
                idx = (abs(hash(t)) % 6) + 2
                v[idx] = 1.0
            vecs.append(v / (np.linalg.norm(v) or 1.0))
        return np.stack(vecs, axis=0)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def _atk(span: str, critic: str, sev: float = 0.5, t: str = "factual") -> Attack:
    a = Attack(id=f"a-{span[:4]}-{critic}", type=AttackType(t), span=span,
               evidence="e", severity=sev)
    a.__dict__["critic"] = critic
    a.__dict__["section_id"] = "sec-1"
    a.__dict__["suggested_action"] = None
    return a


def test_consensus_merge_collapses_duplicates() -> None:
    mc = MultiCritic(_FakeEmbedder(), merge_threshold=0.82)
    attacks = [
        _atk("DUP::same issue text", "fact_checker", 0.6),
        _atk("DUP::same issue text again", "citation_auditor", 0.5),
        _atk("a unique logic problem here", "logic_reviewer", 0.7),
    ]
    merged, n_consensus = mc._consensus_merge(attacks)
    # Two DUP:: spans collapse into one cluster; the unique one stays.
    assert len(merged) == 2
    # All three share section_id sec-1 with >=2 distinct critics -> the
    # section is a consensus hot-spot.
    assert n_consensus == 1


def test_consensus_boost_raises_severity() -> None:
    mc = MultiCritic(_FakeEmbedder(), merge_threshold=0.82, consensus_boost=0.15)
    attacks = [
        _atk("DUP::x", "fact_checker", 0.6),
        _atk("DUP::y", "logic_reviewer", 0.6),
    ]
    merged, n_consensus = mc._consensus_merge(attacks)
    assert n_consensus == 1
    # span-level 2 critics -> boost: 0.6 + 0.15 = 0.75
    assert merged[0].severity == 0.75
    assert merged[0].__dict__["n_critics"] == 2


def test_lone_critic_in_lonely_section_penalized() -> None:
    mc = MultiCritic(_FakeEmbedder(), lone_penalty=0.10)
    # Single critic, single section -> not a hot-spot -> penalized.
    a = _atk("only one critic flagged this", "fact_checker", 0.5)
    merged, n_consensus = mc._consensus_merge([a])
    assert n_consensus == 0
    assert merged[0].severity == 0.40  # 0.5 - 0.10


def test_section_hotspot_boosts_even_distinct_spans() -> None:
    """Two critics attack DIFFERENT spans in the SAME section -> the section
    is a hot-spot, so both attacks get boosted (not penalized)."""
    mc = MultiCritic(_FakeEmbedder(), consensus_boost=0.15, lone_penalty=0.10)
    attacks = [
        _atk("distinct span alpha here", "fact_checker", 0.5),
        _atk("distinct span beta there", "citation_auditor", 0.5),
    ]
    merged, n_consensus = mc._consensus_merge(attacks)
    assert n_consensus == 1
    # Both boosted to 0.65, none penalized.
    assert all(abs(m.severity - 0.65) < 1e-9 for m in merged)


def test_merge_sorts_by_severity_desc() -> None:
    mc = MultiCritic(_FakeEmbedder())
    attacks = [
        _atk("low severity unique span", "fact_checker", 0.3),
        _atk("DUP::high", "logic_reviewer", 0.7),
        _atk("DUP::high2", "citation_auditor", 0.7),
    ]
    merged, _ = mc._consensus_merge(attacks)
    sev = [m.severity for m in merged]
    assert sev == sorted(sev, reverse=True)


def test_empty_returns_empty() -> None:
    mc = MultiCritic(_FakeEmbedder())
    merged, n = mc._consensus_merge([])
    assert merged == [] and n == 0
