"""Tests for Red parsing fallback and Blue patch application."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from dr_agent.agents._attack_parse import coerce_attacks as _coerce_attacks
from dr_agent.agents.blue import Blue, BlueInput, _apply_patch
from dr_agent.schemas.attack import Attack, AttackType
from dr_agent.schemas.patch import Patch, PatchAction
from dr_agent.schemas.report import ResearchReport, Section


# ---------- _coerce_attacks ----------


def test_coerce_attacks_accepts_well_formed_json() -> None:
    parsed = {
        "attacks": [
            {
                "section_id": "sec-1",
                "type": "factual",
                "span": "GRPO 由 OpenAI 提出",
                "evidence": "实际由 DeepSeek 提出",
                "severity": 0.9,
                "suggested_action": "MODIFY",
            }
        ]
    }
    out = _coerce_attacks(parsed)
    assert len(out) == 1
    a = out[0]
    assert a.type is AttackType.FACTUAL
    assert a.severity == 0.9
    assert a.__dict__["section_id"] == "sec-1"
    assert a.__dict__["suggested_action"] is PatchAction.MODIFY


def test_coerce_attacks_drops_invalid_entries() -> None:
    parsed = {
        "attacks": [
            {"type": "factual", "span": "good", "evidence": "ok", "severity": 0.5},
            {"type": "unknown_type", "span": "x", "evidence": "y", "severity": 0.5},
            {"type": "logic", "span": "", "evidence": "empty span", "severity": 0.5},
            {"type": "logic"},  # missing fields
        ]
    }
    out = _coerce_attacks(parsed)
    assert len(out) == 1
    assert out[0].span == "good"


def test_coerce_attacks_clamps_severity() -> None:
    parsed = {
        "attacks": [
            {"type": "factual", "span": "abcdefgh", "evidence": "e", "severity": 5.0},
            {"type": "logic", "span": "abcdefgh", "evidence": "e", "severity": -3},
        ]
    }
    out = _coerce_attacks(parsed)
    assert out[0].severity == 1.0
    assert out[1].severity == 0.0


def test_coerce_attacks_handles_non_dict() -> None:
    assert _coerce_attacks(None) == []
    assert _coerce_attacks([1, 2, 3]) == []
    assert _coerce_attacks({"attacks": "not a list"}) == []


# ---------- _apply_patch ----------


def _section(body: str) -> Section:
    return Section(heading="t", body=body, subtask_id="sec-1")


def test_apply_modify_replaces_span() -> None:
    sec = _section("GRPO 由 OpenAI 提出。其他内容。")
    p = Patch(
        attack_id="a",
        action=PatchAction.MODIFY,
        target_span="GRPO 由 OpenAI 提出",
        new_text="GRPO 由 DeepSeek 提出",
    )
    new_sec, applied = _apply_patch(sec, p)
    assert applied is True
    assert "DeepSeek" in new_sec.body
    assert "OpenAI" not in new_sec.body


def test_apply_delete_removes_span() -> None:
    sec = _section("有用句子。无用句子。")
    p = Patch(
        attack_id="a",
        action=PatchAction.DELETE,
        target_span="无用句子。",
    )
    new_sec, applied = _apply_patch(sec, p)
    assert applied is True
    assert "无用" not in new_sec.body
    assert "有用" in new_sec.body


def test_apply_add_appends_after_span() -> None:
    sec = _section("锚点句子。后续。")
    p = Patch(
        attack_id="a",
        action=PatchAction.ADD,
        target_span="锚点句子。",
        new_text="补充说明。",
    )
    new_sec, applied = _apply_patch(sec, p)
    assert applied is True
    assert "锚点句子。补充说明。" in new_sec.body


def test_apply_verify_does_not_change_body() -> None:
    sec = _section("一个事实。")
    p = Patch(
        attack_id="a",
        action=PatchAction.VERIFY,
        target_span="一个事实。",
        verification_evidence="https://source.example/page",
    )
    new_sec, applied = _apply_patch(sec, p)
    assert applied is True
    assert new_sec.body == sec.body


def test_apply_skips_when_span_missing() -> None:
    sec = _section("正文不包含目标。")
    p = Patch(
        attack_id="a",
        action=PatchAction.MODIFY,
        target_span="不存在的片段",
        new_text="任何文本",
    )
    _new_sec, applied = _apply_patch(sec, p)
    assert applied is False


def test_apply_modify_rejects_when_citation_lost() -> None:
    """If the original span contains [3] but the new_text drops it,
    the patch must be rejected to preserve citation density."""
    sec = _section("Transformer 由 Vaswani 等于 2017 年提出 [3]，并改变 NLP。")
    p = Patch(
        attack_id="a",
        action=PatchAction.MODIFY,
        target_span="Transformer 由 Vaswani 等于 2017 年提出 [3]",
        new_text="Transformer 由 Google 提出",  # citation [3] dropped
    )
    new_sec, applied = _apply_patch(sec, p)
    assert applied is False
    assert "[3]" in new_sec.body  # original preserved


def test_apply_modify_accepts_when_citation_preserved() -> None:
    sec = _section("X 是一种算法 [2]。")
    p = Patch(
        attack_id="a",
        action=PatchAction.MODIFY,
        target_span="X 是一种算法 [2]",
        new_text="X 是一种由 DeepSeek 提出的算法 [2]",
    )
    new_sec, applied = _apply_patch(sec, p)
    assert applied is True
    assert "DeepSeek" in new_sec.body
    assert "[2]" in new_sec.body


def test_apply_delete_rejects_when_orphans_citation() -> None:
    """DELETE that wipes the only mention of [4] must be rejected."""
    sec = _section("某事实 [4]。其他无关内容。")
    p = Patch(
        attack_id="a",
        action=PatchAction.DELETE,
        target_span="某事实 [4]。",
    )
    _new_sec, applied = _apply_patch(sec, p)
    assert applied is False


def test_apply_delete_allowed_when_citation_appears_elsewhere() -> None:
    sec = _section("断言 [4]。同源的另一断言 [4]。")
    p = Patch(
        attack_id="a",
        action=PatchAction.DELETE,
        target_span="断言 [4]。",
    )
    new_sec, applied = _apply_patch(sec, p)
    assert applied is True
    assert "[4]" in new_sec.body  # still there from the second mention


# ---------- Patch validator ----------


def test_modify_without_new_text_fails_validation() -> None:
    with pytest.raises(Exception):
        Patch(
            attack_id="a",
            action=PatchAction.MODIFY,
            target_span="x",
        )


def test_verify_without_evidence_fails_validation() -> None:
    with pytest.raises(Exception):
        Patch(
            attack_id="a",
            action=PatchAction.VERIFY,
            target_span="x",
        )
