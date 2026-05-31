"""Tests for ResultEnvelope (orchestrator/envelope.py)."""

from __future__ import annotations

import pytest

from dr_agent.orchestrator.envelope import ResultEnvelope


def test_success_envelope_unwraps() -> None:
    env = ResultEnvelope.success(42, elapsed_s=1.5, trace_id="tr-1")
    assert env.ok
    assert env.value == 42
    assert env.unwrap() == 42
    assert env.error is None


def test_failure_envelope_carries_exception() -> None:
    err = ValueError("boom")
    env = ResultEnvelope.failure(err, elapsed_s=0.1)
    assert not env.ok
    assert env.error is err
    with pytest.raises(ValueError, match="boom"):
        env.unwrap()


def test_envelope_meta_default_is_empty_dict() -> None:
    a = ResultEnvelope.success(1)
    b = ResultEnvelope.success(2)
    a.meta["k"] = "v"
    # Independent dicts (default_factory)
    assert b.meta == {}
