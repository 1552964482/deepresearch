"""Tests for the orchestrator state machine."""

from __future__ import annotations

import pytest

from dr_agent.orchestrator.state_machine import (
    InvalidTransition,
    StateMachine,
    TRANSITIONS,
)
from dr_agent.schemas.task import AgentState


def test_initial_state_is_idle() -> None:
    sm = StateMachine()
    assert sm.state is AgentState.IDLE


def test_happy_path_with_review() -> None:
    """Full pipeline including K=2 Red-Blue rounds should be expressible."""
    sm = StateMachine()
    edges = [
        "start",            # IDLE -> PLANNING
        "plan_ok",          # PLANNING -> SEARCHING
        "search_ok",        # SEARCHING -> READING
        "read_ok",          # READING -> COMPRESSING
        "compress_ok",      # COMPRESSING -> WRITING
        "draft_ok",         # WRITING -> RED_REVIEW
        "has_attacks",      # RED_REVIEW -> BLUE_REVISE
        "another_round",    # BLUE_REVISE -> RED_REVIEW
        "no_attacks",       # RED_REVIEW -> DONE
    ]
    for e in edges:
        sm.fire(e)
    assert sm.state is AgentState.DONE
    assert len(sm.history) == len(edges)


def test_phase1_minimal_path() -> None:
    """The Phase-1 hello-world path skips search and review."""
    sm = StateMachine()
    sm.fire("start")
    sm.fire("plan_skip_search")
    sm.fire("draft_skip_review")
    assert sm.state is AgentState.DONE


def test_invalid_transition_raises() -> None:
    sm = StateMachine()
    with pytest.raises(InvalidTransition):
        sm.fire("plan_ok")  # cannot fire plan_ok from IDLE


def test_can_check_does_not_change_state() -> None:
    sm = StateMachine()
    assert sm.can("start") is True
    assert sm.can("plan_ok") is False
    assert sm.state is AgentState.IDLE


def test_global_timeout_to_done() -> None:
    sm = StateMachine()
    sm.fire("start")
    sm.fire("global_timeout")
    sm.fire("force_converge")
    assert sm.state is AgentState.DONE


def test_history_records_each_transition() -> None:
    sm = StateMachine()
    sm.fire("start")
    sm.fire("plan_skip_search")
    sm.fire("draft_skip_review")
    assert sm.history[0].from_state is AgentState.IDLE
    assert sm.history[0].to_state is AgentState.PLANNING
    assert sm.history[-1].to_state is AgentState.DONE


def test_listeners_fire_on_transition() -> None:
    seen: list[str] = []
    sm = StateMachine(listeners=[lambda r: seen.append(r.edge)])
    sm.fire("start")
    sm.fire("plan_skip_search")
    assert seen == ["start", "plan_skip_search"]


def test_export_mermaid_contains_all_transitions() -> None:
    out = StateMachine.export_mermaid()
    assert "stateDiagram-v2" in out
    # Every legal transition should be present
    for (src, edge), dst in TRANSITIONS.items():
        assert f"{src.value} --> {dst.value} : {edge}" in out


def test_no_unreachable_target_states() -> None:
    """Every state should be reachable from at least one transition."""
    reached = {AgentState.IDLE}  # initial state
    for (_src, _edge), dst in TRANSITIONS.items():
        reached.add(dst)
    expected = set(AgentState)
    assert reached == expected
