"""Table-driven state machine for the 9 main states + 2 error states.

Legal transitions are declared in :data:`TRANSITIONS`. Anything not
listed raises :class:`InvalidTransition` so misuse fails loudly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from dr_agent.schemas.task import AgentState

S = AgentState


# Edge label -> (from_state, to_state)
TRANSITIONS: dict[tuple[AgentState, str], AgentState] = {
    (S.IDLE, "start"): S.PLANNING,
    (S.PLANNING, "plan_ok"): S.SEARCHING,
    (S.PLANNING, "plan_skip_search"): S.WRITING,  # for hello-world / no-search mode
    (S.PLANNING, "plan_error"): S.FAILED,
    (S.PLANNING, "global_timeout"): S.TIMEOUT,
    (S.SEARCHING, "search_ok"): S.READING,
    (S.SEARCHING, "search_all_fail"): S.FAILED,
    (S.SEARCHING, "global_timeout"): S.TIMEOUT,
    (S.READING, "read_ok"): S.COMPRESSING,
    (S.READING, "global_timeout"): S.TIMEOUT,
    (S.COMPRESSING, "compress_ok"): S.WRITING,
    (S.COMPRESSING, "global_timeout"): S.TIMEOUT,
    (S.WRITING, "draft_ok"): S.RED_REVIEW,
    (S.WRITING, "draft_skip_review"): S.DONE,  # hello-world path
    (S.WRITING, "global_timeout"): S.TIMEOUT,
    (S.RED_REVIEW, "has_attacks"): S.BLUE_REVISE,
    (S.RED_REVIEW, "no_attacks"): S.DONE,
    (S.BLUE_REVISE, "another_round"): S.RED_REVIEW,
    (S.BLUE_REVISE, "stop"): S.DONE,
    (S.TIMEOUT, "force_converge"): S.DONE,
}


class InvalidTransition(RuntimeError):
    pass


@dataclass
class TransitionRecord:
    at: datetime
    from_state: AgentState
    to_state: AgentState
    edge: str


@dataclass
class StateMachine:
    state: AgentState = S.IDLE
    history: list[TransitionRecord] = field(default_factory=list)
    listeners: list[Callable[[TransitionRecord], None]] = field(default_factory=list)

    def fire(self, edge: str) -> AgentState:
        target = TRANSITIONS.get((self.state, edge))
        if target is None:
            raise InvalidTransition(f"no transition for ({self.state.value}, {edge!r})")
        rec = TransitionRecord(
            at=datetime.now(timezone.utc),
            from_state=self.state,
            to_state=target,
            edge=edge,
        )
        self.state = target
        self.history.append(rec)
        for cb in self.listeners:
            try:
                cb(rec)
            except Exception:  # noqa: BLE001
                pass
        return target

    def can(self, edge: str) -> bool:
        return (self.state, edge) in TRANSITIONS

    @staticmethod
    def export_mermaid() -> str:
        lines = ["stateDiagram-v2", "    [*] --> IDLE"]
        for (src, edge), dst in TRANSITIONS.items():
            lines.append(f"    {src.value} --> {dst.value} : {edge}")
        lines.append("    DONE --> [*]")
        lines.append("    FAILED --> [*]")
        return "\n".join(lines)
