"""Core task / state-machine data models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class AgentState(str, Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    SEARCHING = "SEARCHING"
    READING = "READING"
    COMPRESSING = "COMPRESSING"
    WRITING = "WRITING"
    RED_REVIEW = "RED_REVIEW"
    BLUE_REVISE = "BLUE_REVISE"
    DONE = "DONE"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


class SubTask(BaseModel):
    id: str
    parent_id: str
    query: str
    expected_outputs: list[str] = Field(default_factory=list)


class ResearchTask(BaseModel):
    id: str
    user_query: str
    subtasks: list[SubTask] = Field(default_factory=list)
    state: AgentState = AgentState.IDLE
    trace_id: str = "-"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: list[str] = Field(default_factory=list)
