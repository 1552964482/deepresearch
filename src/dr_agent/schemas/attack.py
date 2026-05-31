"""Red Agent attack schema."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class AttackType(str, Enum):
    FACTUAL = "factual"
    LOGIC = "logic"
    CITATION = "citation"
    COMPLETENESS = "completeness"


class Attack(BaseModel):
    id: str
    type: AttackType
    span: str
    evidence: str
    severity: float = Field(ge=0.0, le=1.0)
