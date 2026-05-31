"""Blue Agent patch schema."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class PatchAction(str, Enum):
    ADD = "ADD"
    DELETE = "DELETE"
    MODIFY = "MODIFY"
    VERIFY = "VERIFY"


class Patch(BaseModel):
    attack_id: str
    action: PatchAction
    target_span: str
    new_text: str | None = None
    verification_evidence: str | None = None
    accepted: bool = False
    rationale: str = Field(default="")

    @model_validator(mode="after")
    def _check_required(self) -> "Patch":
        if self.action in (PatchAction.ADD, PatchAction.MODIFY) and not self.new_text:
            raise ValueError(f"{self.action.value} patch requires new_text")
        if self.action is PatchAction.VERIFY and not self.verification_evidence:
            raise ValueError("VERIFY patch requires verification_evidence")
        return self
