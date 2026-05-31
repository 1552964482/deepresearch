"""ResultEnvelope: explicit success/failure container that prevents
exception propagation between concurrent DAG branches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class ResultEnvelope(Generic[T]):
    ok: bool
    value: T | None = None
    error: BaseException | None = None
    elapsed_s: float = 0.0
    trace_id: str = "-"
    meta: dict = field(default_factory=dict)

    @classmethod
    def success(cls, value: T, *, elapsed_s: float = 0.0, trace_id: str = "-",
                meta: dict | None = None) -> "ResultEnvelope[T]":
        return cls(ok=True, value=value, elapsed_s=elapsed_s, trace_id=trace_id,
                   meta=meta or {})

    @classmethod
    def failure(cls, error: BaseException, *, elapsed_s: float = 0.0,
                trace_id: str = "-", meta: dict | None = None) -> "ResultEnvelope[T]":
        return cls(ok=False, error=error, elapsed_s=elapsed_s, trace_id=trace_id,
                   meta=meta or {})

    def unwrap(self) -> T:
        if not self.ok:
            assert self.error is not None
            raise self.error
        assert self.value is not None
        return self.value
