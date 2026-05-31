"""Trace-id propagation via contextvars."""

from __future__ import annotations

import contextvars
import secrets
from contextlib import contextmanager
from typing import Iterator

_current_trace: contextvars.ContextVar[str] = contextvars.ContextVar(
    "dr_agent_trace_id", default="-"
)


def new_trace_id() -> str:
    return "tr-" + secrets.token_hex(4)


def get_trace_id() -> str:
    return _current_trace.get()


@contextmanager
def trace_scope(trace_id: str | None = None) -> Iterator[str]:
    tid = trace_id or new_trace_id()
    token = _current_trace.set(tid)
    try:
        yield tid
    finally:
        _current_trace.reset(token)


__all__ = ["new_trace_id", "get_trace_id", "trace_scope"]
