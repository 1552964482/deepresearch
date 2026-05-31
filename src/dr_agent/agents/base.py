"""Abstract Agent base class."""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from loguru import logger

from dr_agent.llm.pool import MimoPool
from dr_agent.orchestrator.envelope import ResultEnvelope
from dr_agent.utils.trace import get_trace_id

InT = TypeVar("InT")
OutT = TypeVar("OutT")


@dataclass
class AgentContext:
    pool: MimoPool
    # memory and other shared resources are added in later phases
    extras: dict[str, Any] | None = None


class AbstractAgent(ABC, Generic[InT, OutT]):
    name: str = "agent"

    @abstractmethod
    async def run(self, inp: InT, ctx: AgentContext) -> ResultEnvelope[OutT]: ...

    async def _safe_run(self, inp: InT, ctx: AgentContext) -> ResultEnvelope[OutT]:
        t0 = time.monotonic()
        tid = get_trace_id()
        try:
            logger.debug("{} start", self.name)
            res = await self.run(inp, ctx)
            elapsed = time.monotonic() - t0
            res.elapsed_s = elapsed
            res.trace_id = tid
            logger.debug("{} done in {:.2f}s ok={}", self.name, elapsed, res.ok)
            return res
        except Exception as e:  # noqa: BLE001
            elapsed = time.monotonic() - t0
            logger.error("{} failed in {:.2f}s: {!r}", self.name, elapsed, e)
            return ResultEnvelope.failure(e, elapsed_s=elapsed, trace_id=tid)


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def parse_json_lenient(text: str) -> Any | None:
    """Best-effort JSON extraction.

    Order:
      1. Direct json.loads on stripped text.
      2. Strip a markdown code fence (```json ... ```).
      3. Regex-extract the first {...} or [...] block.
    """
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip ```json fence
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```\s*$", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    for pattern in (_JSON_BLOCK_RE, _JSON_ARRAY_RE):
        m = pattern.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    return None
