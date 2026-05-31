"""Reader Agent: run three-stage compression and persist key facts to shared memory."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from loguru import logger

from dr_agent.agents.base import AbstractAgent, AgentContext
from dr_agent.agents.searcher import SubTaskSearchOutput
from dr_agent.memory.compress import CompressedSentence, CompressResult, Compressor
from dr_agent.memory.store import MemoryStore
from dr_agent.orchestrator.envelope import ResultEnvelope
from dr_agent.schemas.report import Citation
from dr_agent.schemas.task import SubTask


@dataclass
class ReaderInput:
    task_id: str
    subtask_outputs: list[SubTaskSearchOutput]


@dataclass
class SubTaskReadOutput:
    subtask: SubTask
    facts: list[CompressedSentence]
    citations: list[Citation]
    compress_result: CompressResult


@dataclass
class ReaderOutput:
    by_subtask: list[SubTaskReadOutput]


class Reader(AbstractAgent[ReaderInput, ReaderOutput]):
    name = "reader"

    def __init__(
        self,
        compressor: Compressor,
        memory: MemoryStore,
        *,
        per_subtask_concurrency: int = 6,
    ) -> None:
        self._compressor = compressor
        self._memory = memory
        self._sem = asyncio.Semaphore(per_subtask_concurrency)

    async def run(
        self, inp: ReaderInput, ctx: AgentContext
    ) -> ResultEnvelope[ReaderOutput]:
        outs = await asyncio.gather(
            *(self._run_one(inp.task_id, sto) for sto in inp.subtask_outputs)
        )
        return ResultEnvelope.success(ReaderOutput(by_subtask=outs))

    async def _run_one(
        self, task_id: str, sto: SubTaskSearchOutput
    ) -> SubTaskReadOutput:
        async with self._sem:
            # Compression is CPU-ish (numpy + textrank); push to a thread.
            result = await asyncio.to_thread(
                lambda: self._compressor.compress(
                    sto.subtask.query, sto.chunks, token_budget=4000
                )
            )
            citations = [
                Citation(
                    id=f"c-{i}",
                    title=sr.title or sr.url,
                    url=sr.url,
                    snippet=(sr.content or "")[:200],
                )
                for i, sr in enumerate(sto.raw_results)
            ]
            # Persist facts to shared memory (with dedupe).
            written = 0
            duplicates = 0
            for sent in result.sentences:
                wr = await self._memory.write(
                    sent.text,
                    task_id=task_id,
                    agent_id=f"reader:{sto.subtask.id}",
                )
                if wr.written:
                    written += 1
                else:
                    duplicates += 1
            logger.info(
                "subtask {}: facts kept={}, dedup-skipped={}",
                sto.subtask.id,
                written,
                duplicates,
            )
            return SubTaskReadOutput(
                subtask=sto.subtask,
                facts=result.sentences,
                citations=citations,
                compress_result=result,
            )
