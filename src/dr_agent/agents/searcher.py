"""Searcher Agent: for each SubTask run web search + (optional) page fetch.

Phase-2 minimal flow:
  1. Tavily search (advanced, max_results=5) per SubTask
  2. Use the cleaned ``content`` field returned by Tavily directly as a Chunk
  3. Fall back to fetcher only when content is too short (< 200 chars)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from loguru import logger

from dr_agent.agents.base import AbstractAgent, AgentContext
from dr_agent.orchestrator.envelope import ResultEnvelope
from dr_agent.schemas.task import SubTask
from dr_agent.tools.fetcher import Chunk, Fetcher
from dr_agent.tools.search import SearchResult, WebSearcher

_FETCH_FALLBACK_THRESHOLD = 200  # chars; below this we re-fetch


@dataclass
class SearcherInput:
    subtasks: list[SubTask]
    max_results_per_query: int = 5


@dataclass
class SubTaskSearchOutput:
    subtask: SubTask
    chunks: list[Chunk]
    raw_results: list[SearchResult]


@dataclass
class SearcherOutput:
    by_subtask: list[SubTaskSearchOutput]


class Searcher(AbstractAgent[SearcherInput, SearcherOutput]):
    name = "searcher"

    def __init__(
        self,
        searcher: WebSearcher,
        fetcher: Fetcher | None = None,
        *,
        per_query_concurrency: int = 4,
    ) -> None:
        self._searcher = searcher
        self._fetcher = fetcher
        self._sem = asyncio.Semaphore(per_query_concurrency)

    async def run(
        self, inp: SearcherInput, ctx: AgentContext
    ) -> ResultEnvelope[SearcherOutput]:
        outs = await asyncio.gather(
            *(self._run_one(st, inp.max_results_per_query) for st in inp.subtasks),
            return_exceptions=True,
        )
        cleaned: list[SubTaskSearchOutput] = []
        for st, o in zip(inp.subtasks, outs, strict=True):
            if isinstance(o, Exception):
                logger.warning("subtask {} search error: {}", st.id, o)
                cleaned.append(SubTaskSearchOutput(subtask=st, chunks=[], raw_results=[]))
            else:
                cleaned.append(o)
        total_chunks = sum(len(o.chunks) for o in cleaned)
        logger.info(
            "searcher done: {} subtasks -> {} total chunks",
            len(cleaned),
            total_chunks,
        )
        return ResultEnvelope.success(SearcherOutput(by_subtask=cleaned))

    async def _run_one(
        self, st: SubTask, max_results: int
    ) -> SubTaskSearchOutput:
        async with self._sem:
            results = await self._searcher.search(
                st.query, max_results=max_results, search_depth="advanced"
            )
        chunks = await self._results_to_chunks(results)
        return SubTaskSearchOutput(subtask=st, chunks=chunks, raw_results=results)

    async def _results_to_chunks(
        self, results: list[SearchResult]
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        fetch_targets: list[SearchResult] = []
        for r in results:
            if r.content and len(r.content) >= _FETCH_FALLBACK_THRESHOLD:
                chunks.append(
                    Chunk(url=r.url, title=r.title, text=r.content, source_idx=0)
                )
            elif self._fetcher is not None:
                fetch_targets.append(r)
        if fetch_targets and self._fetcher is not None:
            fetched = await asyncio.gather(
                *(self._fetcher.fetch(r.url, title=r.title) for r in fetch_targets),
                return_exceptions=True,
            )
            for r, f in zip(fetch_targets, fetched, strict=True):
                if isinstance(f, Exception):
                    logger.debug("fetch failed for {}: {}", r.url, f)
                    continue
                chunks.extend(f)
        return chunks
