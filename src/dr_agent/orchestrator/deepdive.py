"""Recursive deep-dive: expand each SubTask's facts by following up on gaps.

After the first Searcher+Reader pass produces facts per SubTask, the
deep-dive loop (depth = D, breadth = B) does, for each level:

  1. GapAnalyzer inspects the SubTask's accumulated facts.
  2. If insufficient, it proposes <= B follow-up sub-questions.
  3. Those follow-ups are searched + read (reusing Searcher/Reader), and the
     new facts are appended to the parent SubTask's fact list.
  4. Recurse on the follow-ups, up to depth D.

A **global search budget** (`max_searches`) hard-caps total Tavily calls so
a runaway expansion can't burn quota; when the budget is exhausted the dive
stops gracefully and returns whatever was gathered.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from loguru import logger

from dr_agent.agents.base import AgentContext
from dr_agent.agents.gap_analyzer import GapAnalyzer, GapInput
from dr_agent.agents.reader import Reader, ReaderInput, SubTaskReadOutput
from dr_agent.agents.searcher import Searcher, SearcherInput
from dr_agent.memory.compress import CompressedSentence
from dr_agent.schemas.report import Citation
from dr_agent.schemas.task import SubTask


@dataclass
class DeepDiveStats:
    levels_explored: int = 0
    followups_generated: int = 0
    searches_used: int = 0
    searches_budget: int = 0
    facts_added: int = 0
    stopped_reason: str = ""
    per_subtask_followups: dict[str, int] = field(default_factory=dict)


@dataclass
class _Budget:
    used: int = 0
    cap: int = 999

    def take(self, n: int) -> int:
        """Reserve up to n searches; return how many were actually granted."""
        remaining = max(self.cap - self.used, 0)
        grant = min(n, remaining)
        self.used += grant
        return grant

    @property
    def exhausted(self) -> bool:
        return self.used >= self.cap


async def deep_dive(
    read_outputs: list[SubTaskReadOutput],
    *,
    task_id: str,
    searcher: Searcher,
    reader: Reader,
    gap_analyzer: GapAnalyzer,
    ctx: AgentContext,
    depth: int = 1,
    breadth: int = 2,
    max_results_per_query: int = 3,
    max_searches: int = 24,
) -> tuple[list[SubTaskReadOutput], DeepDiveStats]:
    """Expand each top-level SubTask's facts via recursive follow-ups.

    Returns the (mutated) read_outputs with extra facts/citations folded in,
    plus stats. The original SubTaskReadOutput objects are replaced by new
    ones whose ``facts`` and ``citations`` include the deep-dive yield.
    """
    stats = DeepDiveStats(searches_budget=max_searches)
    budget = _Budget(cap=max_searches)

    # Map top-level SubTask id -> accumulator of facts/citations.
    accum: dict[str, SubTaskReadOutput] = {ro.subtask.id: ro for ro in read_outputs}

    async def expand(parent_id: str, frontier: list[SubTask], level: int) -> None:
        if level > depth or not frontier or budget.exhausted:
            return
        stats.levels_explored = max(stats.levels_explored, level)

        # 1. Gap analysis for each frontier subtask (parallel, cheap LLM calls).
        gap_envs = await asyncio.gather(
            *(
                gap_analyzer._safe_run(
                    GapInput(
                        subtask=st,
                        facts=[f.text for f in accum[parent_id].facts],
                        breadth=breadth,
                    ),
                    ctx,
                )
                for st in frontier
            )
        )
        followups: list[SubTask] = []
        for env in gap_envs:
            if env.ok and env.value and not env.value.sufficient:
                followups.extend(env.value.followups)
        if not followups:
            return

        # Respect the global search budget.
        grant = budget.take(len(followups))
        if grant <= 0:
            stats.stopped_reason = "search_budget_exhausted"
            return
        followups = followups[:grant]
        stats.followups_generated += len(followups)
        stats.per_subtask_followups[parent_id] = (
            stats.per_subtask_followups.get(parent_id, 0) + len(followups)
        )

        # 2. Search + 3. Read the follow-ups.
        search_env = await searcher._safe_run(
            SearcherInput(subtasks=followups, max_results_per_query=max_results_per_query),
            ctx,
        )
        stats.searches_used += len(followups)
        if not search_env.ok or search_env.value is None:
            return
        read_env = await reader._safe_run(
            ReaderInput(task_id=task_id, subtask_outputs=search_env.value.by_subtask),
            ctx,
        )
        if not read_env.ok or read_env.value is None:
            return

        # 4. Fold new facts/citations back into the parent accumulator.
        new_facts: list[CompressedSentence] = []
        new_cites: list[Citation] = []
        for sub_ro in read_env.value.by_subtask:
            new_facts.extend(sub_ro.facts)
            new_cites.extend(sub_ro.citations)
        if new_facts:
            parent = accum[parent_id]
            merged_facts = parent.facts + new_facts
            merged_cites = _dedupe_citations(parent.citations + new_cites)
            accum[parent_id] = SubTaskReadOutput(
                subtask=parent.subtask,
                facts=merged_facts,
                citations=merged_cites,
                compress_result=parent.compress_result,
            )
            stats.facts_added += len(new_facts)

        # 5. Recurse on the follow-ups (deeper level).
        if level < depth and not budget.exhausted:
            await expand(parent_id, followups, level + 1)

    # Kick off one expansion branch per top-level SubTask.
    await asyncio.gather(
        *(expand(ro.subtask.id, [ro.subtask], 1) for ro in read_outputs)
    )

    if not stats.stopped_reason:
        stats.stopped_reason = "completed" if not budget.exhausted else "search_budget_exhausted"

    logger.info(
        "deep-dive: depth<={}, breadth<={}, {} followups, {}/{} searches used, +{} facts ({})",
        depth,
        breadth,
        stats.followups_generated,
        stats.searches_used,
        stats.searches_budget,
        stats.facts_added,
        stats.stopped_reason,
    )
    return [accum[ro.subtask.id] for ro in read_outputs], stats


def _dedupe_citations(cites: list[Citation]) -> list[Citation]:
    seen: dict[str, Citation] = {}
    for c in cites:
        key = c.url or c.id
        if key not in seen:
            seen[key] = c
    return list(seen.values())
