"""Orchestrator runners.

* :func:`run_minimal`   — Phase-1 Planner -> Writer (no retrieval).
* :func:`run_grounded`  — Phase-2 Planner -> Searcher -> Reader -> Writer
                           with three-stage compression and shared memory.
"""

from __future__ import annotations

import asyncio
import secrets
import time

from loguru import logger

from dr_agent.agents.base import AgentContext
from dr_agent.agents.planner import Planner
from dr_agent.agents.reader import Reader, ReaderInput
from dr_agent.agents.searcher import Searcher, SearcherInput
from dr_agent.agents.writer import GroundedWriter, GroundedWriterInput, Writer, WriterInput
from dr_agent.config import OrchestratorConfig
from dr_agent.llm.judge import JudgeClient
from dr_agent.llm.pool import MimoPool
from dr_agent.memory.compress import Compressor
from dr_agent.memory.embedder import Embedder
from dr_agent.memory.store import MemoryStore
from dr_agent.orchestrator.red_blue_loop import RedBlueResult, run_red_blue_loop
from dr_agent.orchestrator.state_machine import StateMachine
from dr_agent.schemas.report import ResearchReport
from dr_agent.schemas.task import ResearchTask
from dr_agent.tools.fetcher import Fetcher
from dr_agent.tools.search import WebSearcher
from dr_agent.utils.trace import trace_scope


async def run_minimal(
    user_query: str,
    pool: MimoPool,
    *,
    config: OrchestratorConfig | None = None,
    trace_id: str | None = None,
) -> tuple[ResearchReport, StateMachine]:
    """Run Planner -> Writer end-to-end with state-machine tracking."""
    config = config or OrchestratorConfig()
    sm = StateMachine()
    ctx = AgentContext(pool=pool)

    with trace_scope(trace_id) as tid:
        task = ResearchTask(
            id=f"rt-{secrets.token_hex(4)}",
            user_query=user_query,
            trace_id=tid,
        )
        logger.info("research task started: query={!r}", user_query)
        sm.fire("start")  # IDLE -> PLANNING

        # ---- Planner with timeout ----
        t0 = time.monotonic()
        planner = Planner()
        try:
            plan_env = await asyncio.wait_for(
                planner._safe_run(task, ctx),
                timeout=config.subtask_timeout_s,
            )
        except asyncio.TimeoutError:
            sm.fire("plan_error")
            raise RuntimeError("planner timed out") from None

        if not plan_env.ok or not plan_env.value:
            sm.fire("plan_error")
            raise RuntimeError(f"planner failed: {plan_env.error}")

        subtasks = plan_env.value
        task.subtasks = subtasks
        logger.info(
            "planner produced {} subtasks in {:.2f}s", len(subtasks), plan_env.elapsed_s
        )
        # Phase-1 path skips search/read/compress and goes straight to writing.
        sm.fire("plan_skip_search")  # PLANNING -> WRITING

        # ---- Writer with global timeout guard ----
        elapsed = time.monotonic() - t0
        remaining = max(config.global_timeout_s - elapsed, 30.0)
        writer = Writer()
        try:
            write_env = await asyncio.wait_for(
                writer._safe_run(WriterInput(task=task, subtasks=subtasks), ctx),
                timeout=remaining,
            )
        except asyncio.TimeoutError:
            sm.fire("global_timeout")
            sm.fire("force_converge")
            raise RuntimeError("writer timed out, forced convergence") from None

        if not write_env.ok or write_env.value is None:
            raise RuntimeError(f"writer failed: {write_env.error}")

        sm.fire("draft_skip_review")  # WRITING -> DONE
        return write_env.value, sm


async def run_grounded(
    user_query: str,
    pool: MimoPool,
    *,
    embedder: Embedder,
    memory: MemoryStore,
    web_searcher: WebSearcher,
    fetcher: Fetcher | None = None,
    compressor: Compressor | None = None,
    config: OrchestratorConfig | None = None,
    max_results_per_query: int = 5,
    review_rounds: int = 0,
    judge: JudgeClient | None = None,
    in_loop_judge: bool = False,
    reviewer: str = "red",
    depth: int = 0,
    breadth: int = 2,
    deepdive_max_searches: int = 24,
    trace_id: str | None = None,
) -> tuple[ResearchReport, StateMachine, RedBlueResult | None]:
    """Phase-2/3/4 pipeline: Planner -> Searcher -> Reader
    [-> recursive deep-dive if depth>0] -> Writer
    [-> review loop if review_rounds > 0]."""
    config = config or OrchestratorConfig()
    compressor = compressor or Compressor(embedder)
    sm = StateMachine()
    ctx = AgentContext(pool=pool)

    with trace_scope(trace_id) as tid:
        task = ResearchTask(
            id=f"rt-{secrets.token_hex(4)}",
            user_query=user_query,
            trace_id=tid,
        )
        logger.info("[grounded] task started: query={!r}", user_query)
        sm.fire("start")  # IDLE -> PLANNING

        # ---- 1. Planner ----
        t0 = time.monotonic()
        planner = Planner()
        try:
            plan_env = await asyncio.wait_for(
                planner._safe_run(task, ctx),
                timeout=config.subtask_timeout_s,
            )
        except asyncio.TimeoutError:
            sm.fire("plan_error")
            raise RuntimeError("planner timed out") from None
        if not plan_env.ok or not plan_env.value:
            sm.fire("plan_error")
            raise RuntimeError(f"planner failed: {plan_env.error}")
        subtasks = plan_env.value
        task.subtasks = subtasks
        logger.info("planner -> {} subtasks in {:.2f}s", len(subtasks), plan_env.elapsed_s)
        sm.fire("plan_ok")  # PLANNING -> SEARCHING

        # ---- 2. Searcher (parallel across subtasks) ----
        searcher = Searcher(web_searcher, fetcher=fetcher, per_query_concurrency=4)
        try:
            search_env = await asyncio.wait_for(
                searcher._safe_run(
                    SearcherInput(
                        subtasks=subtasks, max_results_per_query=max_results_per_query
                    ),
                    ctx,
                ),
                timeout=config.global_timeout_s - (time.monotonic() - t0),
            )
        except asyncio.TimeoutError:
            sm.fire("global_timeout")
            sm.fire("force_converge")
            raise RuntimeError("searcher timed out") from None
        if not search_env.ok or search_env.value is None:
            sm.fire("search_all_fail")
            raise RuntimeError(f"searcher failed: {search_env.error}")
        search_out = search_env.value
        logger.info(
            "search done in {:.2f}s ({} subtasks)",
            search_env.elapsed_s,
            len(search_out.by_subtask),
        )
        sm.fire("search_ok")  # SEARCHING -> READING

        # ---- 3. Reader (compress + persist to memory) ----
        reader = Reader(compressor, memory, per_subtask_concurrency=6)
        try:
            read_env = await asyncio.wait_for(
                reader._safe_run(
                    ReaderInput(task_id=task.id, subtask_outputs=search_out.by_subtask),
                    ctx,
                ),
                timeout=config.global_timeout_s - (time.monotonic() - t0),
            )
        except asyncio.TimeoutError:
            sm.fire("global_timeout")
            sm.fire("force_converge")
            raise RuntimeError("reader timed out") from None
        if not read_env.ok or read_env.value is None:
            raise RuntimeError(f"reader failed: {read_env.error}")
        read_out = read_env.value
        logger.info("reader done in {:.2f}s", read_env.elapsed_s)
        sm.fire("read_ok")          # READING -> COMPRESSING (compression done inside reader)

        # ---- 3b. Recursive deep-dive (optional) ----
        deepdive_subtask_outputs = read_out.by_subtask
        if depth > 0:
            from dr_agent.agents.gap_analyzer import GapAnalyzer
            from dr_agent.orchestrator.deepdive import deep_dive

            deepdive_subtask_outputs, dd_stats = await deep_dive(
                read_out.by_subtask,
                task_id=task.id,
                searcher=searcher,
                reader=reader,
                gap_analyzer=GapAnalyzer(),
                ctx=ctx,
                depth=depth,
                breadth=breadth,
                max_results_per_query=max_results_per_query,
                max_searches=deepdive_max_searches,
            )
            task.notes.append(
                f"deep-dive: +{dd_stats.facts_added} facts via "
                f"{dd_stats.followups_generated} followups "
                f"({dd_stats.searches_used}/{dd_stats.searches_budget} searches, "
                f"{dd_stats.stopped_reason})"
            )

        sm.fire("compress_ok")      # COMPRESSING -> WRITING

        # ---- 4. Writer (grounded) ----
        writer = GroundedWriter()
        remaining = max(config.global_timeout_s - (time.monotonic() - t0), 60.0)
        try:
            write_env = await asyncio.wait_for(
                writer._safe_run(
                    GroundedWriterInput(task=task, read_outputs=deepdive_subtask_outputs),
                    ctx,
                ),
                timeout=remaining,
            )
        except asyncio.TimeoutError:
            sm.fire("global_timeout")
            sm.fire("force_converge")
            raise RuntimeError("writer timed out, forced convergence") from None
        if not write_env.ok or write_env.value is None:
            raise RuntimeError(f"writer failed: {write_env.error}")

        draft = write_env.value
        if review_rounds <= 0:
            sm.fire("draft_skip_review")  # WRITING -> DONE
            return draft, sm, None

        # ---- 5. Adversarial review (Phase 3 / Multi-Critic) ----
        rb_result = await run_red_blue_loop(
            draft,
            ctx,
            sm=sm,
            max_rounds=review_rounds,
            judge=judge,
            in_loop_judge=in_loop_judge,
            user_query=user_query,
            reviewer=reviewer,
            embedder=embedder,
        )
        return rb_result.final_report, sm, rb_result
