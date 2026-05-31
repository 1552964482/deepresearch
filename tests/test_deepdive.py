"""Offline tests for the recursive deep-dive orchestrator and its budget.

We stub Searcher / Reader / GapAnalyzer with lightweight fakes so no LLM or
network is involved.
"""

from __future__ import annotations

import pytest

from dr_agent.agents.gap_analyzer import GapOutput
from dr_agent.agents.reader import ReaderOutput, SubTaskReadOutput
from dr_agent.agents.searcher import SearcherOutput, SubTaskSearchOutput
from dr_agent.memory.compress import CompressedSentence, CompressResult, CompressStats
from dr_agent.orchestrator.deepdive import _Budget, deep_dive
from dr_agent.orchestrator.envelope import ResultEnvelope
from dr_agent.schemas.report import Citation
from dr_agent.schemas.task import SubTask


# ---------- _Budget ----------


def test_budget_grants_up_to_cap() -> None:
    b = _Budget(cap=5)
    assert b.take(3) == 3
    assert b.take(3) == 2  # only 2 left
    assert b.take(1) == 0
    assert b.exhausted


# ---------- fakes ----------


def _fact(text: str, url: str = "https://x/1") -> CompressedSentence:
    return CompressedSentence(
        chunk_url=url, chunk_title="t", text=text, score=1.0, is_protected=False
    )


def _read_output(st: SubTask, n_facts: int) -> SubTaskReadOutput:
    return SubTaskReadOutput(
        subtask=st,
        facts=[_fact(f"fact {st.id} {i}") for i in range(n_facts)],
        citations=[Citation(id=f"c-{st.id}", title="src", url=f"https://x/{st.id}")],
        compress_result=CompressResult(sentences=[], stats=CompressStats()),
    )


class _FakeGap:
    """Returns followups for the first `n_levels` calls, then 'sufficient'."""

    def __init__(self, followups_per_call: int = 2, stop_after: int = 1) -> None:
        self.followups_per_call = followups_per_call
        self.stop_after = stop_after
        self.calls = 0

    async def _safe_run(self, inp, ctx):  # noqa: ANN001
        self.calls += 1
        if self.calls > self.stop_after:
            return ResultEnvelope.success(GapOutput(sufficient=True))
        fus = [
            SubTask(id=f"fu-{self.calls}-{i}", parent_id=inp.subtask.id,
                    query=f"followup {i}", expected_outputs=[])
            for i in range(self.followups_per_call)
        ]
        return ResultEnvelope.success(
            GapOutput(sufficient=False, missing_aspects=["x"], followups=fus)
        )


class _FakeSearcher:
    def __init__(self) -> None:
        self.searched: list[str] = []

    async def _safe_run(self, inp, ctx):  # noqa: ANN001
        outs = []
        for st in inp.subtasks:
            self.searched.append(st.id)
            outs.append(SubTaskSearchOutput(subtask=st, chunks=[], raw_results=[]))
        return ResultEnvelope.success(SearcherOutput(by_subtask=outs))


class _FakeReader:
    async def _safe_run(self, inp, ctx):  # noqa: ANN001
        outs = [_read_output(sto.subtask, 2) for sto in inp.subtask_outputs]
        return ResultEnvelope.success(ReaderOutput(by_subtask=outs))


# ---------- deep_dive ----------


@pytest.mark.asyncio
async def test_deep_dive_adds_facts() -> None:
    root = SubTask(id="root1", parent_id="t", query="q", expected_outputs=[])
    base = [_read_output(root, 3)]
    searcher, reader, gap = _FakeSearcher(), _FakeReader(), _FakeGap(2, stop_after=1)

    out, stats = await deep_dive(
        base, task_id="t", searcher=searcher, reader=reader, gap_analyzer=gap,
        ctx=None, depth=1, breadth=2, max_searches=24,
    )
    # 2 followups searched, each contributing 2 facts -> +4 facts.
    assert stats.followups_generated == 2
    assert stats.facts_added == 4
    assert len(out[0].facts) == 3 + 4


@pytest.mark.asyncio
async def test_deep_dive_respects_search_budget() -> None:
    roots = [
        SubTask(id=f"root{i}", parent_id="t", query=f"q{i}", expected_outputs=[])
        for i in range(5)
    ]
    base = [_read_output(r, 1) for r in roots]
    searcher, reader, gap = _FakeSearcher(), _FakeReader(), _FakeGap(2, stop_after=99)

    out, stats = await deep_dive(
        base, task_id="t", searcher=searcher, reader=reader, gap_analyzer=gap,
        ctx=None, depth=3, breadth=2, max_searches=3,  # tiny budget
    )
    # Never exceed the budget cap.
    assert stats.searches_used <= 3
    assert stats.stopped_reason == "search_budget_exhausted"


@pytest.mark.asyncio
async def test_deep_dive_stops_when_sufficient() -> None:
    root = SubTask(id="root1", parent_id="t", query="q", expected_outputs=[])
    base = [_read_output(root, 3)]
    searcher, reader = _FakeSearcher(), _FakeReader()
    gap = _FakeGap(2, stop_after=0)  # immediately sufficient

    out, stats = await deep_dive(
        base, task_id="t", searcher=searcher, reader=reader, gap_analyzer=gap,
        ctx=None, depth=2, breadth=2, max_searches=24,
    )
    assert stats.followups_generated == 0
    assert stats.facts_added == 0
    assert searcher.searched == []


@pytest.mark.asyncio
async def test_deep_dive_recurses_to_depth() -> None:
    root = SubTask(id="root1", parent_id="t", query="q", expected_outputs=[])
    base = [_read_output(root, 1)]
    searcher, reader = _FakeSearcher(), _FakeReader()
    # Always produce 1 followup so depth controls how deep we go.
    gap = _FakeGap(followups_per_call=1, stop_after=99)

    out, stats = await deep_dive(
        base, task_id="t", searcher=searcher, reader=reader, gap_analyzer=gap,
        ctx=None, depth=2, breadth=1, max_searches=24,
    )
    # depth=2 -> two expansion levels -> 2 followups total for the single root.
    assert stats.levels_explored == 2
    assert stats.followups_generated == 2
