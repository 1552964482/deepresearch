"""Planner Agent: split a user query into 3-8 well-scoped SubTasks.

Output schema (JSON):
    {"subtasks": [{"query": str, "expected_outputs": [str, ...]}, ...]}
"""

from __future__ import annotations

import secrets
from typing import Any

from loguru import logger

from dr_agent.agents.base import AbstractAgent, AgentContext, parse_json_lenient
from dr_agent.orchestrator.envelope import ResultEnvelope
from dr_agent.schemas.task import ResearchTask, SubTask

_SYS = """你是一名研究规划专家。给定一个研究问题，你需要将其拆解为 3-8 个互不重叠、可独立检索的子问题。

要求：
- 子问题之间逻辑递进或互补，不重复
- 每个子问题给出 1-3 个 expected_outputs（如：定义、关键数据、典型案例、对比分析、最新进展）
- 严格输出 JSON，不要任何解释文字

输出格式：
{
  "subtasks": [
    {"query": "...", "expected_outputs": ["...", "..."]},
    ...
  ]
}
"""


class Planner(AbstractAgent[ResearchTask, list[SubTask]]):
    name = "planner"

    async def run(
        self, inp: ResearchTask, ctx: AgentContext
    ) -> ResultEnvelope[list[SubTask]]:
        messages = [
            {"role": "system", "content": _SYS},
            {"role": "user", "content": f"研究问题：{inp.user_query}"},
        ]
        # First attempt: ask for JSON mode.
        result = await ctx.pool.chat(
            messages, json_mode=True, temperature=0.4, max_tokens=1500
        )
        parsed = parse_json_lenient(result.content)

        # Second attempt: retry with stricter instruction if first parse failed.
        if not isinstance(parsed, dict) or "subtasks" not in parsed:
            logger.warning("planner first parse failed, retrying with stricter prompt")
            retry_messages = messages + [
                {"role": "assistant", "content": result.content},
                {
                    "role": "user",
                    "content": '上一次输出不符合 JSON 格式。请只输出 {"subtasks": [...]} JSON，不要任何 markdown 或解释。',
                },
            ]
            result = await ctx.pool.chat(
                retry_messages, json_mode=True, temperature=0.2, max_tokens=1500
            )
            parsed = parse_json_lenient(result.content)

        subtasks = self._materialize(parsed, inp)
        if not subtasks:
            # Fallback: single subtask = original query, allows pipeline to continue.
            logger.warning("planner produced no subtasks, falling back to single task")
            subtasks = [
                SubTask(
                    id=f"st-{secrets.token_hex(3)}",
                    parent_id=inp.id,
                    query=inp.user_query,
                    expected_outputs=["综合回答"],
                )
            ]
        return ResultEnvelope.success(subtasks, meta={"raw_content": result.content})

    @staticmethod
    def _materialize(parsed: Any, task: ResearchTask) -> list[SubTask]:
        if not isinstance(parsed, dict):
            return []
        items = parsed.get("subtasks", [])
        if not isinstance(items, list):
            return []
        out: list[SubTask] = []
        for item in items[:8]:  # cap at 8
            if not isinstance(item, dict):
                continue
            q = item.get("query")
            if not isinstance(q, str) or not q.strip():
                continue
            eo = item.get("expected_outputs", [])
            if not isinstance(eo, list):
                eo = []
            out.append(
                SubTask(
                    id=f"st-{secrets.token_hex(3)}",
                    parent_id=task.id,
                    query=q.strip(),
                    expected_outputs=[str(x) for x in eo if x],
                )
            )
        return out
