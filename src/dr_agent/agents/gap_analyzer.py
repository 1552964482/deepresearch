"""GapAnalyzer: decides whether a SubTask's gathered facts are sufficient,
and if not, proposes follow-up sub-questions for a deeper dive.

This is what turns the system from a single-layer RAG into a *recursive*
deep-research agent (cf. GPT-Researcher / dzhng deep-research breadth/depth).

Output schema (JSON):
    {
      "sufficient": true | false,
      "missing_aspects": ["...", ...],     # what's still unknown
      "followups": ["sub-question 1", ...] # at most `breadth` items
    }

If ``sufficient`` is true or ``followups`` is empty, recursion stops for
that branch.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field

from dr_agent.agents.base import AbstractAgent, AgentContext, parse_json_lenient
from dr_agent.orchestrator.envelope import ResultEnvelope
from dr_agent.schemas.task import SubTask

_SYSTEM = """你是一名研究深挖规划专家。给定一个子问题以及目前已检索到的关键事实，
判断这些事实是否足以充分回答该子问题；如不足，提出更深入的后续追问。

要求：
- 严格输出 JSON，键名固定为 sufficient / missing_aspects / followups
- sufficient: 布尔值。若现有事实已能充分、具体地回答子问题，填 true
- missing_aspects: 字符串数组，列出仍然缺失或模糊的关键面向（可为空）
- followups: 字符串数组，针对 missing_aspects 提出的、可独立检索的后续子问题
  - 每个 followup 必须比原子问题更**具体、更深入**（追问机制、数据、对比、案例、最新进展）
  - 不要重复原子问题；不要泛泛而问
  - 至多 {breadth} 个；若 sufficient 为 true 则返回空数组
- 宁缺毋滥：只有当深挖确实能显著提升回答质量时才提 followup"""

_USER_TEMPLATE = """原子问题：{query}

期望产出维度：{expected}

目前已检索到的关键事实（节选）：
{facts}

请判断是否需要更深入的追问，并按 JSON 格式输出。"""


@dataclass
class GapInput:
    subtask: SubTask
    facts: list[str]
    breadth: int = 2


@dataclass
class GapOutput:
    sufficient: bool
    missing_aspects: list[str] = field(default_factory=list)
    followups: list[SubTask] = field(default_factory=list)
    raw: str = ""


class GapAnalyzer(AbstractAgent[GapInput, GapOutput]):
    name = "gap_analyzer"

    async def run(self, inp: GapInput, ctx: AgentContext) -> ResultEnvelope[GapOutput]:
        facts_block = "\n".join(f"- {f}" for f in inp.facts[:40]) or "（暂无）"
        expected = "、".join(inp.subtask.expected_outputs) or "全面回答"
        messages = [
            {"role": "system", "content": _SYSTEM.format(breadth=inp.breadth)},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(
                    query=inp.subtask.query, expected=expected, facts=facts_block
                ),
            },
        ]
        result = await ctx.pool.chat(
            messages, json_mode=True, temperature=0.3, max_tokens=800
        )
        parsed = parse_json_lenient(result.content)
        return ResultEnvelope.success(
            self._materialize(parsed, inp, result.content)
        )

    @staticmethod
    def _materialize(parsed: object, inp: GapInput, raw: str) -> GapOutput:
        if not isinstance(parsed, dict):
            # On parse failure, treat as sufficient (stop recursion) — safe default.
            return GapOutput(sufficient=True, raw=raw)
        sufficient = bool(parsed.get("sufficient", True))
        missing = parsed.get("missing_aspects", [])
        if not isinstance(missing, list):
            missing = []
        fu_raw = parsed.get("followups", [])
        if not isinstance(fu_raw, list):
            fu_raw = []
        followups: list[SubTask] = []
        for q in fu_raw[: inp.breadth]:
            if not isinstance(q, str) or not q.strip():
                continue
            followups.append(
                SubTask(
                    id=f"st-{secrets.token_hex(3)}",
                    parent_id=inp.subtask.id,
                    query=q.strip(),
                    expected_outputs=[f"深挖：{inp.subtask.query}"],
                )
            )
        # If model said insufficient but gave no followups, nothing to expand.
        if not followups:
            sufficient = True
        return GapOutput(
            sufficient=sufficient,
            missing_aspects=[str(m) for m in missing if m],
            followups=followups,
            raw=raw,
        )
