"""Red Agent: produces structured attacks against a research report.

Four attack dimensions: factual / logic / citation / completeness.

Output schema (JSON):
    {
      "attacks": [
        {
          "section_id": "<id from input>",
          "type": "factual" | "logic" | "citation" | "completeness",
          "span": "<exact substring of the section body>",
          "evidence": "<why this is wrong / missing / weak>",
          "severity": 0.0..1.0,
          "suggested_action": "ADD" | "DELETE" | "MODIFY" | "VERIFY"
        },
        ...
      ]
    }

Shared parsing (serialize / coerce / 3-layer fallback) lives in
``agents/_attack_parse.py`` and is reused by the Multi-Critic agent.
"""

from __future__ import annotations

from dataclasses import dataclass

from dr_agent.agents._attack_parse import parse_with_fallback, serialize_report
from dr_agent.agents.base import AbstractAgent, AgentContext
from dr_agent.orchestrator.envelope import ResultEnvelope
from dr_agent.schemas.attack import Attack
from dr_agent.schemas.report import ResearchReport

_SYSTEM = """你是一名严格但具建设性的研究报告审稿人。你将审视一份带引用的研究报告，从四个维度提出**结构化攻击**：

1. factual：事实性错误，与公认事实冲突或与所引来源矛盾
2. logic：内部逻辑不一致、推理跳跃或自相矛盾
3. citation：引用缺失、引用与论断不符、关键数据点未标注来源
4. completeness：在子问题范围内，应当被讨论但缺失的关键面向（如对比维度缺失、关键案例未提及、术语未解释）

要求：
- 严格输出 JSON，键名固定为 attacks
- 每条攻击 span 必须是被攻击章节正文中的 EXACT 子串（10 字以上，用于后续替换或追加锚点）
- **若 span 中含有 [n] 形式的引用标记，必须原样保留在 span 字段中**
- severity ∈ [0.0, 1.0]
  - factual / logic 类问题严重：建议 ≥ 0.6
  - citation 缺失：建议 0.4-0.6
  - completeness 改进：建议 0.3-0.5
- suggested_action ∈ {ADD, DELETE, MODIFY, VERIFY}
  - ADD：在 span 之后追加内容（completeness 类常用）
  - DELETE：删掉 span（仅 factual 严重错误时使用）
  - MODIFY：把 span 改写为正确表述（factual / logic 类常用）
  - VERIFY：保留原文，但需要补充佐证（citation 类常用）
- 优先级与配额：**优先 factual 与 citation**；同一轮中 completeness 类至多 2 条
- 输出至多 8 条攻击，按 severity 从高到低排序；如确实没有任何问题，返回 {"attacks": []}
- 不要凭空捏造问题；宁可少不可错；不要给出风格类（措辞、行文）的攻击"""

_USER_TEMPLATE = """以下是待审视的研究报告（每段标有 section_id）：

{report_text}

请输出符合下述 JSON 格式的攻击列表：
{{
  "attacks": [
    {{"section_id": "...", "type": "factual|logic|citation|completeness",
      "span": "...", "evidence": "...", "severity": 0.0,
      "suggested_action": "ADD|DELETE|MODIFY|VERIFY"}}
  ]
}}"""


@dataclass
class RedInput:
    report: ResearchReport


@dataclass
class RedOutput:
    attacks: list[Attack]
    raw_responses: list[str]
    parse_strategy_used: str  # "direct" / "retry-strict" / "regex" / "empty"


class Red(AbstractAgent[RedInput, RedOutput]):
    name = "red"

    async def run(
        self, inp: RedInput, ctx: AgentContext
    ) -> ResultEnvelope[RedOutput]:
        report_text = serialize_report(inp.report)
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _USER_TEMPLATE.format(report_text=report_text)},
        ]
        attacks, raw_responses, strategy = await parse_with_fallback(
            ctx.pool.chat, messages, source="red", max_tokens=2000
        )
        return ResultEnvelope.success(
            RedOutput(
                attacks=attacks,
                raw_responses=raw_responses,
                parse_strategy_used=strategy,
            )
        )
