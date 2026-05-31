"""Multi-Critic review: three role-specialized critics in place of a single Red.

Motivation
----------
A single Red agent must juggle four attack dimensions at once, which dilutes
its focus and produces noisy, low-precision attacks. Multi-Critic instead
runs three *persona-specialized* critics in parallel:

  * **FactChecker**   — only factual errors / unsupported claims
  * **LogicReviewer** — only internal inconsistencies / reasoning gaps
  * **CitationAuditor** — only citation gaps / mismatches / unsourced data

Their attacks are then merged by a **two-signal consensus**:

  1. *Span-level dedup* — near-duplicate spans (cosine >= merge_threshold via
     bge-small-zh) collapse to their highest-severity representative.
  2. *Section-level hot-spot* — because the three personas cover disjoint
     dimensions, they rarely attack the exact same span; instead, a section
     flagged by multiple personas is treated as a problem hot-spot and its
     attacks are up-weighted. Isolated single-critic attacks are mildly
     down-weighted.

The design intent is **precision over recall**: each persona is narrowly
scoped, so its attacks are higher-quality and more likely to be accepted by
Blue, compared with a single Red juggling four dimensions at once.

The output is a plain ``list[Attack]`` so the existing Blue agent and the
K-round loop consume it unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from dr_agent.agents._attack_parse import parse_with_fallback, serialize_report
from dr_agent.agents.base import AbstractAgent, AgentContext
from dr_agent.memory.embedder import Embedder
from dr_agent.orchestrator.envelope import ResultEnvelope
from dr_agent.schemas.attack import Attack
from dr_agent.schemas.report import ResearchReport


# ---- persona prompts ----

_COMMON_RULES = """要求：
- 严格输出 JSON，键名固定为 attacks
- 每条攻击 span 必须是报告正文中的 EXACT 子串（10 字以上）
- 若 span 含 [n] 引用标记，必须原样保留在 span 字段中
- severity ∈ [0.0, 1.0]
- suggested_action ∈ {ADD, DELETE, MODIFY, VERIFY}
- 只在你的专属职责范围内提攻击；不要越界到其他维度
- 至多 5 条，按 severity 从高到低；无问题则返回 {"attacks": []}
- 不要凭空捏造；不要给风格类（措辞）攻击"""

_FACT_SYS = f"""你是一名**事实核查专员**。你只负责一件事：找出报告中的事实性错误、
与公认事实冲突、或缺乏依据的断言。type 一律填 "factual"。

典型问题：错误的年份/人物/数字、张冠李戴、与所引来源矛盾、过时的事实。
{_COMMON_RULES}"""

_LOGIC_SYS = f"""你是一名**逻辑审稿人**。你只负责一件事：找出报告中的逻辑问题——
内部自相矛盾、推理跳跃、因果倒置、以偏概全。type 一律填 "logic"。

典型问题：前后结论冲突、从个例推全称、缺失必要前提、循环论证。
{_COMMON_RULES}"""

_CITE_SYS = f"""你是一名**引用审计员**。你只负责一件事：检查引用的完整性与对齐——
关键数据/断言缺引用、引用与论断不符、来源质量低。type 一律填 "citation"，
suggested_action 优先 VERIFY 或 ADD。
{_COMMON_RULES}"""

_USER_TEMPLATE = """以下是待审视的研究报告（每段标有 section_id）：

{report_text}

请基于你的专属职责输出 JSON：
{{"attacks": [{{"section_id":"...","type":"...","span":"...",
  "evidence":"...","severity":0.0,"suggested_action":"..."}}]}}"""


_PERSONAS: list[tuple[str, str]] = [
    ("fact_checker", _FACT_SYS),
    ("logic_reviewer", _LOGIC_SYS),
    ("citation_auditor", _CITE_SYS),
]


@dataclass
class MultiCriticInput:
    report: ResearchReport


@dataclass
class CriticTrace:
    critic: str
    n_attacks: int
    parse_strategy: str


@dataclass
class MultiCriticOutput:
    attacks: list[Attack]
    per_critic: list[CriticTrace] = field(default_factory=list)
    n_raw: int = 0          # total attacks before consensus merge
    n_merged: int = 0       # clusters after merge
    n_consensus: int = 0    # clusters flagged by >= 2 critics
    raw_responses: list[str] = field(default_factory=list)
    parse_strategy_used: str = "multi-critic"


class MultiCritic(AbstractAgent[MultiCriticInput, MultiCriticOutput]):
    """Three persona critics + semantic consensus merge."""

    name = "multi_critic"

    def __init__(
        self,
        embedder: Embedder,
        *,
        merge_threshold: float = 0.82,
        consensus_boost: float = 0.15,
        lone_penalty: float = 0.10,
        max_final: int = 8,
    ) -> None:
        self._embedder = embedder
        self._merge_threshold = merge_threshold
        self._consensus_boost = consensus_boost
        self._lone_penalty = lone_penalty
        self._max_final = max_final

    async def run(
        self, inp: MultiCriticInput, ctx: AgentContext
    ) -> ResultEnvelope[MultiCriticOutput]:
        import asyncio

        report_text = serialize_report(inp.report)

        async def run_persona(name: str, sys_prompt: str) -> tuple[CriticTrace, list[Attack], list[str]]:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": _USER_TEMPLATE.format(report_text=report_text)},
            ]
            attacks, raws, strategy = await parse_with_fallback(
                ctx.pool.chat, messages, source=name, max_tokens=1600
            )
            return CriticTrace(critic=name, n_attacks=len(attacks), parse_strategy=strategy), attacks, raws

        results = await asyncio.gather(
            *(run_persona(n, p) for n, p in _PERSONAS),
            return_exceptions=True,
        )

        traces: list[CriticTrace] = []
        all_attacks: list[Attack] = []
        raw_responses: list[str] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning("critic failed: {}", r)
                continue
            trace, attacks, raws = r
            traces.append(trace)
            all_attacks.extend(attacks)
            raw_responses.extend(raws)

        n_raw = len(all_attacks)
        merged, n_consensus = self._consensus_merge(all_attacks)
        merged = merged[: self._max_final]

        logger.info(
            "multi-critic: {} raw attacks -> {} merged ({} consensus) from {} critics",
            n_raw,
            len(merged),
            n_consensus,
            len(traces),
        )

        return ResultEnvelope.success(
            MultiCriticOutput(
                attacks=merged,
                per_critic=traces,
                n_raw=n_raw,
                n_merged=len(merged),
                n_consensus=n_consensus,
                raw_responses=raw_responses,
            )
        )

    def _consensus_merge(self, attacks: list[Attack]) -> tuple[list[Attack], int]:
        """Cluster near-duplicate spans, then apply section-level consensus.

        Two signals:
          1. **Span-level dedup** — near-identical spans (cosine >= threshold)
             are collapsed to their highest-severity representative. This
             mainly removes accidental overlap between critics.
          2. **Section-level consensus** — because the three personas cover
             disjoint dimensions (factual / logic / citation), they rarely
             attack the *same span*; instead, a section flagged by multiple
             personas is a hot-spot. Attacks in such sections get a small
             severity boost; isolated single-critic attacks get a mild
             penalty.

        Returns (merged_attacks_sorted_by_severity_desc, n_consensus_sections).
        """
        if not attacks:
            return [], 0
        spans = [a.span for a in attacks]
        vecs = self._embedder.encode(spans)  # (N, D), L2-normalized

        n = len(attacks)
        used = [False] * n
        clusters: list[list[int]] = []
        sims = vecs @ vecs.T
        for i in range(n):
            if used[i]:
                continue
            group = [i]
            used[i] = True
            for j in range(i + 1, n):
                if used[j]:
                    continue
                if float(sims[i, j]) >= self._merge_threshold:
                    group.append(j)
                    used[j] = True
            clusters.append(group)

        # Section-level critic counts: how many distinct personas touched
        # each section_id.
        section_critics: dict[str, set[str]] = {}
        for a in attacks:
            sec = str(a.__dict__.get("section_id", ""))
            section_critics.setdefault(sec, set()).add(
                str(a.__dict__.get("critic", "?"))
            )
        consensus_sections = {
            sec for sec, crits in section_critics.items() if len(crits) >= 2
        }

        merged: list[Attack] = []
        for group in clusters:
            members = [attacks[k] for k in group]
            span_critics = {m.__dict__.get("critic", "?") for m in members}
            rep = max(members, key=lambda m: m.severity)
            sev = rep.severity
            sec = str(rep.__dict__.get("section_id", ""))
            # Span-level multi-critic OR section-level hot-spot -> boost.
            if len(span_critics) >= 2 or sec in consensus_sections:
                sev = min(1.0, sev + self._consensus_boost)
            elif len(span_critics) == 1:
                sev = max(0.0, sev - self._lone_penalty)
            new = Attack(
                id=rep.id,
                type=rep.type,
                span=rep.span,
                evidence=rep.evidence,
                severity=sev,
            )
            new.__dict__["section_id"] = sec
            new.__dict__["suggested_action"] = rep.__dict__.get("suggested_action")
            new.__dict__["critic"] = ",".join(sorted(span_critics))
            new.__dict__["n_critics"] = len(span_critics)
            merged.append(new)

        merged.sort(key=lambda a: a.severity, reverse=True)
        return merged, len(consensus_sections)
