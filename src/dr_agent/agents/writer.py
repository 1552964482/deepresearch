"""Writer Agent: produce a markdown research report.

Two modes:
  * ``WriterInput`` (Phase 1) — knowledge-only writing, no facts injected.
  * ``GroundedWriterInput`` (Phase 2+) — writing grounded in compressed
    facts and citations produced by the Reader Agent.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from loguru import logger

from dr_agent.agents.base import AbstractAgent, AgentContext
from dr_agent.agents.reader import SubTaskReadOutput
from dr_agent.orchestrator.envelope import ResultEnvelope
from dr_agent.schemas.report import Citation, ResearchReport, Section
from dr_agent.schemas.task import ResearchTask, SubTask

_SECTION_SYS = """你是一名研究报告作者。基于一个子问题，撰写报告中的一个章节。

要求：
- 中文输出
- 使用清晰的小标题与段落，必要时使用 markdown 列表
- 内容要紧扣子问题，避免空话；重要事实给出年份或来源说明（如有）
- 请坦诚标注不确定信息（如「据 2024 年公开资料……」）
- 只输出章节正文，不要包含一级标题或外层标题
"""

_SUMMARY_SYS = """你是一名研究报告主编。基于以下章节正文，写出 150-250 字的执行摘要。

要求：
- 概括核心结论，不展开细节
- 保留关键数字 / 年份 / 名词
- 中文输出，无 markdown 标题
"""


@dataclass
class WriterInput:
    task: ResearchTask
    subtasks: list[SubTask]


class Writer(AbstractAgent[WriterInput, ResearchReport]):
    name = "writer"

    async def run(
        self, inp: WriterInput, ctx: AgentContext
    ) -> ResultEnvelope[ResearchReport]:
        # Write all sections in parallel (will respect MimoPool concurrency).
        section_coros = [self._write_section(st, ctx) for st in inp.subtasks]
        sections = await asyncio.gather(*section_coros, return_exceptions=True)

        ok_sections: list[Section] = []
        for st, sec in zip(inp.subtasks, sections, strict=False):
            if isinstance(sec, Exception):
                logger.warning("section failed for subtask {}: {}", st.id, sec)
                ok_sections.append(
                    Section(
                        heading=f"{st.query}（生成失败）",
                        body=f"该章节生成失败：{sec}",
                        subtask_id=st.id,
                    )
                )
            else:
                ok_sections.append(sec)

        # Title is just the user query distilled; summary written from section bodies.
        body_concat = "\n\n".join(
            f"## {s.heading}\n{s.body}" for s in ok_sections
        )[:8000]
        summary = await self._write_summary(body_concat, ctx)
        title = self._derive_title(inp.task.user_query)

        report = ResearchReport(
            task_id=inp.task.id,
            user_query=inp.task.user_query,
            title=title,
            summary=summary,
            sections=ok_sections,
        )
        return ResultEnvelope.success(report)

    async def _write_section(self, st: SubTask, ctx: AgentContext) -> Section:
        eo = "、".join(st.expected_outputs) if st.expected_outputs else "全面回答"
        messages = [
            {"role": "system", "content": _SECTION_SYS},
            {
                "role": "user",
                "content": (
                    f"子问题：{st.query}\n"
                    f"期望产出维度：{eo}\n\n"
                    f"请写出该章节正文。"
                ),
            },
        ]
        result = await ctx.pool.chat(messages, temperature=0.5, max_tokens=1200)
        return Section(
            heading=st.query,
            body=result.content.strip(),
            subtask_id=st.id,
        )

    async def _write_summary(self, body: str, ctx: AgentContext) -> str:
        messages = [
            {"role": "system", "content": _SUMMARY_SYS},
            {"role": "user", "content": body},
        ]
        result = await ctx.pool.chat(messages, temperature=0.3, max_tokens=400)
        return result.content.strip()

    @staticmethod
    def _derive_title(query: str) -> str:
        q = query.strip()
        if len(q) > 40:
            q = q[:40] + "…"
        return f"研究报告：{q}"


_GROUNDED_SECTION_SYS = """你是一名严谨的研究报告作者。基于一个子问题以及检索得到的事实片段（带编号），撰写报告中的一个章节。

要求：
- 中文输出
- **必须**基于提供的事实片段，不要凭空编造
- 在使用事实时用 [n] 形式标注引用编号（n 来自事实编号），允许同一句话有多个引用
- 如某些事实彼此矛盾，简要说明并以更可信源为准
- 如事实不足以回答某点，可坦诚指出"现有资料中未能确认……"
- 使用 markdown 小标题、段落、列表，不要包含一级标题或外层标题"""


@dataclass
class GroundedWriterInput:
    task: ResearchTask
    read_outputs: list[SubTaskReadOutput]


class GroundedWriter(AbstractAgent[GroundedWriterInput, ResearchReport]):
    """Writer that grounds each section in compressed facts + citations."""

    name = "writer"

    async def run(
        self, inp: GroundedWriterInput, ctx: AgentContext
    ) -> ResultEnvelope[ResearchReport]:
        section_coros = [self._write_section(ro, ctx) for ro in inp.read_outputs]
        sections_or_exc = await asyncio.gather(*section_coros, return_exceptions=True)

        ok_sections: list[Section] = []
        for ro, sec in zip(inp.read_outputs, sections_or_exc, strict=True):
            if isinstance(sec, Exception):
                logger.warning("section failed for subtask {}: {}", ro.subtask.id, sec)
                ok_sections.append(
                    Section(
                        heading=f"{ro.subtask.query}（生成失败）",
                        body=f"该章节生成失败：{sec}",
                        subtask_id=ro.subtask.id,
                    )
                )
            else:
                ok_sections.append(sec)

        # Build the global citation list, deduplicated by URL.
        all_cites = self._merge_citations([ro.citations for ro in inp.read_outputs])

        body_concat = "\n\n".join(
            f"## {s.heading}\n{s.body}" for s in ok_sections
        )[:8000]
        summary = await Writer()._write_summary(body_concat, ctx)
        title = Writer._derive_title(inp.task.user_query)

        return ResultEnvelope.success(
            ResearchReport(
                task_id=inp.task.id,
                user_query=inp.task.user_query,
                title=title,
                summary=summary,
                sections=ok_sections,
                citations=all_cites,
            )
        )

    async def _write_section(
        self, ro: SubTaskReadOutput, ctx: AgentContext
    ) -> Section:
        if not ro.facts:
            # No facts -> fall back to knowledge-only writing for this section.
            logger.warning(
                "subtask {} has no facts; writing knowledge-only", ro.subtask.id
            )
            messages = [
                {"role": "system", "content": _SECTION_SYS},
                {
                    "role": "user",
                    "content": (
                        f"子问题：{ro.subtask.query}\n"
                        f"期望产出：{'、'.join(ro.subtask.expected_outputs) or '全面回答'}\n\n"
                        f"暂未检索到外部资料，请基于通用知识谨慎作答，并标注「据通用知识」。"
                    ),
                },
            ]
            result = await ctx.pool.chat(messages, temperature=0.4, max_tokens=1200)
            return Section(
                heading=ro.subtask.query,
                body=result.content.strip(),
                subtask_id=ro.subtask.id,
                citations=ro.citations,
            )

        # Build the numbered fact bank. We label citations [c-0], [c-1], ...
        # but sentences inside the bank carry their own (chunk_url) so the
        # writer can cross-reference; we expose it as [n] in the prompt.
        url_to_cite = {c.url: c for c in ro.citations}
        # Map url -> simple index for prompt
        url_to_idx: dict[str, int] = {}
        for i, c in enumerate(ro.citations):
            url_to_idx[c.url] = i + 1

        fact_lines: list[str] = []
        for fact in ro.facts[:60]:  # safety cap on prompt size
            idx = url_to_idx.get(fact.chunk_url, 0)
            tag = f"[{idx}]" if idx else "[?]"
            fact_lines.append(f"- {tag} {fact.text}")
        fact_block = "\n".join(fact_lines)

        cite_lines = [
            f"[{url_to_idx[c.url]}] {c.title} — {c.url}"
            for c in ro.citations
            if c.url in url_to_idx
        ]
        cite_block = "\n".join(cite_lines)

        eo = "、".join(ro.subtask.expected_outputs) if ro.subtask.expected_outputs else "全面回答"
        messages = [
            {"role": "system", "content": _GROUNDED_SECTION_SYS},
            {
                "role": "user",
                "content": (
                    f"子问题：{ro.subtask.query}\n"
                    f"期望产出维度：{eo}\n\n"
                    f"事实片段（已按相关性筛选与压缩）：\n{fact_block}\n\n"
                    f"引用列表：\n{cite_block}\n\n"
                    f"请基于以上事实撰写章节正文，沿用 [n] 引用标注。"
                ),
            },
        ]
        result = await ctx.pool.chat(messages, temperature=0.4, max_tokens=1500)

        # Filter section-level citations to those actually mentioned in body
        body = result.content.strip()
        mentioned: list[Citation] = []
        for c in ro.citations:
            idx = url_to_idx.get(c.url)
            if idx and f"[{idx}]" in body:
                mentioned.append(c)
        if not mentioned:
            mentioned = ro.citations  # fallback: keep all

        return Section(
            heading=ro.subtask.query,
            body=body,
            subtask_id=ro.subtask.id,
            citations=mentioned,
        )

    @staticmethod
    def _merge_citations(per_section: list[list[Citation]]) -> list[Citation]:
        seen: dict[str, Citation] = {}
        for cites in per_section:
            for c in cites:
                key = c.url or c.id
                if key not in seen:
                    seen[key] = c
        return list(seen.values())
