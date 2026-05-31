"""Blue Agent: produces patches in response to Red's attacks.

For each Red attack, Blue emits a structured Patch with action type
(ADD / DELETE / MODIFY / VERIFY). Patches are then applied via simple
in-section string replacement.

Citation invariant: MODIFY patches must preserve every ``[n]`` citation
marker present in the target span. DELETE patches that would orphan a
citation are rejected. This prevents the regression observed during the
first Phase-3 ablation where Blue silently stripped citations while
addressing completeness attacks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loguru import logger
from pydantic import ValidationError

from dr_agent.agents.base import AbstractAgent, AgentContext, parse_json_lenient
from dr_agent.orchestrator.envelope import ResultEnvelope
from dr_agent.schemas.attack import Attack
from dr_agent.schemas.patch import Patch, PatchAction
from dr_agent.schemas.report import Citation, ResearchReport, Section


_SYSTEM = """你是一名细致的研究报告修订者。你将收到一份带 section 标记的报告草稿和一组针对它的攻击（attacks）。
你的任务是逐条产出修订动作（Patch），并保持报告整体结构与语气不变。

每个 Patch 必须严格符合 JSON schema：
  {
    "attack_id": "...",            // 来自输入
    "action": "ADD" | "DELETE" | "MODIFY" | "VERIFY",
    "target_span": "...",          // 必须是被攻击章节正文中的 EXACT 子串
    "new_text": "...",             // ADD/MODIFY 必填，DELETE 留空，VERIFY 留空
    "verification_evidence": "...",// VERIFY 必填，其他动作可留空
    "rationale": "..."             // 一句话解释你为什么这么改
  }

动作语义：
- ADD：在 target_span 之后追加 new_text
- DELETE：删掉 target_span（new_text 留空）
- MODIFY：把 target_span 整体替换为 new_text
- VERIFY：保留原文不动，但在 verification_evidence 中给出佐证（用于审计）

**引用保留规则（重要）**：
- 如 target_span 中含有 [1] [2] 等引用标记，new_text 必须**原样保留所有引用标记**，位置与原文相对应
- ADD 操作的 new_text 中如有具体事实/数据，必须沿用上下文已有的 [n] 编号；若不能引用现有编号则不要 ADD
- 不允许凭空创造新的 [n] 编号

要求：
- 严格输出 {"patches": [...]} JSON
- target_span 必须能在对应章节正文中定位（一字不差），否则 patch 无效
- 修订要克制：不大改写、不改变核心论点，只修复被攻击的具体问题
- 当攻击是 completeness 类但你**没有可引用的具体事实**时，宁可输出空 patches 列表，也不要凭空补内容"""


_USER_TEMPLATE = """报告草稿（按 section_id 分段）：

{report_text}

针对该草稿的攻击列表：
{attack_list}

请按上述规则输出 patches 列表。"""


@dataclass
class BlueInput:
    report: ResearchReport
    attacks: list[Attack]


@dataclass
class PatchApplyStat:
    accepted: int = 0
    skipped_no_match: int = 0
    skipped_invalid: int = 0


@dataclass
class BlueOutput:
    patches: list[Patch]
    revised_report: ResearchReport
    apply_stats: PatchApplyStat = field(default_factory=PatchApplyStat)


def _format_attack(a: Attack) -> str:
    section_id = a.__dict__.get("section_id", "")
    sug = a.__dict__.get("suggested_action")
    sug_str = sug.value if sug is not None else "?"
    return (
        f'- attack_id={a.id} section_id={section_id} type={a.type.value} '
        f'severity={a.severity:.2f} suggested_action={sug_str}\n'
        f'  span="{a.span[:200]}"\n'
        f'  evidence: {a.evidence[:300]}'
    )


def _serialize(report: ResearchReport, max_chars: int = 12000) -> str:
    parts = [f"# {report.title}\n\n[section_id=__summary__] ## Summary\n{report.summary}\n"]
    for sec in report.sections:
        parts.append(
            f"\n[section_id={sec.subtask_id}] ## {sec.heading}\n{sec.body}\n"
        )
    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…(truncated)"
    return text


_CITATION_RE = re.compile(r"\[\s*\d+\s*\]")


def _citations_in(text: str) -> set[str]:
    return {m.group(0).replace(" ", "") for m in _CITATION_RE.finditer(text or "")}


def _apply_patch(section: Section, patch: Patch) -> tuple[Section, bool]:
    """Apply a single patch to a section by string match.

    Returns (new_section, applied)
    """
    body = section.body
    span = patch.target_span
    if not span or span not in body:
        return section, False

    if patch.action is PatchAction.DELETE:
        # Refuse DELETEs that would drop a citation that does not appear
        # elsewhere in the section. This guards against Blue/Red regressions
        # where a high-quality cited claim is wiped out.
        span_cites = _citations_in(span)
        body_without = body.replace(span, "", 1)
        body_cites_after = _citations_in(body_without)
        dropped = span_cites - body_cites_after
        if dropped:
            return section, False
        new_body = body_without
    elif patch.action is PatchAction.MODIFY:
        # Invariant: new_text must preserve every citation present in span.
        span_cites = _citations_in(span)
        new_cites = _citations_in(patch.new_text or "")
        if not span_cites.issubset(new_cites):
            return section, False
        new_body = body.replace(span, patch.new_text or span, 1)
    elif patch.action is PatchAction.ADD:
        new_body = body.replace(span, span + (patch.new_text or ""), 1)
    elif patch.action is PatchAction.VERIFY:
        # No body change; rationale recorded via patch.accepted=True
        return section, True
    else:
        return section, False

    new_section = section.model_copy(update={"body": new_body})
    return new_section, True


class Blue(AbstractAgent[BlueInput, BlueOutput]):
    name = "blue"

    async def run(
        self, inp: BlueInput, ctx: AgentContext
    ) -> ResultEnvelope[BlueOutput]:
        if not inp.attacks:
            # Nothing to do.
            return ResultEnvelope.success(
                BlueOutput(patches=[], revised_report=inp.report)
            )

        report_text = _serialize(inp.report)
        attack_list = "\n".join(_format_attack(a) for a in inp.attacks)
        messages = [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(
                    report_text=report_text, attack_list=attack_list
                ),
            },
        ]

        # Single attempt with strict JSON. The 3-layer fallback is the Red
        # Agent's responsibility (more important there); Blue is simpler.
        result = await ctx.pool.chat(
            messages, json_mode=True, temperature=0.2, max_tokens=2400
        )
        parsed = parse_json_lenient(result.content)
        patches = self._coerce_patches(parsed)

        # Apply patches.
        section_map: dict[str, Section] = {
            s.subtask_id or "": s for s in inp.report.sections
        }
        attack_to_section: dict[str, str] = {
            a.id: str(a.__dict__.get("section_id", "")) for a in inp.attacks
        }
        stats = PatchApplyStat()
        accepted_patches: list[Patch] = []
        for p in patches:
            sec_id = attack_to_section.get(p.attack_id, "")
            sec = section_map.get(sec_id)
            if sec is None:
                # Unknown section_id; try to find by global match
                logger.debug("blue patch for unknown section_id={}", sec_id)
                stats.skipped_no_match += 1
                continue
            new_sec, applied = _apply_patch(sec, p)
            if applied:
                section_map[sec_id] = new_sec
                p.accepted = True
                accepted_patches.append(p)
                stats.accepted += 1
            else:
                stats.skipped_no_match += 1

        revised_sections = [
            section_map.get(s.subtask_id or "", s) for s in inp.report.sections
        ]
        # Citations may need to merge per-section after patching, keep as-is.
        revised_report = inp.report.model_copy(
            update={"sections": revised_sections}
        )

        logger.info(
            "blue: {} patches accepted, {} skipped(no_match), {} invalid",
            stats.accepted,
            stats.skipped_no_match,
            stats.skipped_invalid,
        )
        return ResultEnvelope.success(
            BlueOutput(
                patches=accepted_patches,
                revised_report=revised_report,
                apply_stats=stats,
            )
        )

    @staticmethod
    def _coerce_patches(parsed: object) -> list[Patch]:
        if not isinstance(parsed, dict):
            return []
        items = parsed.get("patches", [])
        if not isinstance(items, list):
            return []
        out: list[Patch] = []
        for raw in items:
            if not isinstance(raw, dict):
                continue
            action_raw = str(raw.get("action", "")).upper()
            try:
                action = PatchAction(action_raw)
            except ValueError:
                continue
            try:
                p = Patch(
                    attack_id=str(raw.get("attack_id", "")),
                    action=action,
                    target_span=str(raw.get("target_span", "")),
                    new_text=raw.get("new_text") or None,
                    verification_evidence=raw.get("verification_evidence") or None,
                    rationale=str(raw.get("rationale", "")),
                )
            except ValidationError:
                continue
            if not p.attack_id or not p.target_span:
                continue
            out.append(p)
        return out


# Citation is re-exported for convenience by callers that want to add
# evidence URLs to VERIFY patches.
__all__ = ["Blue", "BlueInput", "BlueOutput", "Citation"]
