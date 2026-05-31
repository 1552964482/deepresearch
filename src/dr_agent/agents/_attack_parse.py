"""Shared attack-parsing utilities for Red and Multi-Critic agents.

Centralizes:
  * report serialization (with section_id tags)
  * lenient attack coercion (type / span / severity / section_id / action)
  * the 3-layer JSON-parsing fallback loop
    (direct -> strict-retry -> regex)

Both ``agents/red.py`` and ``agents/critics.py`` import from here so the
robustness guarantees are identical across single-Red and Multi-Critic
review modes.
"""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable

from loguru import logger
from pydantic import ValidationError

from dr_agent.agents.base import parse_json_lenient
from dr_agent.schemas.attack import Attack, AttackType
from dr_agent.schemas.patch import PatchAction
from dr_agent.schemas.report import ResearchReport


def serialize_report(report: ResearchReport, max_chars: int = 12000) -> str:
    parts: list[str] = [f"# {report.title}\n\n## Summary\n{report.summary}\n"]
    for sec in report.sections:
        parts.append(
            f"\n[section_id={sec.subtask_id}] ## {sec.heading}\n{sec.body}\n"
        )
    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…(truncated)"
    return text


def coerce_attacks(parsed: object, *, source: str | None = None) -> list[Attack]:
    """Best-effort conversion of a parsed JSON object into Attack objects.

    Extra metadata (``section_id``, ``suggested_action``, ``critic``) is
    stashed on the dataclass __dict__ since Attack's pydantic schema does
    not declare them.
    """
    if not isinstance(parsed, dict):
        return []
    items = parsed.get("attacks")
    if not isinstance(items, list):
        return []
    out: list[Attack] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        try:
            atype = AttackType(str(raw.get("type", "")).lower())
        except ValueError:
            continue
        span = raw.get("span", "")
        if not isinstance(span, str) or not span.strip():
            continue
        try:
            severity = float(raw.get("severity", 0.5))
        except (TypeError, ValueError):
            severity = 0.5
        severity = max(0.0, min(1.0, severity))
        section_id = raw.get("section_id") or ""
        sug_raw = str(raw.get("suggested_action", "")).upper()
        try:
            suggested = PatchAction(sug_raw) if sug_raw else None
        except ValueError:
            suggested = None
        try:
            atk = Attack(
                id=f"atk-{secrets.token_hex(3)}",
                type=atype,
                span=span.strip(),
                evidence=str(raw.get("evidence", ""))[:1000],
                severity=severity,
            )
        except ValidationError:
            continue
        atk.__dict__["section_id"] = section_id
        atk.__dict__["suggested_action"] = suggested
        if source is not None:
            atk.__dict__["critic"] = source
        out.append(atk)
    return out


def is_empty_attacks(parsed: object) -> bool:
    return (
        isinstance(parsed, dict)
        and isinstance(parsed.get("attacks"), list)
        and len(parsed["attacks"]) == 0
    )


ChatFn = Callable[..., Awaitable[object]]


async def parse_with_fallback(
    chat: ChatFn,
    messages: list[dict[str, str]],
    *,
    source: str | None = None,
    max_tokens: int = 2000,
) -> tuple[list[Attack], list[str], str]:
    """Run the 3-layer JSON-parsing fallback against a chat callable.

    Returns (attacks, raw_responses, strategy) where strategy is one of
    "direct" / "retry-strict" / "regex" / "empty".

    ``chat`` is anything with the MimoPool.chat signature; it must accept
    ``json_mode``, ``temperature`` and ``max_tokens`` kwargs and return an
    object exposing ``.content``.
    """
    raw_responses: list[str] = []

    # Layer 1: direct
    r1 = await chat(messages, json_mode=True, temperature=0.3, max_tokens=max_tokens)
    raw_responses.append(r1.content)
    parsed1 = parse_json_lenient(r1.content)
    attacks = coerce_attacks(parsed1, source=source)
    if attacks or is_empty_attacks(parsed1):
        return attacks, raw_responses, ("direct" if attacks else "empty")

    # Layer 2: strict retry
    logger.warning("[{}] layer-1 parse failed; retrying strict", source or "critic")
    retry_messages = messages + [
        {"role": "assistant", "content": r1.content},
        {
            "role": "user",
            "content": (
                "上一次输出无法解析为合法 JSON。请仅输出符合 schema 的 JSON，"
                "不要 markdown 代码块、不要任何解释文字。"
            ),
        },
    ]
    r2 = await chat(
        retry_messages, json_mode=True, temperature=0.0, max_tokens=max_tokens
    )
    raw_responses.append(r2.content)
    parsed2 = parse_json_lenient(r2.content)
    attacks = coerce_attacks(parsed2, source=source)
    if attacks or is_empty_attacks(parsed2):
        return attacks, raw_responses, "retry-strict"

    # Layer 3: regex over any prior response
    logger.warning("[{}] layer-2 parse failed; regex fallback", source or "critic")
    for content in reversed(raw_responses):
        atks = coerce_attacks(parse_json_lenient(content), source=source)
        if atks:
            return atks, raw_responses, "regex"

    logger.warning("[{}] all 3 layers failed; empty", source or "critic")
    return [], raw_responses, "empty"
