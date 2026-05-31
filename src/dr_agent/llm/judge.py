"""JudgeClient: independent gpt-5.4 channel for LLM-as-Judge scoring.

Crucially, this client is **not** part of MimoPool. The model family used
for judging must differ from the model family being evaluated to avoid
self-preference bias (Zheng et al., 2023; Panickssery et al., 2024).

If the independent endpoint is unavailable (e.g. provider outage),
:class:`JudgeClient` can be configured to fall back to :class:`MimoPool`,
but every score produced via fallback is tagged ``backend='mimo-fallback'``
and ``self_bias_risk=True`` so downstream reports can flag it.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger
from pydantic import BaseModel, Field

from dr_agent.config import JudgeConfig
from dr_agent.llm.errors import LLMPermanent, LLMTransient

if TYPE_CHECKING:
    from dr_agent.llm.pool import MimoPool


class JudgeRubric(BaseModel):
    """Five-dimension rubric used for LLM-as-Judge scoring."""

    accuracy: str = "Factual correctness; key facts must match real-world knowledge."
    completeness: str = "Coverage of the question; no major aspects missing."
    logic: str = "Internal consistency; conclusions follow from evidence."
    citation_quality: str = "Citations exist, are relevant, and support claims."
    readability: str = "Clear structure, smooth prose, appropriate technical depth."


class JudgeScore(BaseModel):
    accuracy: float = Field(ge=1.0, le=5.0)
    completeness: float = Field(ge=1.0, le=5.0)
    logic: float = Field(ge=1.0, le=5.0)
    citation_quality: float = Field(ge=1.0, le=5.0)
    readability: float = Field(ge=1.0, le=5.0)
    overall: float = Field(ge=1.0, le=5.0)
    samples_used: int = 0
    sample_std: float = 0.0
    raw_samples: list[dict[str, Any]] = Field(default_factory=list)
    backend: str = "judge"
    self_bias_risk: bool = False


SYSTEM_PROMPT = """You are an expert reviewer for technical research reports.
Score the report on five dimensions, each from 1.0 to 5.0 (one decimal place).

Dimensions:
- accuracy: {accuracy}
- completeness: {completeness}
- logic: {logic}
- citation_quality: {citation_quality}
- readability: {readability}

Return STRICT JSON only, no prose:
{{"accuracy": float, "completeness": float, "logic": float,
  "citation_quality": float, "readability": float, "rationale": "<short>"}}
"""

USER_PROMPT = """Question:
{question}

Report:
{report}
"""


@dataclass
class _JudgeMetrics:
    total: int = 0
    success: int = 0
    parse_failures: int = 0
    total_latency_s: float = 0.0


class JudgeClient:
    """Async client for the independent judge model (gpt-5.4 via aveve.xyz)."""

    def __init__(
        self,
        config: JudgeConfig,
        *,
        request_timeout_s: float = 90.0,
        max_retries: int = 3,
        fallback_pool: "MimoPool | None" = None,
        fallback_model: str | None = None,
    ) -> None:
        self.config = config
        self._sem = asyncio.Semaphore(config.concurrency)
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(request_timeout_s),
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
        )
        self._metrics = _JudgeMetrics()
        self._fallback_pool = fallback_pool
        self._fallback_model = fallback_model
        self._fallback_locked = False  # set after first fallback success
        logger.info(
            "JudgeClient ready: model={}, concurrency={}, n_samples={}{}",
            config.model,
            config.concurrency,
            config.n_samples,
            f", fallback=mimo[{fallback_model}]" if fallback_pool else "",
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "JudgeClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def score(
        self,
        *,
        question: str,
        report: str,
        rubric: JudgeRubric | None = None,
        n_samples: int | None = None,
    ) -> JudgeScore:
        rubric = rubric or JudgeRubric()
        n = n_samples if n_samples is not None else self.config.n_samples

        # Run n samples in parallel; retain only those that parsed successfully.
        results = await asyncio.gather(
            *[self._score_once(question, report, rubric) for _ in range(n)],
            return_exceptions=True,
        )
        parsed: list[dict[str, Any]] = []
        used_fallback = False
        for r in results:
            if isinstance(r, Exception):
                logger.warning("judge sample failed: {}", r)
                continue
            payload, fb = r
            parsed.append(payload)
            used_fallback = used_fallback or fb
        if not parsed and self._fallback_pool is not None:
            # Last-ditch fallback: try once via mimo
            logger.warning("primary judge fully failed; trying mimo fallback")
            try:
                payload, _ = await self._score_once_fallback(question, report, rubric)
                parsed = [payload]
                used_fallback = True
                self._fallback_locked = True
            except Exception as e:  # noqa: BLE001
                logger.error("fallback judge also failed: {}", e)
        if not parsed:
            raise LLMTransient("all judge samples failed (primary + fallback)")

        def m(k: str) -> float:
            return mean(float(p[k]) for p in parsed)

        dims = ("accuracy", "completeness", "logic", "citation_quality", "readability")
        avg = {k: m(k) for k in dims}
        overall = mean(avg.values())
        std = pstdev([mean(float(p[k]) for k in dims) for p in parsed]) if len(parsed) > 1 else 0.0

        return JudgeScore(
            **avg,
            overall=overall,
            samples_used=len(parsed),
            sample_std=std,
            raw_samples=parsed,
            backend=("mimo-fallback" if used_fallback else self.config.model),
            self_bias_risk=used_fallback,
        )

    async def _score_once(
        self, question: str, report: str, rubric: JudgeRubric
    ) -> tuple[dict[str, Any], bool]:
        sys_prompt = SYSTEM_PROMPT.format(**rubric.model_dump())
        user_prompt = USER_PROMPT.format(question=question, report=report)

        if self._fallback_locked and self._fallback_pool is not None:
            return await self._score_once_fallback(question, report, rubric)

        async with self._sem:
            self._metrics.total += 1
            t0 = time.monotonic()
            try:
                content = await self._chat_with_retry(sys_prompt, user_prompt)
            except (LLMTransient, LLMPermanent) as e:
                if self._fallback_pool is None:
                    raise
                logger.warning("primary judge failed ({}); falling back to mimo", e)
                self._fallback_locked = True
                # Fall through to fallback below.
                content = None
            self._metrics.total_latency_s += time.monotonic() - t0

        if content is None:
            return await self._score_once_fallback(question, report, rubric)

        parsed = self._parse_json(content)
        if parsed is None:
            self._metrics.parse_failures += 1
            raise LLMPermanent("judge returned non-JSON content")
        self._metrics.success += 1
        return parsed, False

    async def _score_once_fallback(
        self, question: str, report: str, rubric: JudgeRubric
    ) -> tuple[dict[str, Any], bool]:
        assert self._fallback_pool is not None
        sys_prompt = SYSTEM_PROMPT.format(**rubric.model_dump()) + (
            "\n\n注意：这是 self-judge 降级路径，请尽量克制，避免对自身风格输出过分宽容。"
        )
        user_prompt = USER_PROMPT.format(question=question, report=report)
        result = await self._fallback_pool.chat(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            json_mode=True,
            temperature=0.0,
            max_tokens=600,
            model=self._fallback_model,
        )
        parsed = self._parse_json(result.content)
        if parsed is None:
            raise LLMPermanent("fallback judge returned non-JSON content")
        return parsed, True

    async def _chat_with_retry(self, system: str, user: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = await self._client.post(
                    "/chat/completions",
                    json={
                        "model": self.config.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "temperature": 0.0,
                        "response_format": {"type": "json_object"},
                    },
                )
                if resp.status_code == 429:
                    raise LLMTransient(f"judge 429: {resp.text[:200]}")
                if resp.status_code >= 500:
                    raise LLMTransient(f"judge {resp.status_code}: {resp.text[:200]}")
                if not resp.is_success:
                    raise LLMPermanent(
                        f"judge {resp.status_code}: {resp.text[:200]}"
                    )
                data = resp.json()
                return data["choices"][0]["message"]["content"] or ""
            except (httpx.TimeoutException, httpx.HTTPError, LLMTransient) as e:
                last_exc = e
                wait = 2.0 * (2 ** attempt)
                logger.warning("judge attempt {} failed ({}); retry in {}s", attempt + 1, e, wait)
                await asyncio.sleep(wait)
        raise LLMTransient(f"judge exhausted retries: {last_exc!r}")

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        """Robust JSON parse: direct -> regex-extract first {...} block."""
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None

    def stats(self) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "total": self._metrics.total,
            "success": self._metrics.success,
            "parse_failures": self._metrics.parse_failures,
            "avg_latency_s": (
                round(self._metrics.total_latency_s / self._metrics.success, 3)
                if self._metrics.success
                else 0.0
            ),
        }
