"""MimoPool: multi-key load-balanced LLM client with per-key throttling.

Design highlights (résumé-defensible):

* Each API key has its own ``asyncio.Semaphore`` (concurrency cap) and its
  own :class:`SlidingWindowBucket` (RPM cap). A request occupies BOTH
  before being dispatched.
* Key selection is **least-in-flight**: pick the key with the smallest
  current ``in_flight`` count, breaking ties by smallest recent RPM usage.
  This avoids hot-spotting a single key during burst traffic.
* On 429 a key enters exponential backoff and is skipped during selection
  until its cool-down expires.
* Failure cascade: same-key retry (1) -> alternate-key retry (1) -> raise
  :class:`LLMUnavailable`.
* Per-key metrics (in_flight, total, success, total_latency_s, last_error)
  are exposed via :meth:`stats` for the ``dr-agent pool-stats`` CLI.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from loguru import logger

from dr_agent.config import MimoConfig, mask_key
from dr_agent.llm.errors import (
    LLMPermanent,
    LLMRateLimited,
    LLMTransient,
    LLMUnavailable,
)
from dr_agent.llm.token_bucket import SlidingWindowBucket
from dr_agent.utils.trace import get_trace_id


@dataclass
class ChatResult:
    """Normalized chat completion result."""

    content: str
    model: str
    usage: dict[str, int]  # prompt_tokens / completion_tokens / total_tokens
    raw: dict[str, Any]
    key_idx: int
    latency_s: float


@dataclass
class _KeySlot:
    idx: int
    api_key: str
    sem: asyncio.Semaphore
    bucket: SlidingWindowBucket
    in_flight: int = 0
    total: int = 0
    success: int = 0
    total_latency_s: float = 0.0
    last_error: str | None = None
    cooldown_until: float = 0.0  # monotonic timestamp; 0 means available

    @property
    def available(self) -> bool:
        return time.monotonic() >= self.cooldown_until

    @property
    def avg_latency_s(self) -> float:
        return self.total_latency_s / self.success if self.success else 0.0


class MimoPool:
    """Load-balanced async client for the mimo (xiaomi) OpenAI-compatible API."""

    def __init__(
        self,
        config: MimoConfig,
        *,
        request_timeout_s: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self.config = config
        self._request_timeout_s = request_timeout_s
        self._max_retries = max_retries
        self._slots: list[_KeySlot] = [
            _KeySlot(
                idx=i,
                api_key=k,
                sem=asyncio.Semaphore(config.safe_concurrency_per_key),
                bucket=SlidingWindowBucket(rpm=config.rpm_per_key),
            )
            for i, k in enumerate(config.api_keys)
        ]
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(request_timeout_s),
            limits=httpx.Limits(
                max_connections=config.total_concurrency * 2,
                max_keepalive_connections=config.total_concurrency,
            ),
        )
        logger.info(
            "MimoPool ready: {} keys, model={}, total_concurrency={}",
            len(self._slots),
            config.model,
            config.total_concurrency,
        )
        for s in self._slots:
            logger.debug("  key[{}]={}", s.idx, mask_key(s.api_key))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "MimoPool":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # ---------- selection ----------

    def _pick_slot(self, exclude: set[int] | None = None) -> _KeySlot | None:
        exclude = exclude or set()
        candidates = [s for s in self._slots if s.idx not in exclude and s.available]
        if not candidates:
            return None
        # least-in-flight, tie-break on smaller recent RPM usage
        candidates.sort(key=lambda s: (s.in_flight, s.bucket.usage()))
        return candidates[0]

    def _earliest_cooldown_release(self, exclude: set[int] | None = None) -> float:
        exclude = exclude or set()
        future = [
            s.cooldown_until - time.monotonic()
            for s in self._slots
            if s.idx not in exclude and not s.available
        ]
        return min(future) if future else 0.0

    # ---------- public ----------

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> ChatResult:
        """Send a chat completion. Retries across keys on transient failures."""
        used: set[int] = set()
        last_exc: Exception | None = None
        attempts = 0
        max_total_attempts = self._max_retries + 1

        while attempts < max_total_attempts:
            slot = self._pick_slot(exclude=used if attempts > 0 else None)
            if slot is None:
                # All keys excluded or in cooldown; wait for nearest one to recover
                wait = self._earliest_cooldown_release(exclude=used)
                if wait <= 0:
                    break
                logger.warning("all keys throttled, sleeping {:.2f}s", wait)
                await asyncio.sleep(min(wait + 0.1, 30.0))
                continue
            try:
                return await self._dispatch(
                    slot,
                    messages,
                    json_mode=json_mode,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=model or self.config.model,
                )
            except LLMRateLimited as e:
                last_exc = e
                slot.last_error = f"429: {e}"
                cool = e.retry_after or 5.0 * (2 ** attempts)
                slot.cooldown_until = time.monotonic() + cool
                logger.warning(
                    "key[{}] 429, cooldown={:.1f}s, switching", slot.idx, cool
                )
                used.add(slot.idx)
                attempts += 1
            except LLMTransient as e:
                last_exc = e
                slot.last_error = f"transient: {e}"
                logger.warning("key[{}] transient: {}", slot.idx, e)
                attempts += 1
                if attempts < max_total_attempts:
                    used.add(slot.idx)  # try a different key next
            except LLMPermanent:
                # No point retrying on permanent error.
                raise

        raise LLMUnavailable(
            f"all keys exhausted after {attempts} attempts; last_error={last_exc!r}"
        )

    async def _dispatch(
        self,
        slot: _KeySlot,
        messages: list[dict[str, str]],
        *,
        json_mode: bool,
        temperature: float,
        max_tokens: int | None,
        model: str,
    ) -> ChatResult:
        await slot.bucket.acquire()
        async with slot.sem:
            slot.in_flight += 1
            slot.total += 1
            t0 = time.monotonic()
            try:
                payload: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }
                if max_tokens is not None:
                    payload["max_tokens"] = max_tokens
                if json_mode:
                    payload["response_format"] = {"type": "json_object"}

                trace = get_trace_id()
                logger.debug(
                    "key[{}] -> {} msgs, json={}, temp={:.2f}, trace={}",
                    slot.idx,
                    len(messages),
                    json_mode,
                    temperature,
                    trace,
                )

                resp = await self._client.post(
                    "/chat/completions",
                    headers={
                        "Authorization": f"Bearer {slot.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                self._raise_for_status(resp)
                data = resp.json()
                content = data["choices"][0]["message"]["content"] or ""
                latency = time.monotonic() - t0
                slot.success += 1
                slot.total_latency_s += latency
                slot.last_error = None
                return ChatResult(
                    content=content,
                    model=data.get("model", model),
                    usage=data.get("usage", {}),
                    raw=data,
                    key_idx=slot.idx,
                    latency_s=latency,
                )
            except httpx.TimeoutException as e:
                raise LLMTransient(f"timeout: {e}") from e
            except httpx.HTTPError as e:
                raise LLMTransient(f"network: {e}") from e
            finally:
                slot.in_flight -= 1

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        if resp.is_success:
            return
        text = resp.text[:500]
        if resp.status_code == 429:
            ra = resp.headers.get("retry-after")
            try:
                retry_after = float(ra) if ra else None
            except ValueError:
                retry_after = None
            raise LLMRateLimited(f"429 from upstream: {text}", retry_after=retry_after)
        if resp.status_code >= 500:
            raise LLMTransient(f"upstream {resp.status_code}: {text}")
        raise LLMPermanent(f"upstream {resp.status_code}: {text}")

    # ---------- introspection ----------

    def stats(self) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "total_concurrency": self.config.total_concurrency,
            "keys": [
                {
                    "idx": s.idx,
                    "key": mask_key(s.api_key),
                    "in_flight": s.in_flight,
                    "total": s.total,
                    "success": s.success,
                    "avg_latency_s": round(s.avg_latency_s, 3),
                    "rpm_used": s.bucket.usage(),
                    "rpm_limit": self.config.rpm_per_key,
                    "available": s.available,
                    "last_error": s.last_error,
                }
                for s in self._slots
            ],
        }
