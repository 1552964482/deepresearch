"""Tests for MimoPool: least-in-flight selection, retry/cooldown, key isolation."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from dr_agent.config import MimoConfig
from dr_agent.llm.errors import LLMUnavailable
from dr_agent.llm.pool import MimoPool


def _make_pool(n_keys: int = 4, per_key_concurrency: int = 4) -> MimoPool:
    cfg = MimoConfig(
        api_keys=tuple(f"tp-testapikey-{i:03d}-zzzz" for i in range(n_keys)),
        base_url="https://example.test/v1",
        model="mimo-test",
        rpm_per_key=1000,  # huge, so RPM doesn't gate tests
        safe_concurrency_per_key=per_key_concurrency,
    )
    return MimoPool(cfg, request_timeout_s=5.0, max_retries=2)


def _ok_response(content: str = "hello") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "resp-1",
            "model": "mimo-test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


@pytest.mark.asyncio
async def test_chat_returns_content_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _make_pool()
    used_keys: list[str] = []

    async def fake_post(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        used_keys.append(kwargs["headers"]["Authorization"])
        return _ok_response("ok")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    res = await pool.chat([{"role": "user", "content": "hi"}])
    assert res.content == "ok"
    assert res.key_idx in (0, 1, 2, 3)
    assert len(used_keys) == 1
    await pool.aclose()


@pytest.mark.asyncio
async def test_least_in_flight_distributes_across_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When 8 calls fire in parallel against 4 keys with capacity 4, the
    pool should distribute roughly evenly (specifically: every key used)."""
    pool = _make_pool(n_keys=4, per_key_concurrency=4)
    barrier = asyncio.Event()
    counts = {0: 0, 1: 0, 2: 0, 3: 0}

    async def fake_post(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        token = kwargs["headers"]["Authorization"]
        idx = int(token.rsplit("-", 2)[-2])
        counts[idx] += 1
        # Hold open until barrier is set so concurrency is observable
        await barrier.wait()
        return _ok_response()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    async def go() -> None:
        await pool.chat([{"role": "user", "content": "x"}])

    tasks = [asyncio.create_task(go()) for _ in range(8)]
    # Yield until the inner posts are pending
    for _ in range(20):
        await asyncio.sleep(0.005)
        if sum(counts.values()) == 8:
            break
    barrier.set()
    await asyncio.gather(*tasks)

    # Every key should have been used at least once
    assert all(v > 0 for v in counts.values()), counts
    # Even-ish distribution: max not more than 2x min (8 calls / 4 keys = 2 each)
    assert max(counts.values()) - min(counts.values()) <= 2
    await pool.aclose()


@pytest.mark.asyncio
async def test_429_triggers_cooldown_and_key_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First key returns 429, the pool must switch to another key."""
    pool = _make_pool(n_keys=2, per_key_concurrency=1)
    seen_tokens: list[str] = []

    async def fake_post(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        token = kwargs["headers"]["Authorization"]
        seen_tokens.append(token)
        if "testapikey-000" in token:
            return httpx.Response(
                429, headers={"retry-after": "1"}, text="rate limited"
            )
        return _ok_response("recovered")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    res = await pool.chat([{"role": "user", "content": "x"}])
    assert res.content == "recovered"
    # First call hit key 0, then we switched to key 1.
    assert "testapikey-000" in seen_tokens[0]
    assert "testapikey-001" in seen_tokens[-1]
    # Slot 0 should be in cooldown
    assert pool._slots[0].available is False
    await pool.aclose()


@pytest.mark.asyncio
async def test_5xx_is_transient_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _make_pool(n_keys=2, per_key_concurrency=1)
    calls = {"n": 0}

    async def fake_post(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="upstream down")
        return _ok_response("ok2")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    res = await pool.chat([{"role": "user", "content": "x"}])
    assert res.content == "ok2"
    assert calls["n"] == 2
    await pool.aclose()


@pytest.mark.asyncio
async def test_all_keys_failing_raises_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _make_pool(n_keys=2, per_key_concurrency=1)

    async def fake_post(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return httpx.Response(503, text="dead")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    with pytest.raises(LLMUnavailable):
        await pool.chat([{"role": "user", "content": "x"}])
    await pool.aclose()


@pytest.mark.asyncio
async def test_stats_exposes_per_key_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _make_pool(n_keys=2, per_key_concurrency=2)

    async def fake_post(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return _ok_response()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    for _ in range(3):
        await pool.chat([{"role": "user", "content": "x"}])

    stats = pool.stats()
    assert stats["model"] == "mimo-test"
    assert stats["total_concurrency"] == 4
    total = sum(k["total"] for k in stats["keys"])
    assert total == 3
    # Mask should never include the raw key
    for k in stats["keys"]:
        assert "testapikey" not in k["key"]  # masked
        assert "..." in k["key"]
    await pool.aclose()
