"""Tests for SlidingWindowBucket (per-key RPM limiter)."""

from __future__ import annotations

import asyncio
import time

import pytest

from dr_agent.llm.token_bucket import SlidingWindowBucket


@pytest.mark.asyncio
async def test_bucket_admits_below_capacity() -> None:
    bucket = SlidingWindowBucket(rpm=5)
    t0 = time.monotonic()
    for _ in range(5):
        await bucket.acquire()
    # Should be near-instant.
    assert time.monotonic() - t0 < 0.5


@pytest.mark.asyncio
async def test_bucket_blocks_when_full() -> None:
    bucket = SlidingWindowBucket(rpm=2, window_s=0.5)
    await bucket.acquire()
    await bucket.acquire()
    t0 = time.monotonic()
    await bucket.acquire()  # third should wait until oldest expires (~0.5s)
    waited = time.monotonic() - t0
    assert 0.4 < waited < 1.5


@pytest.mark.asyncio
async def test_usage_counts_within_window() -> None:
    bucket = SlidingWindowBucket(rpm=10, window_s=10)
    for _ in range(3):
        await bucket.acquire()
    assert bucket.usage() == 3


@pytest.mark.asyncio
async def test_invalid_rpm_raises() -> None:
    with pytest.raises(ValueError):
        SlidingWindowBucket(rpm=0)
    with pytest.raises(ValueError):
        SlidingWindowBucket(rpm=-3)
