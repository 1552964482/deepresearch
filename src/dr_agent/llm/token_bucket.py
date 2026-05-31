"""Sliding-window token bucket for per-key RPM throttling.

Each ``acquire()`` call records a timestamp; if the number of recorded
timestamps within the trailing 60s window has reached ``rpm``, the call
waits until the oldest timestamp expires.

This is intentionally simple: timestamps are pruned lazily, the data
structure is a deque, and concurrent access is guarded by an asyncio.Lock.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque


class SlidingWindowBucket:
    """A 60-second sliding-window rate limiter.

    Args:
        rpm: max requests per 60-second window.
        window_s: window size in seconds (default 60).
    """

    def __init__(self, rpm: int, window_s: float = 60.0) -> None:
        if rpm <= 0:
            raise ValueError("rpm must be positive")
        self.rpm = rpm
        self.window_s = window_s
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    async def acquire(self) -> None:
        """Block until the bucket has capacity, then consume one slot."""
        while True:
            async with self._lock:
                now = time.monotonic()
                self._prune(now)
                if len(self._timestamps) < self.rpm:
                    self._timestamps.append(now)
                    return
                wait_s = self.window_s - (now - self._timestamps[0]) + 0.01
            # Sleep outside the lock so other callers can prune.
            await asyncio.sleep(max(wait_s, 0.05))

    def usage(self) -> int:
        """Current count within the trailing window (best-effort, non-locked)."""
        self._prune(time.monotonic())
        return len(self._timestamps)
