"""LLM-layer exception hierarchy."""

from __future__ import annotations


class LLMError(Exception):
    """Base class for all LLM-gateway errors."""


class LLMRateLimited(LLMError):
    """Raised on HTTP 429 or provider-level rate-limit signals."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMTransient(LLMError):
    """Transient (retryable) error: 5xx, network glitch, timeout."""


class LLMPermanent(LLMError):
    """Permanent (non-retryable) error: 4xx other than 429, malformed request."""


class LLMUnavailable(LLMError):
    """All keys exhausted / all retries failed."""
