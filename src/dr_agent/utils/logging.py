"""Loguru configuration with API-key masking."""

from __future__ import annotations

import re
import sys

from loguru import logger

from dr_agent.config import get_settings, mask_key

_KEY_PATTERN = re.compile(r"(?P<prefix>tp-|sk-)[A-Za-z0-9_-]{16,}")


def _mask_filter(record: dict) -> bool:
    """Mutate the record's message to mask API-key-like substrings."""
    msg = record.get("message", "")
    if "tp-" in msg or "sk-" in msg:
        record["message"] = _KEY_PATTERN.sub(
            lambda m: mask_key(m.group(0)), msg
        )
    return True


_configured = False


def setup_logging(level: str | None = None) -> None:
    """Configure loguru once. Idempotent."""
    global _configured
    if _configured:
        return
    settings = get_settings()
    logger.remove()
    logger.add(
        sys.stderr,
        level=level or settings.log_level,
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level: <7}</level> | "
            "<cyan>{extra[trace]}</cyan> | "
            "<level>{message}</level>"
        ),
        filter=_mask_filter,
        enqueue=True,
    )
    logger.configure(extra={"trace": "-"})
    _configured = True


__all__ = ["setup_logging", "logger"]
