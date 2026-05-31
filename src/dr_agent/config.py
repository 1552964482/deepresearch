"""Global configuration loaded from .env.

All settings are exposed via :func:`get_settings` (cached singleton).
Secret values (API keys) are stored in :class:`Settings` but never logged
verbatim; use :func:`mask_key` when emitting them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


def _project_root() -> Path:
    # src/dr_agent/config.py -> project root is two parents up from src/
    return Path(__file__).resolve().parents[2]


def mask_key(key: str) -> str:
    """Mask an API key for logging: keep first 6 + last 4 chars."""
    if not key:
        return "<empty>"
    if len(key) <= 12:
        return key[:2] + "***"
    return f"{key[:6]}...{key[-4:]}"


@dataclass(frozen=True)
class MimoConfig:
    api_keys: tuple[str, ...]
    base_url: str
    model: str
    rpm_per_key: int
    safe_concurrency_per_key: int

    @property
    def total_concurrency(self) -> int:
        return self.safe_concurrency_per_key * len(self.api_keys)


@dataclass(frozen=True)
class JudgeConfig:
    api_key: str
    base_url: str
    model: str
    concurrency: int = 4
    n_samples: int = 3


@dataclass(frozen=True)
class OrchestratorConfig:
    subtask_timeout_s: float = 90.0
    global_timeout_s: float = 600.0
    batch_failure_threshold: float = 0.30
    max_red_blue_rounds: int = 2


@dataclass(frozen=True)
class Settings:
    project_root: Path
    mimo: MimoConfig
    judge: JudgeConfig
    orch: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    log_level: str = "INFO"


def _collect_mimo_keys() -> tuple[str, ...]:
    """Collect mimo API keys from .env.

    Order of preference:
      1. Numbered slots OPENAI_API_KEY1..N (the user has 4 keys configured)
      2. MIMO_API_KEY as a fallback single key
      3. OPENAI_API_KEY (unnumbered) only if it differs from MIMO_API_KEY
    Duplicate values are deduplicated while preserving first-seen order.
    """
    seen: dict[str, None] = {}
    for i in range(1, 9):  # support up to 8 numbered slots
        v = os.getenv(f"OPENAI_API_KEY{i}")
        if v:
            seen.setdefault(v, None)
    for name in ("MIMO_API_KEY", "OPENAI_API_KEY"):
        v = os.getenv(name)
        if v:
            seen.setdefault(v, None)
    return tuple(seen.keys())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    root = _project_root()
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)

    mimo_keys = _collect_mimo_keys()
    if not mimo_keys:
        raise RuntimeError(
            "No mimo API keys found. Expected at least one of "
            "OPENAI_API_KEY1..N / MIMO_API_KEY / OPENAI_API_KEY in .env"
        )

    mimo = MimoConfig(
        api_keys=mimo_keys,
        base_url=os.getenv(
            "MIMO_BASE_URL",
            os.getenv("OPENAI_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1"),
        ),
        model=os.getenv("MIMO_MODEL", os.getenv("OPENAI_MODEL", "mimo-v2.5-pro")),
        rpm_per_key=int(os.getenv("MIMO_RPM_PER_KEY", "100")),
        safe_concurrency_per_key=int(os.getenv("MIMO_SAFE_CONCURRENCY_PER_KEY", "4")),
    )

    judge_key = os.getenv("JUDGE_API_KEY")
    if not judge_key:
        raise RuntimeError("JUDGE_API_KEY missing in .env")

    judge = JudgeConfig(
        api_key=judge_key,
        base_url=os.getenv("JUDGE_BASE_URL", "https://api.aveve.xyz/v1"),
        model=os.getenv("JUDGE_MODEL", "gpt-5.4"),
        concurrency=int(os.getenv("JUDGE_CONCURRENCY", "4")),
        n_samples=int(os.getenv("JUDGE_N_SAMPLES", "3")),
    )

    return Settings(
        project_root=root,
        mimo=mimo,
        judge=judge,
        orch=OrchestratorConfig(),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
