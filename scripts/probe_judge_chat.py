"""Probe each judge-side model with a short chat request."""

from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv

CANDIDATES = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.5"]


def main() -> None:
    load_dotenv()
    base = (os.getenv("JUDGE_BASE_URL") or "").rstrip("/")
    key = os.getenv("JUDGE_API_KEY") or ""
    if not base or not key:
        raise SystemExit("JUDGE_BASE_URL / JUDGE_API_KEY missing")

    for model in CANDIDATES:
        try:
            r = httpx.post(
                base + "/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "say only ok"}],
                    "temperature": 0.0,
                    "max_tokens": 5,
                },
                timeout=30.0,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  {model:<16} EXC: {e!r}")
            continue
        if r.is_success:
            try:
                content = r.json()["choices"][0]["message"]["content"]
            except Exception:  # noqa: BLE001
                content = r.text[:120]
            print(f"  ✅ {model:<16} -> {content!r}")
        else:
            body = r.text[:120].replace("\n", " ")
            print(f"  ❌ {model:<16} HTTP {r.status_code}: {body}")


if __name__ == "__main__":
    main()
