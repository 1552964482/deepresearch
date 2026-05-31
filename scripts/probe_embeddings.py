"""Probe mimo / OpenAI-compatible base for embeddings support.

Tries a list of common embedding model names against the configured
MIMO_BASE_URL and reports which (if any) work.
"""

from __future__ import annotations

import asyncio
import os

import httpx
from dotenv import load_dotenv

CANDIDATES = [
    # mimo / xiaomi guesses
    "mimo-embedding",
    "mimo-embedding-v1",
    "mimo-embed",
    # OpenAI native
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
    # bge series often hosted on Chinese providers
    "bge-m3",
    "bge-large-zh-v1.5",
    "bge-large-en-v1.5",
    "BAAI/bge-m3",
    # Qwen embeddings
    "text-embedding-v1",
    "text-embedding-v2",
    "Qwen3-Embedding-0.6B",
]


async def try_one(client: httpx.AsyncClient, base: str, key: str, model: str) -> tuple[str, str]:
    try:
        r = await client.post(
            f"{base}/embeddings",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "input": "hello world"},
            timeout=20.0,
        )
        if r.is_success:
            data = r.json()
            dim = len(data["data"][0]["embedding"])
            return ("OK", f"dim={dim}")
        body = r.text[:200].replace("\n", " ")
        return (f"HTTP {r.status_code}", body)
    except Exception as e:  # noqa: BLE001
        return ("ERR", repr(e)[:200])


async def list_models(client: httpx.AsyncClient, base: str, key: str) -> str:
    try:
        r = await client.get(
            f"{base}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=15.0,
        )
        if r.is_success:
            data = r.json()
            ids = [m.get("id") for m in data.get("data", [])]
            return "\n".join(["  - " + str(i) for i in ids]) or "  (empty)"
        return f"  HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:  # noqa: BLE001
        return f"  ERR: {e!r}"


async def main() -> None:
    load_dotenv()
    base = os.getenv("MIMO_BASE_URL") or os.getenv("OPENAI_BASE_URL") or ""
    key = (
        os.getenv("MIMO_API_KEY")
        or os.getenv("OPENAI_API_KEY1")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )
    if not base or not key:
        raise SystemExit("MIMO_BASE_URL / MIMO_API_KEY not set in .env")
    base = base.rstrip("/")

    async with httpx.AsyncClient() as c:
        print(f"== probing {base}")
        print("\n[GET /models]")
        print(await list_models(c, base, key))

        print("\n[POST /embeddings]")
        for m in CANDIDATES:
            status, info = await try_one(c, base, key, m)
            tag = "✅" if status == "OK" else "❌"
            print(f"  {tag} {m:<32}  {status:<10}  {info}")


if __name__ == "__main__":
    asyncio.run(main())
