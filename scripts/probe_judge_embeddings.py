"""Probe the JUDGE_BASE_URL (aveve.xyz) for embeddings support."""

from __future__ import annotations

import asyncio
import os

import httpx
from dotenv import load_dotenv

CANDIDATES = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
    "bge-m3",
    "BAAI/bge-m3",
    "voyage-3",
    "voyage-3-large",
]


async def try_one(c: httpx.AsyncClient, base: str, key: str, model: str) -> tuple[str, str]:
    try:
        r = await c.post(
            f"{base}/embeddings",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "input": "hello world"},
            timeout=20.0,
        )
        if r.is_success:
            d = r.json()
            return ("OK", f"dim={len(d['data'][0]['embedding'])}")
        return (f"HTTP {r.status_code}", r.text[:160].replace("\n", " "))
    except Exception as e:  # noqa: BLE001
        return ("ERR", repr(e)[:160])


async def main() -> None:
    load_dotenv()
    base = (os.getenv("JUDGE_BASE_URL") or "").rstrip("/")
    key = os.getenv("JUDGE_API_KEY") or ""
    if not base or not key:
        raise SystemExit("JUDGE_BASE_URL / JUDGE_API_KEY missing in .env")

    async with httpx.AsyncClient() as c:
        print(f"== probing {base}")

        print("\n[GET /models]")
        try:
            r = await c.get(
                f"{base}/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=15.0,
            )
            if r.is_success:
                ids = [m.get("id") for m in r.json().get("data", [])]
                # Only print embedding-like models to keep output short.
                emb_like = [i for i in ids if i and ("embed" in i.lower() or "voyage" in i.lower() or "bge" in i.lower())]
                print(f"  total models: {len(ids)}")
                print(f"  embedding-like models ({len(emb_like)}):")
                for i in emb_like[:30]:
                    print(f"    - {i}")
            else:
                print(f"  HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:  # noqa: BLE001
            print(f"  ERR: {e!r}")

        print("\n[POST /embeddings]")
        for m in CANDIDATES:
            status, info = await try_one(c, base, key, m)
            tag = "✅" if status == "OK" else "❌"
            print(f"  {tag} {m:<32}  {status:<10}  {info}")


if __name__ == "__main__":
    asyncio.run(main())
