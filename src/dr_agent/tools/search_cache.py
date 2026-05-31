"""Persistent SQLite cache for Tavily search results.

Cache key: SHA-1 of (query, max_results, search_depth). Stored payload is
the JSON-serialized list of :class:`SearchResult`. Entries older than
``ttl_days`` are ignored on read.

Why bother caching:
  * Tavily's free tier is 1000 searches/month — running the 35-question
    ResearchBench across multiple ablation configs blows through it fast.
  * Identical (query, max_results) pairs across re-runs of the same
    benchmark hit the cache and consume zero quota.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path

from loguru import logger

from dr_agent.tools.search import SearchResult


def _key(query: str, max_results: int, search_depth: str) -> str:
    h = hashlib.sha1()
    h.update(query.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(max_results).encode("ascii"))
    h.update(b"\x00")
    h.update(search_depth.encode("ascii"))
    return h.hexdigest()


class SearchCache:
    def __init__(self, db_path: str | Path, *, ttl_days: int = 14) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ttl_s = ttl_days * 86400
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tavily_cache (
                k TEXT PRIMARY KEY,
                query TEXT,
                payload TEXT,
                created_at REAL
            )
            """
        )
        self._conn.commit()
        self._hits = 0
        self._misses = 0

    def get(
        self, query: str, max_results: int, search_depth: str
    ) -> list[SearchResult] | None:
        k = _key(query, max_results, search_depth)
        cur = self._conn.execute(
            "SELECT payload, created_at FROM tavily_cache WHERE k=?", (k,)
        )
        row = cur.fetchone()
        if row is None:
            self._misses += 1
            return None
        payload, created = row
        if (time.time() - created) > self._ttl_s:
            self._misses += 1
            return None
        try:
            items = json.loads(payload)
        except json.JSONDecodeError:
            self._misses += 1
            return None
        self._hits += 1
        logger.debug("tavily-cache hit: {!r}", query[:60])
        return [SearchResult(**item) for item in items]

    def put(
        self,
        query: str,
        max_results: int,
        search_depth: str,
        results: list[SearchResult],
    ) -> None:
        k = _key(query, max_results, search_depth)
        payload = json.dumps([asdict(r) for r in results], ensure_ascii=False)
        self._conn.execute(
            """
            INSERT INTO tavily_cache (k, query, payload, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(k) DO UPDATE SET payload=excluded.payload,
              created_at=excluded.created_at
            """,
            (k, query, payload, time.time()),
        )
        self._conn.commit()

    def stats(self) -> dict[str, int]:
        cur = self._conn.execute("SELECT COUNT(*) FROM tavily_cache")
        n = int(cur.fetchone()[0])
        return {"entries": n, "hits": self._hits, "misses": self._misses}

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass
