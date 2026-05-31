"""Web search wrapper around Tavily.

Tavily returns LLM-ready search results: each item already includes a
cleaned ``content`` snippet, so for many queries we don't even need to
re-fetch the page. Heavy fetching is delegated to :class:`Fetcher` only
when ``include_raw_content`` is requested or content is too short.

Optionally accepts a :class:`SearchCache` so repeated benchmark runs hit
SQLite instead of consuming quota.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger
from tavily import AsyncTavilyClient

if TYPE_CHECKING:
    from dr_agent.tools.search_cache import SearchCache


@dataclass
class SearchResult:
    title: str
    url: str
    content: str
    score: float = 0.0
    raw_content: str | None = None


class WebSearcher:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        cache: "SearchCache | None" = None,
    ) -> None:
        key = api_key or os.getenv("TAVILY_API_KEY")
        if not key:
            raise RuntimeError("TAVILY_API_KEY is required for WebSearcher")
        self._client = AsyncTavilyClient(api_key=key)
        self._cache = cache

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        search_depth: str = "advanced",
        include_raw_content: bool = False,
    ) -> list[SearchResult]:
        """Run one search query.

        If a ``SearchCache`` was wired in, an exact-match cache hit returns
        instantly without consuming Tavily quota. Errors are logged and
        converted into an empty list so a Searcher Agent can keep going
        across multiple SubTasks.
        """
        if self._cache is not None and not include_raw_content:
            cached = self._cache.get(query, max_results, search_depth)
            if cached is not None:
                return cached

        try:
            resp = await self._client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_raw_content=include_raw_content,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("tavily search failed for {!r}: {}", query, e)
            return []

        out: list[SearchResult] = []
        for item in resp.get("results", []):
            out.append(
                SearchResult(
                    title=item.get("title") or "",
                    url=item.get("url") or "",
                    content=item.get("content") or "",
                    score=float(item.get("score") or 0.0),
                    raw_content=item.get("raw_content"),
                )
            )
        logger.debug("tavily {!r} -> {} results", query, len(out))

        if self._cache is not None and not include_raw_content and out:
            self._cache.put(query, max_results, search_depth, out)
        return out

    async def batch_search(
        self,
        queries: list[str],
        *,
        max_results: int = 5,
        concurrency: int = 4,
    ) -> dict[str, list[SearchResult]]:
        sem = asyncio.Semaphore(concurrency)

        async def one(q: str) -> tuple[str, list[SearchResult]]:
            async with sem:
                return q, await self.search(q, max_results=max_results)

        pairs = await asyncio.gather(*(one(q) for q in queries))
        return dict(pairs)
