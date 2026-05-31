"""Tests for the SQLite-backed Tavily search cache."""

from __future__ import annotations

from pathlib import Path

from dr_agent.tools.search import SearchResult
from dr_agent.tools.search_cache import SearchCache


def _sample_results() -> list[SearchResult]:
    return [
        SearchResult(
            title="Doc One",
            url="https://example.com/a",
            content="lorem ipsum dolor sit amet",
            score=0.93,
        ),
        SearchResult(
            title="Doc Two",
            url="https://example.com/b",
            content="quick brown fox jumps",
            score=0.81,
            raw_content=None,
        ),
    ]


def test_round_trip_get_returns_same_payload(tmp_path: Path) -> None:
    cache = SearchCache(tmp_path / "tavily.db")
    cache.put("what is grpo", 5, "advanced", _sample_results())
    out = cache.get("what is grpo", 5, "advanced")
    assert out is not None
    assert len(out) == 2
    assert out[0].url == "https://example.com/a"
    assert out[0].score == 0.93
    cache.close()


def test_miss_on_different_args(tmp_path: Path) -> None:
    cache = SearchCache(tmp_path / "tavily.db")
    cache.put("q", 5, "advanced", _sample_results())
    assert cache.get("q", 10, "advanced") is None
    assert cache.get("q", 5, "basic") is None
    assert cache.get("Q", 5, "advanced") is None  # case-sensitive
    cache.close()


def test_stats_reflects_hits_and_misses(tmp_path: Path) -> None:
    cache = SearchCache(tmp_path / "tavily.db")
    assert cache.get("anything", 5, "advanced") is None  # miss
    cache.put("q", 5, "advanced", _sample_results())
    cache.get("q", 5, "advanced")  # hit
    cache.get("q", 5, "advanced")  # hit
    s = cache.stats()
    assert s["hits"] == 2
    assert s["misses"] == 1
    assert s["entries"] == 1
    cache.close()


def test_persists_across_instances(tmp_path: Path) -> None:
    db = tmp_path / "tavily.db"
    a = SearchCache(db)
    a.put("q", 5, "advanced", _sample_results())
    a.close()
    b = SearchCache(db)
    out = b.get("q", 5, "advanced")
    assert out is not None and len(out) == 2
    b.close()


def test_ttl_expires_old_entries(tmp_path: Path, monkeypatch) -> None:
    cache = SearchCache(tmp_path / "tavily.db", ttl_days=1)
    # Put with a backdated created_at by manipulating the row directly
    cache.put("q", 5, "advanced", _sample_results())
    # Simulate a 30-day-old entry
    cache._conn.execute(
        "UPDATE tavily_cache SET created_at = ?", (1.0,)
    )
    cache._conn.commit()
    assert cache.get("q", 5, "advanced") is None
    cache.close()
