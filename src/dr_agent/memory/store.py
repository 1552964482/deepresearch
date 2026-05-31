"""SQLite + numpy vector memory store.

Design choice: brute-force cosine over a numpy matrix kept in process memory,
synced from SQLite on startup and on every write. For < 10k records this
is faster (and far simpler to reason about) than running a vector database.

Schema:
    memory(
        id TEXT PRIMARY KEY,
        task_id TEXT,
        agent_id TEXT,
        text TEXT,
        embedding BLOB,        -- raw float32 bytes, length = dim * 4
        created_at REAL,
        is_duplicate INTEGER DEFAULT 0,
        contradicts TEXT       -- JSON list of contradicting ids
    )
"""

from __future__ import annotations

import asyncio
import json
import secrets
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

from dr_agent.memory.embedder import Embedder


@dataclass
class MemoryItem:
    id: str
    task_id: str
    agent_id: str
    text: str
    embedding: np.ndarray | None = None
    created_at: float = 0.0
    is_duplicate: bool = False
    contradicts_with: list[str] = field(default_factory=list)


@dataclass
class WriteResult:
    item: MemoryItem
    written: bool
    duplicate_of: str | None = None
    contradicts: list[str] = field(default_factory=list)


class MemoryStore:
    """Cross-Agent shared memory with vector search, dedupe, and contradiction
    detection."""

    DUP_THRESHOLD = 0.92
    CONTRADICTION_THRESHOLD = 0.30  # cosine similarity below this AND same lead-noun -> conflict
    CONTRADICTION_LEAD_CHARS = 12

    def __init__(self, db_path: str | Path, embedder: Embedder) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        self._lock = asyncio.Lock()  # serializes writes; reads can be concurrent
        # In-memory cache for fast cosine search
        self._ids: list[str] = []
        self._matrix: np.ndarray = np.zeros((0, embedder.dim), dtype=np.float32)
        self._items: dict[str, MemoryItem] = {}
        self._reload_from_disk()
        logger.info(
            "MemoryStore ready at {} ({} items in cache)",
            self.db_path,
            len(self._ids),
        )

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at REAL NOT NULL,
                is_duplicate INTEGER DEFAULT 0,
                contradicts TEXT DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_memory_task ON memory(task_id);
            CREATE INDEX IF NOT EXISTS idx_memory_agent ON memory(agent_id);
            """
        )
        self._conn.commit()

    def _reload_from_disk(self) -> None:
        cur = self._conn.execute(
            "SELECT id, task_id, agent_id, text, embedding, created_at, is_duplicate, contradicts FROM memory"
        )
        ids: list[str] = []
        vecs: list[np.ndarray] = []
        items: dict[str, MemoryItem] = {}
        for row in cur:
            (rid, tid, aid, text, emb_blob, created, is_dup, contra_json) = row
            v = np.frombuffer(emb_blob, dtype=np.float32)
            ids.append(rid)
            vecs.append(v)
            items[rid] = MemoryItem(
                id=rid,
                task_id=tid,
                agent_id=aid,
                text=text,
                embedding=v,
                created_at=created,
                is_duplicate=bool(is_dup),
                contradicts_with=json.loads(contra_json) if contra_json else [],
            )
        self._ids = ids
        self._items = items
        if vecs:
            self._matrix = np.stack(vecs, axis=0)
        else:
            self._matrix = np.zeros((0, self.embedder.dim), dtype=np.float32)

    # ---------- public API ----------

    async def write(
        self,
        text: str,
        *,
        task_id: str,
        agent_id: str,
        skip_dedupe: bool = False,
        skip_contradict: bool = False,
    ) -> WriteResult:
        if not text.strip():
            raise ValueError("cannot write empty text")
        # Truncate excessive content to protect the DB.
        if len(text) > 8192:
            text = text[:8192]
        emb = self.embedder.encode_one(text)

        async with self._lock:
            duplicate_of: str | None = None
            contradicts: list[str] = []
            if not skip_dedupe and self._matrix.shape[0] > 0:
                sims = self._matrix @ emb
                top_idx = int(np.argmax(sims))
                top_sim = float(sims[top_idx])
                if top_sim >= self.DUP_THRESHOLD:
                    duplicate_of = self._ids[top_idx]
                    logger.debug(
                        "dedupe hit (sim={:.3f}): new vs {}",
                        top_sim,
                        duplicate_of,
                    )

            if not skip_contradict and duplicate_of is None:
                contradicts = self._detect_contradictions(text, emb)

            item = MemoryItem(
                id=f"mem-{secrets.token_hex(4)}",
                task_id=task_id,
                agent_id=agent_id,
                text=text,
                embedding=emb,
                created_at=time.time(),
                is_duplicate=duplicate_of is not None,
                contradicts_with=contradicts,
            )

            if duplicate_of is not None:
                # Don't actually persist duplicates; just report them.
                return WriteResult(item=item, written=False, duplicate_of=duplicate_of)

            self._persist(item)
            self._ids.append(item.id)
            self._matrix = (
                np.vstack([self._matrix, emb[None, :]])
                if self._matrix.shape[0]
                else emb[None, :].astype(np.float32)
            )
            self._items[item.id] = item
            return WriteResult(item=item, written=True, contradicts=contradicts)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        agent_filter: str | None = None,
        task_filter: str | None = None,
        min_sim: float = 0.0,
    ) -> list[tuple[MemoryItem, float]]:
        if self._matrix.shape[0] == 0:
            return []
        q = self.embedder.encode_one(query)
        sims = self._matrix @ q
        order = np.argsort(-sims)
        out: list[tuple[MemoryItem, float]] = []
        for idx in order:
            score = float(sims[idx])
            if score < min_sim:
                break
            item = self._items[self._ids[idx]]
            if agent_filter and item.agent_id != agent_filter:
                continue
            if task_filter and item.task_id != task_filter:
                continue
            out.append((item, score))
            if len(out) >= top_k:
                break
        return out

    def all_for_task(self, task_id: str) -> list[MemoryItem]:
        return [it for it in self._items.values() if it.task_id == task_id]

    # ---------- internals ----------

    def _persist(self, item: MemoryItem) -> None:
        assert item.embedding is not None
        self._conn.execute(
            """
            INSERT INTO memory
              (id, task_id, agent_id, text, embedding, created_at, is_duplicate, contradicts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.id,
                item.task_id,
                item.agent_id,
                item.text,
                item.embedding.tobytes(),
                item.created_at,
                int(item.is_duplicate),
                json.dumps(item.contradicts_with),
            ),
        )
        self._conn.commit()

    def _detect_contradictions(self, text: str, emb: np.ndarray) -> list[str]:
        """Lightweight contradiction heuristic: same leading subject, low cosine.

        We treat the first ``CONTRADICTION_LEAD_CHARS`` characters as a crude
        subject anchor. If two records share that prefix yet have low semantic
        similarity, flag them as potentially contradictory. This is intentionally
        cheap; a future iteration can plug in NLI.
        """
        if self._matrix.shape[0] == 0:
            return []
        lead = text.strip()[: self.CONTRADICTION_LEAD_CHARS]
        if not lead:
            return []
        flagged: list[str] = []
        sims = self._matrix @ emb
        for idx, item_id in enumerate(self._ids):
            other = self._items[item_id]
            if other.text.strip()[: self.CONTRADICTION_LEAD_CHARS] != lead:
                continue
            if float(sims[idx]) < self.CONTRADICTION_THRESHOLD:
                flagged.append(item_id)
        return flagged

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass

    def stats(self) -> dict[str, int]:
        return {"items": len(self._ids), "dim": self.embedder.dim}
