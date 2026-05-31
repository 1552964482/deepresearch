"""Three-stage semantic compression: L1 embedding filter -> L2 TextRank -> L3 protected spans.

L1 (Embedding cosine filter)
    Drop chunks whose cosine similarity to the SubTask query is below a
    configurable threshold.

L2 (TextRank sentence selection)
    Within each surviving chunk, rank sentences by graph centrality (jieba
    tokenized for Chinese, whitespace fallback for English) and keep top-k.

L3 (Protected spans)
    Identify sentences that contain numbers / years / quoted text /
    capitalized acronyms and ALWAYS keep them, even if they didn't make
    the L2 cut.

The output ``CompressResult`` exposes per-stage statistics so the
ablation experiment scripts can plot "no compress / L1 / L1+L2 / L1+L2+L3".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import jieba
import networkx as nx
import numpy as np
from loguru import logger

# Eagerly initialize jieba's prefix dict so that worker threads don't each
# trigger a separate "Building prefix dict from the default dictionary" load.
jieba.initialize()

from dr_agent.memory.embedder import Embedder
from dr_agent.tools.fetcher import Chunk


@dataclass
class CompressStats:
    raw_chunks: int = 0
    raw_chars: int = 0
    after_l1_chunks: int = 0
    after_l1_chars: int = 0
    after_l2_sentences: int = 0
    after_l2_chars: int = 0
    after_l3_sentences: int = 0
    after_l3_chars: int = 0
    protected_added: int = 0


@dataclass
class CompressedSentence:
    chunk_url: str
    chunk_title: str
    text: str
    score: float
    is_protected: bool


@dataclass
class CompressResult:
    sentences: list[CompressedSentence] = field(default_factory=list)
    stats: CompressStats = field(default_factory=CompressStats)

    def joined(self, separator: str = "\n") -> str:
        return separator.join(s.text for s in self.sentences)

    def by_chunk(self) -> dict[str, list[CompressedSentence]]:
        out: dict[str, list[CompressedSentence]] = {}
        for s in self.sentences:
            out.setdefault(s.chunk_url, []).append(s)
        return out


# Regex helpers
_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])|\n+|(?<=[.!?])\s+")
_NUM_RE = re.compile(r"\d+(?:[\.,]\d+)?%?|\d{4}\s*年|[A-Z]{2,}")
_QUOTE_RE = re.compile(r"[\"\u201c\u201d\u2018\u2019][^\"\u201c\u201d\u2018\u2019]{4,}[\"\u201c\u201d\u2018\u2019]")


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    # Discard sentences shorter than 8 chars (they are usually noise like footnote markers).
    return [p for p in parts if len(p) >= 8]


def _is_protected(sent: str) -> bool:
    if _NUM_RE.search(sent):
        return True
    if _QUOTE_RE.search(sent):
        return True
    return False


def _tokenize(s: str) -> list[str]:
    """Tokenize for TextRank; jieba for CJK, whitespace fallback otherwise."""
    if any("\u4e00" <= c <= "\u9fff" for c in s):
        return [t for t in jieba.cut(s) if t.strip()]
    return [t for t in s.lower().split() if t.strip()]


def _textrank_select(sentences: list[str], top_k: int) -> list[tuple[int, float]]:
    """Return list of (index, score) for the top-k sentences."""
    if not sentences:
        return []
    if len(sentences) <= top_k:
        return [(i, 1.0) for i in range(len(sentences))]

    tokens = [set(_tokenize(s)) for s in sentences]
    n = len(sentences)
    g = nx.Graph()
    for i in range(n):
        g.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(tokens[i] & tokens[j])
            if inter == 0:
                continue
            denom = (np.log(len(tokens[i]) + 1) + np.log(len(tokens[j]) + 1)) or 1.0
            w = inter / denom
            if w > 0:
                g.add_edge(i, j, weight=w)
    try:
        pr = nx.pagerank(g, weight="weight", max_iter=100)
    except Exception:  # noqa: BLE001
        pr = {i: 1.0 / n for i in range(n)}
    ranked = sorted(pr.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k]


class Compressor:
    """Three-stage compressor: L1 embedding -> L2 TextRank -> L3 protected spans."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        l1_threshold: float = 0.45,
        l2_top_k_per_chunk: int = 8,
        l3_protect: bool = True,
    ) -> None:
        self.embedder = embedder
        self.l1_threshold = l1_threshold
        self.l2_top_k_per_chunk = l2_top_k_per_chunk
        self.l3_protect = l3_protect

    def compress(
        self,
        query: str,
        chunks: list[Chunk],
        *,
        token_budget: int | None = None,
    ) -> CompressResult:
        stats = CompressStats(
            raw_chunks=len(chunks),
            raw_chars=sum(len(c.text) for c in chunks),
        )
        if not chunks:
            return CompressResult(stats=stats)

        # ---- L1: embedding filter on chunk level ----
        q_vec = self.embedder.encode_one(query)
        chunk_vecs = self.embedder.encode([c.text for c in chunks])
        sims = chunk_vecs @ q_vec
        kept_pairs = [
            (c, float(s)) for c, s in zip(chunks, sims, strict=True)
            if float(s) >= self.l1_threshold
        ]
        # If everything got filtered (low-quality fetches), keep top-3 anyway.
        if not kept_pairs:
            order = np.argsort(-sims)[:3]
            kept_pairs = [(chunks[i], float(sims[i])) for i in order]
        stats.after_l1_chunks = len(kept_pairs)
        stats.after_l1_chars = sum(len(c.text) for c, _ in kept_pairs)
        logger.debug(
            "L1: {}/{} chunks kept (threshold={:.2f})",
            stats.after_l1_chunks,
            stats.raw_chunks,
            self.l1_threshold,
        )

        # ---- L2: per-chunk sentence-level TextRank ----
        l2_sents: list[CompressedSentence] = []
        l3_extras: list[CompressedSentence] = []
        for chunk, _chunk_sim in kept_pairs:
            sentences = _split_sentences(chunk.text)
            if not sentences:
                continue
            top = _textrank_select(sentences, self.l2_top_k_per_chunk)
            kept_idx = {idx for idx, _ in top}
            for idx, score in top:
                l2_sents.append(
                    CompressedSentence(
                        chunk_url=chunk.url,
                        chunk_title=chunk.title,
                        text=sentences[idx],
                        score=float(score),
                        is_protected=False,
                    )
                )
            # ---- L3: protect any sentence with numbers / quotes / acronyms ----
            if self.l3_protect:
                for idx, sent in enumerate(sentences):
                    if idx in kept_idx:
                        continue
                    if _is_protected(sent):
                        l3_extras.append(
                            CompressedSentence(
                                chunk_url=chunk.url,
                                chunk_title=chunk.title,
                                text=sent,
                                score=0.0,
                                is_protected=True,
                            )
                        )

        stats.after_l2_sentences = len(l2_sents)
        stats.after_l2_chars = sum(len(s.text) for s in l2_sents)
        stats.protected_added = len(l3_extras)

        merged = l2_sents + l3_extras
        stats.after_l3_sentences = len(merged)
        stats.after_l3_chars = sum(len(s.text) for s in merged)

        # Optional token-budget pruning (rough char-based).
        if token_budget is not None:
            char_budget = token_budget * 3  # conservative ~1 token = 3 chars for CJK mixed
            cumulative = 0
            kept: list[CompressedSentence] = []
            # Order: protected first, then by score desc.
            ordered = sorted(merged, key=lambda s: (not s.is_protected, -s.score))
            for s in ordered:
                if cumulative + len(s.text) > char_budget:
                    break
                kept.append(s)
                cumulative += len(s.text)
            merged = kept
            stats.after_l3_sentences = len(merged)
            stats.after_l3_chars = sum(len(s.text) for s in merged)

        logger.info(
            "compress: chunks {}->{} | sents L2={} L3+={} | chars {}->{}",
            stats.raw_chunks,
            stats.after_l1_chunks,
            stats.after_l2_sentences,
            stats.protected_added,
            stats.raw_chars,
            stats.after_l3_chars,
        )
        return CompressResult(sentences=merged, stats=stats)
