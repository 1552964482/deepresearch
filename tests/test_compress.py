"""Tests for the three-stage compressor.

These tests use a fake :class:`Embedder` that does not load any real model,
so they're fast and offline.
"""

from __future__ import annotations

import hashlib

import numpy as np

from dr_agent.memory.compress import (
    CompressedSentence,
    Compressor,
    _is_protected,
    _split_sentences,
)
from dr_agent.tools.fetcher import Chunk


class _FakeEmbedder:
    """Deterministic 8-dim embedder with simple keyword-based directions.

    Texts containing the *target_keyword* receive a vector aligned with the
    target axis (axis 0); other texts get a near-orthogonal vector.
    """

    dim = 8

    def __init__(self, target_keyword: str = "GRPO") -> None:
        self.target_keyword = target_keyword

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vecs = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            if self.target_keyword in t:
                # Strong signal on axis 0
                v[0] = 1.0
                v[1] = 0.05  # small noise
            else:
                # Use hash to pick an off-axis direction; never on axis 0
                h = int(hashlib.md5(t.encode()).hexdigest(), 16)
                idx = (h % 6) + 2  # 2..7
                v[idx] = 1.0
            n = np.linalg.norm(v)
            if n > 0:
                v /= n
            vecs.append(v)
        return np.stack(vecs, axis=0)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def _make_chunk(text: str, *, url: str = "https://x/page") -> Chunk:
    return Chunk(url=url, title="demo", text=text, source_idx=0)


def test_split_sentences_drops_short_fragments() -> None:
    text = "好。短句。这是一个明显较长的句子，足以保留下来。"
    out = _split_sentences(text)
    assert any("足以保留" in s for s in out)
    # short fragments under the 8-char threshold are dropped
    assert all(len(s) >= 8 for s in out)


def test_is_protected_recognizes_numbers_and_quotes() -> None:
    assert _is_protected("InstructGPT 论文于 2022 年发表") is True
    assert _is_protected("准确率达到 95%") is True
    assert _is_protected("OpenAI 发布了 ChatGPT") is True  # acronym OpenAI? actually 'OpenAI' matches [A-Z]{2,}
    assert _is_protected("普通陈述句没有特殊标记") is False
    assert _is_protected('正如他所说："核心是相对优势估计"。') is True


def test_compressor_filters_off_topic_chunks() -> None:
    embedder = _FakeEmbedder(target_keyword="GRPO")
    compressor = Compressor(embedder, l1_threshold=0.45, l2_top_k_per_chunk=4)
    chunks = [
        _make_chunk(
            "GRPO 是一种强化学习算法。它移除了价值函数。它使用组内相对优势。它由 DeepSeek 提出。"
        ),
        _make_chunk("今天天气真好。我去散步了。看见了一只猫。猫很可爱。"),
        _make_chunk("GRPO 训练效率高。GRPO 节省显存。GRPO 适合 LLM。GRPO 由 DeepSeek 提出。"),
    ]
    result = compressor.compress("什么是 GRPO 算法", chunks)
    # Off-topic chunk should be filtered at L1
    assert result.stats.raw_chunks == 3
    assert result.stats.after_l1_chunks == 2
    # No surviving sentence should come from the off-topic chunk
    for s in result.sentences:
        assert "天气" not in s.text and "散步" not in s.text


def test_l3_protects_numbers_and_years() -> None:
    embedder = _FakeEmbedder(target_keyword="Transformer")
    compressor = Compressor(embedder, l1_threshold=0.3, l2_top_k_per_chunk=2)
    text = (
        "Transformer 是 2017 年提出的架构。"
        "Transformer 有很多组件。"
        "Transformer 应用在 NLP 任务。"
        "Transformer 改变了世界。"
        "Transformer 论文标题是 Attention Is All You Need。"
        "在 GLUE 基准上准确率提升了 12.3% 这是一个重要里程碑。"
    )
    result = compressor.compress("Transformer 历史", [_make_chunk(text)])
    # The 12.3% sentence and 2017 sentence must be present (either via L2 or L3)
    joined = result.joined()
    assert "12.3%" in joined
    assert "2017" in joined


def test_l3_recovers_dropped_protected_sentence() -> None:
    """When L2 leaves out a sentence that contains a number / year /
    quote, L3 should re-include it via the protected-spans path."""
    embedder = _FakeEmbedder(target_keyword="key")
    # top_k=2 forces L2 to drop 3 of 5 sentences; the protected one should
    # be recovered by L3 if it isn't in the top-2.
    compressor = Compressor(embedder, l1_threshold=0.0, l2_top_k_per_chunk=2)
    text = (
        "key 第一句普通观察足够长。"
        "key 第二句普通观察足够长。"
        "key 第三句普通观察足够长。"
        "key 第四句普通观察足够长。"
        "key 在某次实验中准确率高达 95% 之多。"
    )
    result = compressor.compress("key 关键问题", [_make_chunk(text)])
    # L2 keeps 2 sentences; L3 must add the protected one if it was dropped.
    joined = result.joined()
    assert "95%" in joined
    # Protected count is at least 1 unless L2 itself happened to keep it.
    if "95%" not in {s.text for s in result.sentences if not s.is_protected}:
        assert result.stats.protected_added >= 1
