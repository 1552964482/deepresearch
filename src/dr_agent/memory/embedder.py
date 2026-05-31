"""Local embedder with on-disk + in-process LRU caching.

Uses ``sentence-transformers`` with ``BAAI/bge-small-zh-v1.5`` (95 MB, 512-dim,
already L2-normalized so cosine similarity == dot product).

GPU is auto-detected; falls back to CPU if no CUDA. Embeddings are cached
in-process keyed by (model, text-hash).
"""

from __future__ import annotations

import hashlib
import os
import threading
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


# Anaconda + torch OpenMP conflict workaround
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


_DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
_DEFAULT_DIM = 512


def _hash(text: str, model: str) -> str:
    h = hashlib.sha1()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


class Embedder:
    """Sentence-transformer-backed embedder with LRU cache."""

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str | None = None,
        cache_size: int = 10_000,
    ) -> None:
        self.model_name = model_name
        self._cache_size = cache_size
        self._cache: dict[str, np.ndarray] = {}
        self._cache_order: list[str] = []
        self._lock = threading.Lock()
        self._device = device or self._auto_device()
        self._model: "SentenceTransformer | None" = None
        logger.info(
            "Embedder configured: model={}, device={} (lazy-loaded on first use)",
            model_name,
            self._device,
        )

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

    def _ensure_model(self) -> "SentenceTransformer":
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self._device)
            # Warm up: forces weight materialization off the meta device and
            # primes the CUDA context. Without this, the first call from a
            # worker thread can raise "Cannot copy out of meta tensor".
            try:
                self._model.encode(
                    ["__warmup__"], convert_to_numpy=True, show_progress_bar=False
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Embedder warmup failed: {}", e)
            logger.info("Embedder loaded model on {}", self._device)
        return self._model

    def warmup(self) -> None:
        """Force-load the model in the current thread (caller-controlled)."""
        self._ensure_model()

    @property
    def dim(self) -> int:
        # Known for bge-small-zh-v1.5; for other models we fall back to encoding.
        if self.model_name == _DEFAULT_MODEL:
            return _DEFAULT_DIM
        return int(self._ensure_model().get_sentence_embedding_dimension() or _DEFAULT_DIM)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode a batch of texts into L2-normalized vectors.

        Returns a (N, dim) float32 numpy array.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        keys = [_hash(t, self.model_name) for t in texts]
        out: list[np.ndarray | None] = [None] * len(texts)

        # Cache lookup
        miss_idx: list[int] = []
        miss_texts: list[str] = []
        with self._lock:
            for i, k in enumerate(keys):
                if k in self._cache:
                    out[i] = self._cache[k]
                else:
                    miss_idx.append(i)
                    miss_texts.append(texts[i])

        # Encode misses
        if miss_texts:
            model = self._ensure_model()
            vecs = model.encode(
                miss_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype(np.float32)
            with self._lock:
                for i, v in zip(miss_idx, vecs, strict=True):
                    self._cache[keys[i]] = v
                    self._cache_order.append(keys[i])
                    out[i] = v
                # LRU eviction
                while len(self._cache_order) > self._cache_size:
                    old = self._cache_order.pop(0)
                    self._cache.pop(old, None)

        return np.stack([o for o in out if o is not None], axis=0)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def cache_stats(self) -> dict[str, int]:
        with self._lock:
            return {"size": len(self._cache), "limit": self._cache_size}
