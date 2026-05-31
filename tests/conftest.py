"""Shared pytest fixtures."""

from __future__ import annotations

import os

# Anaconda + torch OpenMP workaround. Must precede any import that pulls torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
