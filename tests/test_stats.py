"""Tests for bootstrap CI and Cohen's d (eval/stats.py)."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
from scipy import stats as scipy_stats

from dr_agent.eval.stats import bootstrap_ci, cohens_d


def test_bootstrap_ci_contains_point_estimate() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(loc=0.5, scale=0.1, size=50).tolist()
    ci = bootstrap_ci(data, n_iters=500)
    assert ci.low <= ci.point <= ci.high


def test_bootstrap_ci_widens_with_smaller_sample() -> None:
    rng = np.random.default_rng(1)
    big = rng.normal(loc=0.5, scale=0.2, size=200).tolist()
    small = big[:20]
    ci_big = bootstrap_ci(big, n_iters=500, method="percentile")
    ci_small = bootstrap_ci(small, n_iters=500, method="percentile")
    assert (ci_small.high - ci_small.low) > (ci_big.high - ci_big.low)


def test_bootstrap_ci_close_to_normal_theory() -> None:
    """For a roughly Gaussian sample, bootstrap CI should be close to a t-CI."""
    rng = np.random.default_rng(2)
    data = rng.normal(loc=10.0, scale=2.0, size=100).tolist()
    ci = bootstrap_ci(data, n_iters=2000, method="percentile")
    # t-based CI for the mean
    arr = np.asarray(data)
    t_lo, t_hi = scipy_stats.t.interval(
        0.95, df=len(arr) - 1, loc=arr.mean(), scale=scipy_stats.sem(arr)
    )
    # Within 0.15 — bootstrap with 2000 iters has noise but should be close.
    assert math.isclose(ci.low, t_lo, abs_tol=0.15)
    assert math.isclose(ci.high, t_hi, abs_tol=0.15)


def test_bootstrap_handles_nan_and_empty() -> None:
    ci_empty = bootstrap_ci([], n_iters=100)
    assert math.isnan(ci_empty.point)
    ci_nan = bootstrap_ci([float("nan"), float("nan")], n_iters=100)
    assert math.isnan(ci_nan.point)


def test_bootstrap_seed_is_deterministic() -> None:
    data = list(range(50))
    a = bootstrap_ci(data, n_iters=300, seed=123)
    b = bootstrap_ci(data, n_iters=300, seed=123)
    assert a.low == b.low
    assert a.high == b.high


def test_cohens_d_zero_for_identical_groups() -> None:
    rng = np.random.default_rng(7)
    data = rng.normal(0, 1, 50).tolist()
    assert cohens_d(data, data) == pytest.approx(0.0, abs=1e-12)


def test_cohens_d_signs_correctly() -> None:
    rng = np.random.default_rng(8)
    a = rng.normal(0.0, 1.0, 80).tolist()
    b = rng.normal(1.0, 1.0, 80).tolist()
    d = cohens_d(a, b)
    # mean(a) < mean(b), so (a - b) / pooled is negative
    assert d < -0.5


def test_cohens_d_thresholds() -> None:
    """Smoke check the magnitude buckets used in our compare table."""
    rng = np.random.default_rng(9)
    base = rng.normal(0.0, 1.0, 200).tolist()
    small = rng.normal(0.2, 1.0, 200).tolist()
    medium = rng.normal(0.5, 1.0, 200).tolist()
    large = rng.normal(0.8, 1.0, 200).tolist()
    # group_a vs group_b => d for (a - b)/pooled. Use base as "a", treatment as "b".
    d_small = cohens_d(small, base)
    d_med = cohens_d(medium, base)
    d_large = cohens_d(large, base)
    assert 0.05 < d_small < 0.45
    assert 0.30 < d_med < 0.70
    assert 0.55 < d_large < 1.00


def test_cohens_d_empty_returns_nan() -> None:
    assert math.isnan(cohens_d([1.0], [2.0]))
    assert math.isnan(cohens_d([], []))
