"""Bootstrap CI and Cohen's d.

The CI implementation provides both the ordinary percentile bootstrap
and the BCa (bias-corrected and accelerated) variant; BCa is preferred
for small samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats


@dataclass
class CI:
    point: float
    low: float
    high: float
    method: str = "percentile"
    confidence: float = 0.95


def _percentile_ci(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.percentile(samples, 100 * alpha))
    hi = float(np.percentile(samples, 100 * (1.0 - alpha)))
    return lo, hi


def _bca_ci(
    data: np.ndarray,
    samples: np.ndarray,
    statistic_fn,
    confidence: float,
) -> tuple[float, float]:
    """BCa bootstrap CI. Falls back to percentile if z0/a are degenerate."""
    n = len(data)
    point = statistic_fn(data)

    # Bias correction z0
    p0 = float((samples < point).mean())
    if p0 in (0.0, 1.0):
        return _percentile_ci(samples, confidence)
    z0 = scipy_stats.norm.ppf(p0)

    # Acceleration via jackknife
    jack = np.empty(n)
    for i in range(n):
        jack[i] = statistic_fn(np.delete(data, i))
    jack_mean = jack.mean()
    num = ((jack_mean - jack) ** 3).sum()
    den = 6.0 * ((jack_mean - jack) ** 2).sum() ** 1.5
    if den == 0:
        return _percentile_ci(samples, confidence)
    a = num / den

    z_alpha = scipy_stats.norm.ppf((1.0 - confidence) / 2.0)
    z_1ma = scipy_stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    alpha1 = scipy_stats.norm.cdf(z0 + (z0 + z_alpha) / (1.0 - a * (z0 + z_alpha)))
    alpha2 = scipy_stats.norm.cdf(z0 + (z0 + z_1ma) / (1.0 - a * (z0 + z_1ma)))
    if not (np.isfinite(alpha1) and np.isfinite(alpha2)):
        return _percentile_ci(samples, confidence)
    lo = float(np.percentile(samples, 100 * alpha1))
    hi = float(np.percentile(samples, 100 * alpha2))
    return lo, hi


def bootstrap_ci(
    data: list[float] | np.ndarray,
    *,
    statistic: str = "mean",
    n_iters: int = 1000,
    confidence: float = 0.95,
    method: str = "bca",
    seed: int | None = 42,
) -> CI:
    """Bootstrap a confidence interval.

    Args:
        data: Observations.
        statistic: "mean" or "median".
        n_iters: Number of bootstrap replicates.
        confidence: e.g. 0.95.
        method: "percentile" or "bca".
        seed: rng seed for reproducibility.
    """
    arr = np.asarray([x for x in data if not (isinstance(x, float) and np.isnan(x))], dtype=float)
    if arr.size == 0:
        return CI(point=float("nan"), low=float("nan"), high=float("nan"), method=method, confidence=confidence)

    if statistic == "mean":
        stat_fn = np.mean
    elif statistic == "median":
        stat_fn = np.median
    else:
        raise ValueError(f"unsupported statistic: {statistic}")

    rng = np.random.default_rng(seed)
    samples = np.empty(n_iters)
    for i in range(n_iters):
        idx = rng.integers(0, len(arr), size=len(arr))
        samples[i] = stat_fn(arr[idx])

    point = float(stat_fn(arr))
    if method == "bca":
        lo, hi = _bca_ci(arr, samples, stat_fn, confidence)
    elif method == "percentile":
        lo, hi = _percentile_ci(samples, confidence)
    else:
        raise ValueError(f"unsupported CI method: {method}")
    return CI(point=point, low=lo, high=hi, method=method, confidence=confidence)


def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    """Cohen's d for the difference of means (pooled SD)."""
    a = np.asarray([x for x in group_a if not (isinstance(x, float) and np.isnan(x))], dtype=float)
    b = np.asarray([x for x in group_b if not (isinstance(x, float) and np.isnan(x))], dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    pooled = np.sqrt(((a.size - 1) * var_a + (b.size - 1) * var_b) / (a.size + b.size - 2))
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def paired_cohens_d(group_a: list[float], group_b: list[float]) -> float:
    """Cohen's d_z for paired observations.

    d_z = mean(diff) / sd(diff), where diff = a - b.

    More appropriate than independent-samples Cohen's d for ablation
    studies where the same questions are scored under different configs.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if a.size != b.size or a.size < 2:
        return float("nan")
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if a.size < 2:
        return float("nan")
    diff = a - b
    sd = diff.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(diff.mean() / sd)


def paired_diff_ci(
    group_a: list[float],
    group_b: list[float],
    *,
    n_iters: int = 1000,
    confidence: float = 0.95,
    seed: int | None = 42,
) -> CI:
    """Bootstrap CI of the paired mean difference (a - b)."""
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if a.size != b.size or a.size == 0:
        return CI(point=float("nan"), low=float("nan"), high=float("nan"))
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if a.size == 0:
        return CI(point=float("nan"), low=float("nan"), high=float("nan"))
    diff = a - b
    rng = np.random.default_rng(seed)
    samples = np.empty(n_iters)
    for i in range(n_iters):
        idx = rng.integers(0, len(diff), size=len(diff))
        samples[i] = diff[idx].mean()
    lo, hi = _percentile_ci(samples, confidence)
    return CI(point=float(diff.mean()), low=lo, high=hi, method="paired-percentile",
              confidence=confidence)
