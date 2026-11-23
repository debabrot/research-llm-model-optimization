"""Metrics and paired statistics (PLAN §8).

Point metrics (RMSE / MAE / R²) plus the paired analysis used by the pre-registered
decision rule: per-(dataset, seed) differences between two arms, their mean, a 95%
bootstrap CI and a Wilcoxon signed-rank test.

With 5 seeds and one deterministic run per arm, the smallest two-sided Wilcoxon
p-value possible is 1/16 = 0.0625, so the bootstrap CI carries the decision; the
Wilcoxon p is reported alongside it.

With several LLM runs per seed, the pre-registered CI resamples (seed, run) pairs as
if independent. A seed-clustered CI (:func:`cluster_bootstrap_ci`) is reported next to
it; with one run per seed the two coincide.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y)
    ss_res = np.sum((y - np.asarray(pred)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"rmse": rmse(y, pred), "mae": mae(y, pred), "r2": r2(y, pred)}


# --------------------------------------------------------------------------- #
# Paired statistics
# --------------------------------------------------------------------------- #
PAIR_KEYS = ("dataset", "seed")


def paired_differences(
    df: pd.DataFrame,
    arm_a: str,
    arm_b: str,
    metric: str = "test_rmse",
    keys: Sequence[str] = PAIR_KEYS,
    arm_col: str = "arm",
) -> pd.DataFrame:
    """``metric(a) - metric(b)`` for every row of arm ``a``, paired with arm ``b`` on ``keys``.

    If ``a`` has several rows per key (e.g. 3 LLM runs per seed), each row is paired
    with ``b``'s value for that key (``b``'s rows are averaged first if it has
    several). Negative differences mean ``a`` has the lower metric.
    """
    keys = list(keys)
    a = df[df[arm_col] == arm_a]
    b = df[df[arm_col] == arm_b].groupby(keys, as_index=False)[metric].mean()
    m = a.merge(b, on=keys, suffixes=("_a", "_b"), how="inner")
    out = m[keys].copy()
    out["a"], out["b"] = m[f"{metric}_a"].to_numpy(), m[f"{metric}_b"].to_numpy()
    out["diff"] = out["a"] - out["b"]
    return out.reset_index(drop=True)


def bootstrap_ci(
    x: Iterable[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``x``. NaN for fewer than 2 values (undefined)."""
    x = np.asarray(list(x), dtype=float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    if np.all(x == x[0]):
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def cluster_bootstrap_ci(
    diffs: pd.DataFrame, cluster: str = "seed", n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Two-stage bootstrap: resample clusters (seeds), then rows (LLM runs) within each.

    Use when there are several LLM runs per seed: the runs share a seed's data split and
    base model, so treating them as independent (``bootstrap_ci``) makes the CI too narrow.
    Equals the plain bootstrap when every cluster has one row.
    """
    groups = [g["diff"].to_numpy(float) for _, g in diffs.groupby(cluster, sort=True)]
    if len(groups) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    k = len(groups)
    means = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, k, size=k)
        vals = [groups[i][rng.integers(0, len(groups[i]), size=len(groups[i]))] for i in picked]
        means[b] = np.mean(np.concatenate(vals))
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def wilcoxon_p(x: Iterable[float]) -> float:
    """Two-sided Wilcoxon signed-rank p-value; NaN if undefined (no non-zero differences)."""
    from scipy.stats import wilcoxon

    x = np.asarray(list(x), dtype=float)
    if len(x) == 0 or np.all(x == 0):
        return float("nan")
    return float(wilcoxon(x, zero_method="wilcox", alternative="two-sided").pvalue)


def compare_arms(
    df: pd.DataFrame,
    arm_a: str,
    arm_b: str,
    metric: str = "test_rmse",
    by: str = "dataset",
    n_boot: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per ``by`` group: mean paired difference, 95% bootstrap CI, Wilcoxon p, a-wins."""
    diffs = paired_differences(df, arm_a, arm_b, metric)
    rows = []
    for g, d in diffs.groupby(by, sort=True):
        x = d["diff"].to_numpy()
        lo, hi = bootstrap_ci(x, n_boot=n_boot, seed=seed)
        clo, chi = cluster_bootstrap_ci(d, "seed", n_boot=n_boot, seed=seed)
        rows.append(
            {
                by: g,
                "a": arm_a,
                "b": arm_b,
                "metric": metric,
                "n": len(x),
                "mean_diff": float(x.mean()),
                "ci_lo": lo,
                "ci_hi": hi,
                "ci_excludes_0": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
                "n_seeds": int(d["seed"].nunique()),
                "cluster_ci_lo": clo,
                "cluster_ci_hi": chi,
                "cluster_ci_excludes_0": bool(np.isfinite(clo) and (clo > 0 or chi < 0)),
                "wilcoxon_p": wilcoxon_p(x),
                "a_wins": int(np.sum(x < 0)),
                "mean_rel_diff_pct": float(100 * np.mean(x / d["b"].to_numpy())),
            }
        )
    return pd.DataFrame(rows)
