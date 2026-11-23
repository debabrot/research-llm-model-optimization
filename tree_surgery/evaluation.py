"""Metrics (PLAN §8). Phase 1 has point metrics only; paired stats and bootstrap CIs land in Phase 2."""

from __future__ import annotations

import numpy as np


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
