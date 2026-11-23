"""Numerical leaf-edit optimizers (PLAN §1, §4, §6).

With the tree structure frozen, an edit is a delta vector ``d`` over leaves and the
edited model predicts ``y_hat_base + X @ d`` (``X`` = one-hot leaf design matrix).
Fitting an edit to calibration residuals ``r = y_cal - y_hat_base(cal)`` is
therefore linear regression on leaf indicators:

* **values for a fixed leaf subset S** -> ridge regression on ``X[:, S]``
  (closed form, :func:`ridge_on_subset`);
* **which k leaves** -> sparse regression: greedy OMP (:func:`omp_path`) or the
  Lasso regularisation path (:func:`lasso_supports`).

Ridge objective (PLAN §6, "on the mean-squared-error objective")::

    min_d  (1/n) * ||r - X_S d||^2  +  lambda * ||d||^2

so ``lambda`` is on the same scale for every dataset size. ``lambda`` is chosen on
validation by :func:`fit_ridge_select_lambda`; calibration is the only data the
deltas are fitted on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import scipy.sparse as sp

DEFAULT_LAMBDAS: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


# --------------------------------------------------------------------------- #
# Ridge on a leaf subset
# --------------------------------------------------------------------------- #
def _dense_cols(X: sp.spmatrix, cols: Sequence[int]) -> np.ndarray:
    return np.asarray(X[:, list(cols)].toarray(), dtype=np.float64)


def ridge_on_subset(X: sp.spmatrix, r: np.ndarray, cols: Sequence[int], lam: float) -> np.ndarray:
    """Closed-form ridge deltas for leaf columns ``cols``.

    Returns a length-``len(cols)`` vector ``d`` minimising
    ``(1/n)||r - X[:, cols] d||^2 + lam ||d||^2``. ``lam = 0`` is ordinary least
    squares (minimum-norm solution if ``X[:, cols]`` is rank deficient).
    """
    cols = list(cols)
    if not cols:
        return np.zeros(0)
    n = X.shape[0]
    A = _dense_cols(X, cols)
    G = A.T @ A / n
    b = A.T @ np.asarray(r, dtype=np.float64) / n
    if lam > 0:
        return np.linalg.solve(G + lam * np.eye(len(cols)), b)
    return np.linalg.lstsq(G, b, rcond=None)[0]


@dataclass
class RidgeFit:
    cols: list[int]
    deltas: np.ndarray          # aligned with cols
    lam: float
    cal_rmse: float
    val_rmse: float
    val_rmse_by_lambda: dict[float, float] = field(default_factory=dict)


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def fit_ridge_select_lambda(
    X_cal: sp.spmatrix,
    r_cal: np.ndarray,
    X_val: sp.spmatrix,
    r_val: np.ndarray,
    cols: Sequence[int],
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
) -> RidgeFit:
    """Fit ridge deltas on calibration for each lambda; keep the lambda with the best validation RMSE.

    ``r_*`` are residuals of the *base* model on that split. Ties go to the larger
    lambda (more shrinkage).
    """
    cols = [int(c) for c in cols]
    A_cal, A_val = _dense_cols(X_cal, cols), _dense_cols(X_val, cols)
    best: RidgeFit | None = None
    by_lam: dict[float, float] = {}
    for lam in sorted(lambdas, reverse=True):
        d = ridge_on_subset(X_cal, r_cal, cols, lam)
        val = _rmse(r_val - A_val @ d)
        by_lam[float(lam)] = val
        if best is None or val < best.val_rmse - 1e-15:
            best = RidgeFit(cols, d, float(lam), _rmse(r_cal - A_cal @ d), val)
    assert best is not None
    best.val_rmse_by_lambda = dict(sorted(by_lam.items()))
    return best


# --------------------------------------------------------------------------- #
# Which leaves: sparse selection
# --------------------------------------------------------------------------- #
def omp_path(X: sp.spmatrix, r: np.ndarray, k: int) -> list[int]:
    """Greedy Orthogonal Matching Pursuit on the leaf design matrix.

    At each step picks the leaf whose (unit-normalised) indicator column is most
    correlated with the current residual, then refits least squares on all leaves
    picked so far. Leaf columns can be exact duplicates on calibration (leaves in
    different trees covering the same rows), so the picked leaf may be a twin of the
    "true" one; the fitted function is the same. Returns the leaves in the order they were picked, so
    ``path[:j]`` is the OMP support of size ``j``. Leaves with no calibration rows
    are never picked (an edit there cannot be fitted). Stops early if the residual
    is exhausted or no leaf has non-zero correlation.
    """
    X = sp.csc_matrix(X, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel())
    usable = norms > 0
    inv_norm = np.where(usable, 1.0 / np.where(usable, norms, 1.0), 0.0)
    picked: list[int] = []
    resid = r.copy()
    tol = 1e-10 * max(float(np.linalg.norm(r)), 1e-300)
    for _ in range(k):
        if np.linalg.norm(resid) <= tol:
            break
        score = np.abs(X.T @ resid) * inv_norm
        if picked:
            score[picked] = -np.inf
        j = int(np.argmax(score))
        if not np.isfinite(score[j]) or score[j] <= 1e-12:
            break
        picked.append(j)
        d = ridge_on_subset(X, r, picked, lam=0.0)
        resid = r - _dense_cols(X, picked) @ d
    return picked


def lasso_supports(
    X: sp.spmatrix,
    r: np.ndarray,
    k: int,
    n_supports: int,
    n_alphas: int = 60,
    alpha_ratio: float = 0.02,
    max_iter: int = 2_000,
) -> list[list[int]]:
    """Distinct supports of size 1..k along the Lasso path, thinned to ``n_supports``.

    Columns are scaled to unit norm (so the L1 penalty treats small and large leaves
    alike, as OMP does); leaves with no calibration rows are dropped. Only the sparse
    top of the path matters (<= k leaves), so the alpha grid runs geometrically from
    ``alpha_max`` (empty support) down to ``alpha_ratio * alpha_max`` instead of the
    full path, which is far cheaper. Supports appear in path order (strongest penalty
    first); if there are more than ``n_supports`` with <= k leaves, evenly spaced ones
    are kept, always including the largest. Only supports are returned: values are
    refitted by ridge downstream.
    """
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import lasso_path

    X = sp.csc_matrix(X, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel())
    usable = np.flatnonzero(norms > 0)
    Xn = sp.csc_matrix(X[:, usable] @ sp.diags(1.0 / norms[usable]))
    alpha_max = float(np.max(np.abs(Xn.T @ r))) / len(r)
    if alpha_max <= 0:
        return []
    alphas = alpha_max * np.geomspace(1.0, alpha_ratio, n_alphas)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)
        _, coefs_u, _ = lasso_path(Xn, r, alphas=alphas, max_iter=max_iter, tol=1e-4)
    coefs = np.zeros((X.shape[1], coefs_u.shape[1]))
    coefs[usable] = coefs_u
    seen: set[tuple[int, ...]] = set()
    supports: list[list[int]] = []
    for j in range(coefs.shape[1]):
        nz = np.flatnonzero(np.abs(coefs[:, j]) > 0)
        if len(nz) == 0 or len(nz) > k:
            if len(nz) > k:
                break
            continue
        # order by |coef| so truncation (if a caller wants it) keeps the strongest
        nz = nz[np.argsort(-np.abs(coefs[nz, j]), kind="stable")]
        key = tuple(sorted(int(c) for c in nz))
        if key not in seen:
            seen.add(key)
            supports.append([int(c) for c in nz])
    if len(supports) > n_supports:
        idx = np.unique(np.round(np.linspace(0, len(supports) - 1, n_supports)).astype(int))
        supports = [supports[i] for i in idx]
    return supports
