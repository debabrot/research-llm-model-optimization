import numpy as np
import pytest
import scipy.sparse as sp

from tree_surgery.design_matrix import leaf_design_matrix
from tree_surgery.optimizers import (
    fit_ridge_select_lambda,
    lasso_supports,
    omp_path,
    ridge_on_subset,
)


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    X = sp.csr_matrix((rng.random((400, 30)) < 0.2).astype(float))
    r = rng.normal(size=400)
    return X, r


def test_ridge_matches_closed_form(toy):
    X, r = toy
    cols = [1, 4, 7, 20]
    A = X[:, cols].toarray()
    n = len(r)
    for lam in (1e-3, 0.1, 10.0):
        expected = np.linalg.solve(A.T @ A / n + lam * np.eye(len(cols)), A.T @ r / n)
        np.testing.assert_allclose(ridge_on_subset(X, r, cols, lam), expected, rtol=1e-10, atol=1e-12)


def test_ridge_lambda_zero_is_ols(toy):
    X, r = toy
    cols = [0, 2, 3]
    A = X[:, cols].toarray()
    np.testing.assert_allclose(ridge_on_subset(X, r, cols, 0.0), np.linalg.lstsq(A, r, rcond=None)[0], atol=1e-10)


def test_ridge_shrinks_with_lambda(toy):
    X, r = toy
    norms = [np.linalg.norm(ridge_on_subset(X, r, [1, 2, 3], lam)) for lam in (1e-3, 1e-1, 1, 10, 100)]
    assert all(a >= b for a, b in zip(norms, norms[1:]))


def test_ridge_never_worse_than_no_edit_on_fit_data(toy):
    X, r = toy
    for lam in (1e-3, 1, 100):
        d = ridge_on_subset(X, r, [5, 6, 7, 8], lam)
        assert np.mean((r - X[:, [5, 6, 7, 8]] @ d) ** 2) <= np.mean(r ** 2) + 1e-15


def test_lambda_selected_on_validation(toy):
    X, r = toy
    rng = np.random.default_rng(1)
    Xv = sp.csr_matrix((rng.random((300, 30)) < 0.2).astype(float))
    rv = rng.normal(size=300)
    fit = fit_ridge_select_lambda(X, r, Xv, rv, [3, 9, 11])
    assert set(fit.val_rmse_by_lambda) == {1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0}
    assert fit.val_rmse == pytest.approx(min(fit.val_rmse_by_lambda.values()))
    assert fit.val_rmse_by_lambda[fit.lam] == pytest.approx(fit.val_rmse)


# ---- sparse selection on a real leaf design matrix ------------------------ #
@pytest.fixture(scope="module")
def real_X(delivery):
    data, booster = delivery
    return leaf_design_matrix(booster, data["cal"].X)


def _planted(X, k=3, seed=0):
    counts = np.asarray(X.sum(axis=0)).ravel()
    rng = np.random.default_rng(seed)
    cols = rng.choice(np.flatnonzero(counts >= 100), size=k, replace=False)
    d = np.zeros(X.shape[1])
    d[cols] = rng.choice([-1, 1], size=k) * rng.uniform(3, 5, size=k)
    return sorted(cols.tolist()), d


def recall_up_to_twins(X, planted, picked):
    """Planted leaves found, counting a picked leaf with an identical calibration column as a hit."""
    Xc = sp.csc_matrix(X)
    picked_cols = [Xc[:, p].toarray().ravel() for p in picked]
    return sum(any(np.array_equal(Xc[:, c].toarray().ravel(), q) for q in picked_cols) for c in planted)


@pytest.mark.parametrize("plant_seed", [0, 1, 2, 3])
def test_omp_recovers_planted_function_noiseless(real_X, plant_seed):
    cols, d = _planted(real_X, seed=plant_seed)
    r = real_X @ d
    path = omp_path(real_X, r, 10)
    fit = ridge_on_subset(real_X, r, path, 0.0)
    assert np.sqrt(np.mean((r - real_X[:, path] @ fit) ** 2)) < 1e-8 * np.sqrt(np.mean(r ** 2))
    assert recall_up_to_twins(real_X, cols, path) == 3
    assert len(path) < 10  # stops once the residual is fitted


def test_omp_recovers_planted_leaves_with_noise(real_X):
    cols, d = _planted(real_X, seed=3)
    r = real_X @ d + np.random.default_rng(0).normal(0, 0.5, real_X.shape[0])
    assert recall_up_to_twins(real_X, cols, omp_path(real_X, r, 10)) == 3


def test_omp_path_shape(real_X):
    r = np.random.default_rng(0).normal(size=real_X.shape[0])
    path = omp_path(real_X, r, 10)
    counts = np.asarray(real_X.sum(axis=0)).ravel()
    assert 1 <= len(path) <= 10
    assert len(set(path)) == len(path)
    assert all(counts[c] > 0 for c in path)


def test_omp_stops_when_residual_is_zero(real_X):
    assert omp_path(real_X, np.zeros(real_X.shape[0]), 10) == []


def test_lasso_supports_respect_budget(real_X):
    cols, d = _planted(real_X, k=4, seed=5)
    r = real_X @ d + np.random.default_rng(1).normal(0, 1.0, real_X.shape[0])
    sups = lasso_supports(real_X, r, k=10, n_supports=10)
    assert 1 <= len(sups) <= 10
    assert all(1 <= len(s) <= 10 for s in sups)
    assert len({tuple(sorted(s)) for s in sups}) == len(sups)
    # the largest support should contain most planted leaves
    assert len(set(cols) & set(max(sups, key=len))) >= 3
