import numpy as np
import pandas as pd
import pytest
import lightgbm as lgb

from tree_surgery.datasets import load_dataset
from tree_surgery.train import fit_base_model


@pytest.fixture(scope="session", params=[0, 1], ids=["seed0", "seed1"])
def delivery(request):
    """(SplitData, booster) for D1 at the plan's real settings, two seeds."""
    data = load_dataset("delivery", seed=request.param)
    booster, info = fit_base_model(data)
    return data, booster


@pytest.fixture(scope="session")
def nan_model():
    """Small model trained on data with NaNs, so splits carry a real missing-value direction."""
    rng = np.random.default_rng(7)
    n = 3000
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.uniform(0, 5, n)})
    y = 2 * X["a"] - X["b"] ** 2 + np.sin(X["c"]) + rng.normal(0, 0.1, n)
    X.loc[rng.random(n) < 0.15, "a"] = np.nan
    X.loc[rng.random(n) < 0.10, "c"] = np.nan
    params = dict(objective="regression", num_leaves=8, learning_rate=0.2, verbose=-1,
                  deterministic=True, force_row_wise=True, num_threads=1, seed=0)
    booster = lgb.train(params, lgb.Dataset(X, y.to_numpy()), num_boost_round=30)
    return X, y.to_numpy(), booster
