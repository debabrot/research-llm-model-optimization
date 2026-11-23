import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import wilcoxon

from tree_surgery.evaluation import bootstrap_ci, compare_arms, paired_differences, wilcoxon_p
from tree_surgery.runner import run_grid, summarize


@pytest.fixture
def toy_df():
    rows = []
    for ds in ("d1", "d2"):
        for s in range(5):
            rows.append({"dataset": ds, "seed": s, "arm": "b", "run": 0, "test_rmse": 1.0 + 0.1 * s})
            for run in range(3):  # 'a' has 3 runs per seed, always 0.05 better
                rows.append({"dataset": ds, "seed": s, "arm": "a", "run": run, "test_rmse": 0.95 + 0.1 * s})
    return pd.DataFrame(rows)


def test_paired_differences_broadcasts_runs(toy_df):
    d = paired_differences(toy_df, "a", "b")
    assert len(d) == 2 * 5 * 3
    np.testing.assert_allclose(d["diff"], -0.05)


def test_compare_arms(toy_df):
    c = compare_arms(toy_df, "a", "b")
    assert list(c.dataset) == ["d1", "d2"]
    assert (c.n == 15).all() and (c.a_wins == 15).all()
    assert c.mean_diff.tolist() == pytest.approx([-0.05, -0.05])
    assert c.ci_excludes_0.all()


def test_bootstrap_ci():
    x = np.random.default_rng(0).normal(1.0, 1.0, 50)
    lo, hi = bootstrap_ci(x)
    assert lo < x.mean() < hi
    assert np.isnan(bootstrap_ci([1.0])[0])
    assert bootstrap_ci([2.0, 2.0, 2.0]) == (2.0, 2.0)


def test_wilcoxon():
    x = np.array([-0.3, -0.1, -0.2, 0.05, -0.4])
    assert wilcoxon_p(x) == pytest.approx(wilcoxon(x).pvalue)
    assert np.isnan(wilcoxon_p(np.zeros(5)))


def test_run_grid_end_to_end(tmp_path):
    df = run_grid(["delivery"], [0], out_dir=tmp_path, verbose=False)
    assert set(df.arm) == {"original", "random_opt", "sparse_opt", "refit", "retrain"}
    for col in ("test_rmse", "test_mae", "test_r2", "cal_rmse", "val_rmse", "gen_gap",
                "test_rmse_vs_truth", "leaves_edited", "frac_params_changed", "llm_cost_usd"):
        assert col in df
    edit = df[df.arm.isin(["random_opt", "sparse_opt"])]
    assert edit.within_budget.all() and (edit.leaves_edited <= 10).all() and (edit.n_candidates <= 10).all()
    assert not df[df.arm.isin(["refit", "retrain"])].within_budget.any()

    for f in ("results.csv", "candidates.jsonl", "summary.md"):
        assert (tmp_path / f).exists()
    recs = [json.loads(l) for l in (tmp_path / "candidates.jsonl").read_text().splitlines()]
    assert recs and not any("test" in k for r in recs for k in r)
    assert sum(r["chosen"] for r in recs if r["arm"] == "sparse_opt") == 1
    assert "Paired differences" in summarize(df)
