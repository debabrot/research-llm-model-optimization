import copy

import numpy as np
import pytest

from tree_surgery.arms import (
    ARMS,
    BudgetError,
    EditArm,
    build_context,
    get_arm,
)
from tree_surgery.config import load_config
from tree_surgery.design_matrix import leaf_values
from tree_surgery.evaluation import rmse

CFG = load_config()
EDIT = ("random_opt", "sparse_opt")


@pytest.fixture(scope="module")
def ctx(delivery):
    data, booster = delivery
    return build_context(data, booster, CFG)


@pytest.fixture(scope="module")
def outputs(ctx):
    return {name: get_arm(name).run(ctx) for name in ARMS}


# ---- isolation -------------------------------------------------------------- #
def test_context_has_no_test_split(ctx, delivery):
    data, _ = delivery
    assert "test" not in ctx.splits
    seen = np.concatenate([s.idx for s in ctx.splits.values()])
    assert not np.isin(data["test"].idx, seen).any()
    assert ctx.X_cal.shape[0] == len(data["cal"].y) and ctx.X_val.shape[0] == len(data["val"].y)


@pytest.mark.parametrize("arm", EDIT + ("refit",))
def test_arm_outcome_independent_of_test_labels(delivery, outputs, arm):
    data, booster = delivery
    poisoned = copy.deepcopy(data)
    poisoned.splits["test"].y = np.full_like(poisoned["test"].y, np.nan)
    poisoned.splits["test"].X = poisoned["test"].X * np.nan
    out = get_arm(arm).run(build_context(poisoned, booster, CFG))
    ref = outputs[arm]
    assert out.chosen == ref.chosen
    assert [c.val_rmse for c in out.candidates] == [c.val_rmse for c in ref.candidates]


def test_arms_do_not_mutate_base(ctx, outputs, delivery):
    data, booster = delivery
    np.testing.assert_array_equal(leaf_values(booster, ctx.index), ctx.patcher.base_values)
    np.testing.assert_array_equal(booster.predict(data["val"].X), ctx.base_pred["val"])


# ---- budget ---------------------------------------------------------------- #
@pytest.mark.parametrize("arm", EDIT)
def test_edit_arms_respect_budget(outputs, arm):
    out = outputs[arm]
    assert out.within_budget
    assert 1 <= len(out.candidates) <= CFG["edit_budget"]["max_candidates"]
    assert all(len(c.leaves) <= CFG["edit_budget"]["max_leaves"] for c in out.candidates)
    assert out.leaves_edited <= CFG["edit_budget"]["max_leaves"]


class _Rogue(EditArm):
    name = "rogue"

    def __init__(self, subsets):
        self.subsets = subsets

    def propose(self, ctx):
        return self.subsets


def test_budget_violations_raise(ctx):
    with pytest.raises(BudgetError, match="leaves"):
        _Rogue([list(range(11))]).run(ctx)
    with pytest.raises(BudgetError, match="candidates"):
        _Rogue([[i] for i in range(11)]).run(ctx)
    with pytest.raises(BudgetError, match="twice"):
        _Rogue([[3, 3]]).run(ctx)
    with pytest.raises(BudgetError, match="exist"):
        _Rogue([[ctx.index.num_leaves]]).run(ctx)


# ---- selection and materialisation ---------------------------------------- #
@pytest.mark.parametrize("arm", EDIT + ("refit",))
def test_chosen_is_best_on_validation(outputs, arm):
    out = outputs[arm]
    vals = [c.val_rmse for c in out.candidates if c.status == "ok"]
    assert out.candidates[out.chosen].val_rmse == min(vals)


@pytest.mark.parametrize("arm", EDIT)
def test_final_model_matches_linear_prediction(ctx, outputs, arm):
    out = outputs[arm]
    best = out.candidates[out.chosen]
    cols = [ctx.index.col(*k) for k in best.leaves]
    linear = ctx.base_pred["val"] + ctx.X_val[:, cols] @ np.asarray(best.deltas)
    np.testing.assert_allclose(out.model.predict(ctx.splits["val"].X), linear, atol=1e-9)
    assert rmse(ctx.splits["val"].y, out.model.predict(ctx.splits["val"].X)) == pytest.approx(best.val_rmse, abs=1e-9)
    assert out.info["linearity_err"] < 1e-9


@pytest.mark.parametrize("arm", EDIT)
def test_edit_arms_never_hurt_calibration(ctx, outputs, arm):
    base_cal = rmse(ctx.splits["cal"].y, ctx.base_pred["cal"])
    assert all(c.cal_rmse <= base_cal + 1e-12 for c in outputs[arm].candidates)


def test_original_is_base(ctx, outputs):
    out = outputs["original"]
    np.testing.assert_array_equal(out.model.predict(ctx.splits["val"].X), ctx.base_pred["val"])
    assert out.leaves_edited == 0


def test_random_opt_deterministic_and_fittable(ctx, outputs):
    again = get_arm("random_opt").run(ctx)
    assert [c.leaves for c in again.candidates] == [c.leaves for c in outputs["random_opt"].candidates]
    n_cal = ctx.n_cal_per_leaf
    for c in again.candidates:
        assert len(c.leaves) == 10
        assert all(n_cal[ctx.index.col(*k)] >= 1 for k in c.leaves)


def test_random_opt_differs_across_seeds(delivery):
    data, booster = delivery
    ctx = build_context(data, booster, CFG)
    ctx2 = copy.copy(ctx)
    ctx2.seed = ctx.seed + 100
    a = get_arm("random_opt").propose(ctx)
    b = get_arm("random_opt").propose(ctx2)
    assert a != b


def test_sparse_opt_omp_candidates_are_nested(outputs):
    cands = outputs["sparse_opt"].candidates
    sizes = [len(c.leaves) for c in cands]
    assert sizes == sorted(sizes) and sizes[0] == 1
    for a, b in zip(cands, cands[1:]):
        assert set(a.leaves) <= set(b.leaves)


def test_sparse_opt_lasso_method(delivery):
    data, booster = delivery
    cfg = copy.deepcopy(CFG)
    cfg["arms"]["sparse_opt"]["method"] = "lasso"
    out = get_arm("sparse_opt").run(build_context(data, booster, cfg))
    assert 1 <= len(out.candidates) <= 10
    assert all(1 <= len(c.leaves) <= 10 for c in out.candidates)


# ---- corruption repair: sparse_opt should find planted damage -------------- #
def test_sparse_opt_repairs_obvious_corruption(delivery):
    """PLAN §9 C1 preview: +2·std(y) on 3 high-traffic leaves; calibration is clean."""
    data, booster = delivery
    clean_ctx = build_context(data, booster, CFG)
    n_cal = clean_ctx.n_cal_per_leaf
    rng = np.random.default_rng(0)
    trees = rng.choice(clean_ctx.index.num_trees, size=3, replace=False)
    corrupted = []
    for t in trees:
        cols = [clean_ctx.index.col(int(t), l) for l in range(clean_ctx.index.leaves_per_tree[t])]
        corrupted.append(clean_ctx.index.key(max(cols, key=lambda c: n_cal[c])))
    delta = 2 * float(np.std(data["train"].y))
    bad = clean_ctx.patcher.candidate().apply({k: delta for k in corrupted}).booster

    ctx = build_context(data, bad, CFG)
    out = get_arm("sparse_opt").run(ctx)
    # Leaf-level recall, counting a leaf with an identical calibration column as a hit.
    # Exact recovery is not identifiable in general (PLAN §9): a leaf holding most of a
    # tree's rows can be repaired through its siblings plus a shift, so require 2/3 here
    # and put the weight on function-level recovery below.
    Xc = ctx.X_cal.tocsc()
    col = lambda k: Xc[:, ctx.index.col(*k)].toarray().ravel()
    edited = [col(k) for k in out.candidates[out.chosen].leaves]
    recall = sum(any(np.array_equal(col(k), e) for e in edited) for k in corrupted)
    assert recall >= 2, f"recall {recall}/3"

    val = data["val"]
    clean_rmse = rmse(val.y, booster.predict(val.X))
    bad_rmse = rmse(val.y, bad.predict(val.X))
    fixed_rmse = rmse(val.y, out.model.predict(val.X))
    recovered = (bad_rmse - fixed_rmse) / (bad_rmse - clean_rmse)
    assert recovered > 0.95


# ---- ceilings ---------------------------------------------------------------- #
def test_refit(outputs):
    out = outputs["refit"]
    assert not out.within_budget
    assert out.info["decay_rate"] in CFG["arms"]["refit"]["decay_rates"]
    assert out.leaves_edited > CFG["edit_budget"]["max_leaves"]


def test_retrain(ctx, outputs):
    out = outputs["retrain"]
    assert not out.within_budget and out.leaves_edited is None
    assert 0 < out.model.num_trees() < CFG["base_model"]["num_boost_round"]
    val = ctx.splits["val"]
    assert rmse(val.y, out.model.predict(val.X)) == pytest.approx(out.candidates[0].val_rmse)


def test_unknown_arm():
    with pytest.raises(ValueError, match="unknown arm"):
        get_arm("llm_select")
