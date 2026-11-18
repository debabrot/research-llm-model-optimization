"""Phase 1 acceptance tests (PLAN §11) plus edit validation."""

import numpy as np
import pytest

from tree_surgery.design_matrix import leaf_assignments, leaf_design_matrix
from tree_surgery.patcher import EditError, ModelPatcher, load_candidate

ATOL = 1e-9


def _pick_leaves(patcher, k, rng):
    """k distinct (tree, leaf) pairs spread across trees."""
    cols = rng.choice(patcher.index.num_leaves, size=k, replace=False)
    return [patcher.index.key(int(c)) for c in cols]


def _populated_leaf(booster, X, tree_id=None):
    """A (tree, leaf) that has samples in X, and one with no samples in X if any."""
    leaves = leaf_assignments(booster, X)
    t = tree_id if tree_id is not None else booster.num_trees() // 2
    vals, counts = np.unique(leaves[:, t], return_counts=True)
    return t, int(vals[np.argmax(counts)])


# --------------------------------------------------------------------------- #
# Acceptance 1: editing leaf (t, l) by delta moves exactly the samples in that leaf, by exactly delta
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("delta", [0.5, -3.25, 1e-3])
def test_single_leaf_edit_moves_only_its_samples(delivery, delta):
    data, booster = delivery
    X = data["test"].X
    patcher = ModelPatcher(booster)
    t, l = _populated_leaf(booster, X)

    base = booster.predict(X)
    cand = patcher.candidate().apply({(t, l): delta})
    new = cand.predict(X)

    in_leaf = leaf_assignments(booster, X)[:, t] == l
    assert in_leaf.any() and (~in_leaf).any()
    np.testing.assert_allclose(new[in_leaf] - base[in_leaf], delta, rtol=0, atol=ATOL)
    np.testing.assert_array_equal(new[~in_leaf], base[~in_leaf])  # untouched rows: bit-identical


def test_every_tree_can_be_edited(delivery):
    """Not just one lucky tree: first, middle and last tree all behave."""
    data, booster = delivery
    X = data["cal"].X
    patcher = ModelPatcher(booster)
    base = booster.predict(X)
    leaves = leaf_assignments(booster, X)
    for t in (0, booster.num_trees() // 2, booster.num_trees() - 1):
        _, l = _populated_leaf(booster, X, tree_id=t)
        new = patcher.candidate().apply({(t, l): 1.0}).predict(X)
        in_leaf = leaves[:, t] == l
        np.testing.assert_allclose(new - base, in_leaf.astype(float), rtol=0, atol=ATOL)


# --------------------------------------------------------------------------- #
# Acceptance 2: y_base + X d equals the edited model's predictions
# --------------------------------------------------------------------------- #
def test_linearity_multi_leaf_edit(delivery):
    data, booster = delivery
    patcher = ModelPatcher(booster)
    rng = np.random.default_rng(0)
    keys = _pick_leaves(patcher, 10, rng)
    deltas = rng.normal(0, 2.0, size=len(keys))
    cand = patcher.candidate().apply({k: d for k, d in zip(keys, deltas)})

    for split in ("cal", "val", "test"):
        X = data[split].X
        A = leaf_design_matrix(booster, X, patcher.index)
        expected = booster.predict(X) + A @ cand.delta_vector()
        np.testing.assert_allclose(cand.predict(X), expected, rtol=0, atol=ATOL)


def test_predictions_equal_design_matrix_times_leaf_values(delivery):
    data, booster = delivery
    patcher = ModelPatcher(booster)
    X = data["test"].X
    A = leaf_design_matrix(booster, X, patcher.index)
    np.testing.assert_allclose(A @ patcher.base_values, booster.predict(X), rtol=0, atol=ATOL)


# --------------------------------------------------------------------------- #
# Acceptance 3: two candidates from the same base share no state
# --------------------------------------------------------------------------- #
def test_candidates_are_independent(delivery):
    data, booster = delivery
    X = data["val"].X
    base_pred = booster.predict(X)
    patcher = ModelPatcher(booster)
    rng = np.random.default_rng(1)
    a, b = patcher.candidate(), patcher.candidate()
    assert a.booster is not b.booster

    a.apply({k: 5.0 for k in _pick_leaves(patcher, 10, rng)})
    np.testing.assert_array_equal(b.predict(X), base_pred)
    np.testing.assert_array_equal(booster.predict(X), base_pred)
    np.testing.assert_array_equal(patcher.base_booster().predict(X), base_pred)

    pred_a = a.predict(X)
    b.apply({k: -5.0 for k in _pick_leaves(patcher, 10, rng)})
    np.testing.assert_array_equal(a.predict(X), pred_a)
    assert b.edit_log != a.edit_log
    assert len(a.edit_log) == len(b.edit_log) == 10


def test_patcher_does_not_track_later_changes_to_source_booster(delivery):
    """The patcher snapshots the base: mutating the original booster later doesn't leak in."""
    data, booster = delivery
    X = data["val"].X
    base_pred = booster.predict(X)
    patcher = ModelPatcher(booster)
    old = booster.get_leaf_output(0, 0)
    try:
        booster.set_leaf_output(0, 0, old + 100.0)
        np.testing.assert_array_equal(patcher.candidate().predict(X), base_pred)
    finally:
        booster.set_leaf_output(0, 0, old)
    np.testing.assert_array_equal(booster.predict(X), base_pred)


# --------------------------------------------------------------------------- #
# Acceptance 4: save -> load round-trips edited values
# --------------------------------------------------------------------------- #
def test_save_load_round_trip(delivery, tmp_path):
    data, booster = delivery
    patcher = ModelPatcher(booster)
    rng = np.random.default_rng(2)
    keys = _pick_leaves(patcher, 10, rng)
    # awkward doubles on purpose: must survive text serialisation exactly
    cand = patcher.candidate().apply({k: float(d) for k, d in zip(keys, rng.normal(0, 1, 10) / 3)})
    cand.save(tmp_path / "cand")

    loaded = load_candidate(tmp_path / "cand", patcher)
    np.testing.assert_array_equal(loaded.leaf_values(), cand.leaf_values())
    np.testing.assert_array_equal(loaded.predict(data["test"].X), cand.predict(data["test"].X))
    assert loaded.edit_log == cand.edit_log
    np.testing.assert_array_equal(loaded.delta_vector(), cand.delta_vector())


# --------------------------------------------------------------------------- #
# Acceptance 5: rollback restores the original predictions exactly
# --------------------------------------------------------------------------- #
def test_rollback_restores_exactly(delivery):
    data, booster = delivery
    X = data["test"].X
    patcher = ModelPatcher(booster)
    base_pred = booster.predict(X)
    rng = np.random.default_rng(3)

    cand = patcher.candidate()
    cand.apply({k: float(d) for k, d in zip(_pick_leaves(patcher, 6, rng), rng.normal(0, 3, 6))})
    after_first = cand.predict(X)
    cand.apply({k: float(d) for k, d in zip(_pick_leaves(patcher, 4, rng), rng.normal(0, 3, 4))})

    cand.rollback(4)
    np.testing.assert_array_equal(cand.predict(X), after_first)
    cand.rollback()
    np.testing.assert_array_equal(cand.predict(X), base_pred)
    np.testing.assert_array_equal(cand.leaf_values(), patcher.base_values)
    assert cand.edit_log == [] and not cand.delta_vector().any()


def test_rollback_of_repeated_edits_to_same_leaf(delivery):
    data, booster = delivery
    X = data["cal"].X
    patcher = ModelPatcher(booster)
    cand = patcher.candidate()
    for d in (0.1, 0.2, -0.7):
        cand.apply({(5, 0): d})
    assert cand.edited_leaves == {(5, 0)}
    cand.rollback()
    np.testing.assert_array_equal(cand.predict(X), booster.predict(X))


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "edits, kwargs, match",
    [
        ({(10**6, 0): 1.0}, {}, "does not exist"),
        ({(0, 99): 1.0}, {}, "does not exist"),
        ({(-1, 0): 1.0}, {}, "does not exist"),
        ([(0, 0, 1.0), (0, 0, 2.0)], {}, "twice"),
        ({(0, 0): float("nan")}, {}, "non-finite"),
        ({(0, 0): float("inf")}, {}, "non-finite"),
        ({(0, 0): 3.0}, {"max_abs_delta": 1.0}, "exceeds bound"),
    ],
)
def test_invalid_edits_rejected(delivery, edits, kwargs, match):
    _, booster = delivery
    with pytest.raises(EditError, match=match):
        ModelPatcher(booster).candidate().apply(edits, **kwargs)


def test_failed_apply_is_atomic(delivery):
    data, booster = delivery
    X = data["cal"].X
    cand = ModelPatcher(booster).candidate()
    with pytest.raises(EditError):
        cand.apply([(0, 0, 1.0), (1, 1, 1.0), (0, 99, 1.0)])  # last one invalid
    assert cand.edit_log == []
    np.testing.assert_array_equal(cand.predict(X), booster.predict(X))


def test_leaf_budget_counts_distinct_leaves_cumulatively(delivery):
    _, booster = delivery
    patcher = ModelPatcher(booster)
    keys = _pick_leaves(patcher, 11, np.random.default_rng(4))
    cand = patcher.candidate().apply({k: 0.1 for k in keys[:6]}, max_leaves=10)
    cand.apply({k: 0.1 for k in keys[:6]}, max_leaves=10)  # re-editing same leaves is fine
    cand.apply({k: 0.1 for k in keys[6:10]}, max_leaves=10)
    with pytest.raises(EditError, match="budget"):
        cand.apply({keys[10]: 0.1}, max_leaves=10)
    assert len(cand.edited_leaves) == 10
