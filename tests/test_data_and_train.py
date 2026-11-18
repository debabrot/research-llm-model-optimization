import numpy as np
import pytest

from tree_surgery.datasets import (
    anon_mapping,
    delivery_noise_sigma,
    delivery_true_function,
    generate_delivery,
    load_dataset,
    split_indices,
)
from tree_surgery.design_matrix import LeafIndex
from tree_surgery.train import fit_base_model, load_model


# ---- splits ---------------------------------------------------------------- #
@pytest.mark.parametrize("n", [20_000, 20_640, 101])
def test_splits_partition_rows(n):
    idx = split_indices(n, seed=0)
    allidx = np.concatenate(list(idx.values()))
    assert len(allidx) == n and len(np.unique(allidx)) == n
    for name, frac in {"train": 0.5, "cal": 0.15, "val": 0.15, "test": 0.2}.items():
        assert abs(len(idx[name]) - frac * n) <= 2


def test_splits_deterministic_and_seed_dependent():
    a, b, c = split_indices(1000, 0), split_indices(1000, 0), split_indices(1000, 1)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["test"], c["test"])


def test_bad_fractions():
    with pytest.raises(ValueError):
        split_indices(100, 0, {"train": 0.5, "cal": 0.5, "val": 0.1, "test": 0.1})


# ---- D1 -------------------------------------------------------------------- #
def test_delivery_deterministic_and_seeded():
    a, b, c = generate_delivery(2000, 0), generate_delivery(2000, 0), generate_delivery(2000, 1)
    assert a.X.equals(b.X) and np.array_equal(a.y, b.y)
    assert not np.array_equal(a.y, c.y)


def test_delivery_truth_and_snr():
    ds = generate_delivery(50_000, seed=3)
    np.testing.assert_allclose(ds.f_true, delivery_true_function(ds.X))
    rms_sigma = np.sqrt(np.mean(delivery_noise_sigma(ds.X) ** 2))
    snr = ds.f_true.std() / rms_sigma
    assert 2.7 < snr < 3.3, snr
    resid = ds.y - ds.f_true
    assert abs(resid.mean()) < 0.1


def test_delivery_split_carries_truth():
    data = load_dataset("delivery", seed=0)
    for s in data.splits.values():
        assert s.f_true is not None and len(s.f_true) == len(s.y) == len(s.X)
    assert data.target_name == "delivery_time_min"


# ---- D2 -------------------------------------------------------------------- #
@pytest.mark.network
def test_california_loads():
    try:
        data = load_dataset("california", seed=0)
    except Exception as exc:  # offline / blocked download
        pytest.skip(f"California Housing unavailable: {exc}")
    assert sum(len(s.y) for s in data.splits.values()) == 20_640
    assert data.feature_names[0] == "MedInc"
    assert data["train"].f_true is None


# ---- anonymisation --------------------------------------------------------- #
def test_anon_mapping():
    names = ["a", "b", "c", "d", "e", "f", "g", "h"]
    m0 = anon_mapping(names, 0)
    assert set(m0) == set(names) and set(m0.values()) == {f"f{i}" for i in range(8)}
    assert m0 == anon_mapping(names, 0)
    assert any(anon_mapping(names, s) != m0 for s in range(1, 5))
    assert any(m0[n] != f"f{i}" for i, n in enumerate(names)), "order should be shuffled"


# ---- base model ------------------------------------------------------------ #
def test_training_is_deterministic():
    data = load_dataset("delivery", seed=2, n_rows=4000)
    b1, _ = fit_base_model(data)
    b2, _ = fit_base_model(data)
    assert b1.model_to_string() == b2.model_to_string()


def test_base_model_truncated_to_best_iteration(delivery):
    data, booster = delivery
    assert 1 < booster.num_trees() < 1000
    assert len(LeafIndex.from_booster(booster)) == sum(
        t["num_leaves"] for t in booster.dump_model()["tree_info"]
    )


def test_base_model_beats_mean_and_saves(tmp_path):
    data = load_dataset("delivery", seed=0)
    booster, info = fit_base_model(data, out_dir=tmp_path / "m")
    assert info.metrics["test"]["r2"] > 0.8
    assert info.metrics["test"]["rmse_vs_truth"] < info.metrics["test"]["rmse"]
    re = load_model(tmp_path / "m")
    np.testing.assert_array_equal(re.predict(data["test"].X), booster.predict(data["test"].X))
    saved = np.load(tmp_path / "m" / "predictions.npz")
    np.testing.assert_array_equal(saved["pred_test"], booster.predict(data["test"].X))
    np.testing.assert_array_equal(saved["idx_test"], data["test"].idx)
