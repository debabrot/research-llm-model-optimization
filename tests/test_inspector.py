import re

import numpy as np
import pytest

from tree_surgery.datasets import anon_mapping
from tree_surgery.design_matrix import leaf_assignments
from tree_surgery.inspector import extract_rules, leaf_table, render_leaf_line, render_leaf_table, sig


def _check_rule_fidelity(booster, X):
    """Every rule's mask must equal LightGBM's own leaf assignment, sample by sample."""
    rules = extract_rules(booster)
    leaves = leaf_assignments(booster, X)
    for (t, l), rule in rules.items():
        np.testing.assert_array_equal(rule.mask(X), leaves[:, t] == l, err_msg=f"T{t}/L{l}")
    return rules


def test_rules_reproduce_leaf_assignment(delivery):
    data, booster = delivery
    rules = _check_rule_fidelity(booster, data["cal"].X)
    assert len(rules) == sum(t["num_leaves"] for t in booster.dump_model()["tree_info"])


def test_rules_with_missing_values(nan_model):
    X, _, booster = nan_model
    rules = _check_rule_fidelity(booster, X)
    texts = [r.text() for r in rules.values()]
    assert any("missing" in s for s in texts), "NaN-trained model should render missing directions"


def test_leaf_table_stats(delivery):
    data, booster = delivery
    cal = data["cal"]
    tab = leaf_table(booster, cal.X, cal.y)

    # each tree partitions the calibration set
    assert (tab.groupby("tree_id").n_cal.sum() == len(cal.y)).all()
    assert (tab.groupby("tree_id").n_train.sum() == len(data["train"].y)).all()

    # spot-check residual stats against a direct computation
    resid = cal.y - booster.predict(cal.X)
    leaves = leaf_assignments(booster, cal.X)
    row = tab.sort_values("n_cal", ascending=False).iloc[3]
    m = leaves[:, int(row.tree_id)] == int(row.leaf_id)
    assert row.n_cal == m.sum()
    assert row.mean_resid == pytest.approx(resid[m].mean(), abs=1e-10)
    assert row.std_resid == pytest.approx(resid[m].std(), abs=1e-8)

    empty = tab[tab.n_cal == 0]
    assert empty.mean_resid.isna().all() and empty.std_resid.isna().all()


def test_render_line_format(delivery):
    data, booster = delivery
    tab = leaf_table(booster, data["cal"].X, data["cal"].y)
    pattern = re.compile(
        r"^T\d+/L\d+ \| rule: .+ \| n_cal=\d+ \| value=-?[\d.]+ \| mean_resid=([+-][\d.]+|nan) \| std_resid=([\d.]+|nan)$"
    )
    lines = render_leaf_table(tab)
    assert len(lines) == len(tab)
    bad = [s for s in lines if not pattern.match(s)]
    assert not bad, bad[:3]


def test_anon_rendering_hides_feature_names(delivery):
    data, booster = delivery
    tab = leaf_table(booster, data["cal"].X, data["cal"].y)
    rules = extract_rules(booster)
    amap = anon_mapping(data.feature_names, data.seed)
    lines = render_leaf_table(tab, rules=rules, name_map=amap)
    joined = "\n".join(lines)
    for name in data.feature_names:
        assert name not in joined
    assert re.search(r"\bf\d+\b", joined)


def test_render_requires_rules_for_renaming(delivery):
    data, booster = delivery
    tab = leaf_table(booster, data["cal"].X, data["cal"].y).head(3)
    with pytest.raises(ValueError):
        render_leaf_table(tab, name_map={"distance_km": "f0"})


@pytest.mark.parametrize(
    "x, kw, expected",
    [
        (0.021234, {}, "0.0212"),
        (143.0, {}, "143"),
        (1234.5, {}, "1230"),
        (0.84, {"sign": True}, "+0.84"),
        (-0.84, {"sign": True}, "-0.84"),
        (6.265000000000001, {}, "6.27"),
        (-0.0, {}, "0"),
        (1.0000000180025095e-35, {}, "0"),  # LightGBM kZeroThreshold
        (0.000145, {}, "0.000145"),
        (float("nan"), {}, "nan"),
    ],
)
def test_sig(x, kw, expected):
    assert sig(x, **kw) == expected


def test_render_leaf_line_matches_plan_example():
    row = dict(tree_id=3, leaf_id=7, n_cal=143, value=0.021, mean_resid=0.84, std_resid=1.9)
    line = render_leaf_line(row, rule_text="distance_km > 12.3 AND rain_mm <= 4.0")
    assert line == (
        "T3/L7 | rule: distance_km > 12.3 AND rain_mm <= 4.0 | n_cal=143 | "
        "value=0.021 | mean_resid=+0.84 | std_resid=1.9"
    )
