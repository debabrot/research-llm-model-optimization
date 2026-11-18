"""Phase 1 in one command: train a model, open a leaf, edit it, check linearity, roll back.

    python scripts/quickstart.py            # D1 delivery data, seed 0
    python scripts/quickstart.py --seed 3
    python scripts/quickstart.py --dataset california   # downloads ~400 KB once
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tree_surgery as ts  # noqa: E402
from tree_surgery.config import load_config  # noqa: E402
from tree_surgery.design_matrix import leaf_assignments  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="delivery", choices=["delivery", "california"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config()

    print(f"1. Loading {args.dataset} (seed {args.seed}) and training the base model ...")
    data = ts.load_dataset(args.dataset, seed=args.seed)
    booster, info = ts.fit_base_model(data, cfg["base_model"])
    m = info.metrics
    truth = f", vs truth {m['test']['rmse_vs_truth']:.3f}" if "rmse_vs_truth" in m["test"] else ""
    print(f"   {info.num_trees} trees, {info.num_leaves_total} leaves; test RMSE {m['test']['rmse']:.3f}{truth}")

    print("\n2. The five calibration leaves with the largest total residual:")
    patcher = ts.ModelPatcher(booster)
    cal = data["cal"]
    table = ts.leaf_table(booster, cal.X, cal.y, patcher.index)
    table["abs_total_resid"] = (table.n_cal * table.mean_resid).abs()
    top = table.sort_values("abs_total_resid", ascending=False).head(5)
    for line in ts.render_leaf_table(top):
        print("   " + line)

    t, l = int(top.iloc[0].tree_id), int(top.iloc[0].leaf_id)
    delta = 1.0
    print(f"\n3. Editing T{t}/L{l} by {delta:+g} ...")
    X = data["test"].X
    base = booster.predict(X)
    cand = patcher.candidate().apply({(t, l): delta})
    change = cand.predict(X) - base
    in_leaf = leaf_assignments(booster, X)[:, t] == l
    print(f"   {in_leaf.sum()} test rows in the leaf moved by {change[in_leaf].min():.9f} .. {change[in_leaf].max():.9f}")
    print(f"   {(~in_leaf).sum()} other rows moved by at most {np.abs(change[~in_leaf]).max():.1e}")

    A = ts.leaf_design_matrix(booster, X, patcher.index)
    err = np.abs(base + A @ cand.delta_vector() - cand.predict(X)).max()
    print(f"\n4. Linearity check: max |y_base + X·d - edited prediction| = {err:.1e}")

    cand.rollback()
    print(f"5. Rolled back; predictions identical to base: {np.array_equal(cand.predict(X), base)}")


if __name__ == "__main__":
    main()
