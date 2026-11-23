"""Experiment grid runner (PLAN §4, §8, §11).

For each (dataset, seed): load + split, train the base model on train (early-stopped
on val), build the arm context (train / cal / val only), run every requested arm,
then score each arm's final model on all four splits. Test is touched here and only
here, after every selection decision has been made.

Outputs, under ``--out`` (default ``results/phase2``):

* ``results.csv``     one row per (dataset, seed, arm[, variant, run])
* ``candidates.jsonl`` every evaluated candidate (calibration / validation scores only)
* ``summary.md``      per-arm means and the paired comparisons

    python -m tree_surgery.runner                                  # full non-LLM grid
    python -m tree_surgery.runner --datasets delivery --seeds 0 1 --arms original sparse_opt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .arms import NON_LLM_ARMS, ArmOutput, build_context, get_arm
from .config import REPO_ROOT, load_config
from .datasets import SplitData, load_dataset
from .evaluation import compare_arms, regression_metrics, rmse
from .train import fit_base_model

SPLITS = ("train", "cal", "val", "test")
EDIT_ARMS = ("random_opt", "sparse_opt")
CEILING_ARMS = ("refit", "retrain")

# Comparisons reported in the summary: (a, b). Negative mean_diff = a is better.
DEFAULT_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("random_opt", "original"),
    ("sparse_opt", "original"),
    ("sparse_opt", "random_opt"),
    ("refit", "original"),
    ("retrain", "original"),
    ("sparse_opt", "retrain"),
)


def score_output(out: ArmOutput, data: SplitData) -> dict[str, Any]:
    """Metrics of an arm's final model on every split (+ RMSE vs the noise-free truth on D1)."""
    row: dict[str, Any] = {}
    for name in SPLITS:
        s = data[name]
        pred = out.model.predict(s.X)
        for k, v in regression_metrics(s.y, pred).items():
            row[f"{name}_{k}"] = v
        if s.f_true is not None:
            row[f"{name}_rmse_vs_truth"] = rmse(s.f_true, pred)
    row["gen_gap"] = row["test_rmse"] - row["cal_rmse"]
    return row


def run_cell(
    dataset: str,
    seed: int,
    arms: Sequence[str],
    cfg: dict[str, Any],
    artifacts_dir: Optional[Path] = None,
    n_rows: Optional[int] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run all ``arms`` for one (dataset, seed). Returns (result rows, candidate records)."""
    t0 = time.perf_counter()
    data = load_dataset(dataset, seed=seed, n_rows=n_rows)
    out_dir = None if artifacts_dir is None else Path(artifacts_dir) / dataset / f"seed{seed}"
    base, info = fit_base_model(data, cfg.get("base_model"), out_dir=out_dir)
    base_s = time.perf_counter() - t0
    ctx = build_context(data, base, cfg)

    rows, cands = [], []
    for arm_name in arms:
        out = get_arm(arm_name).run(ctx)
        row = {
            "dataset": dataset,
            "seed": seed,
            "arm": out.arm,
            "variant": "-",
            "model": "-",
            "run": 0,
            **score_output(out, data),
            "leaves_edited": out.leaves_edited,
            "frac_params_changed": out.frac_params_changed,
            "within_budget": out.within_budget,
            "n_candidates": len(out.candidates),
            "chosen_candidate": out.chosen,
            "lam": out.info.get("lam"),
            "decay_rate": out.info.get("decay_rate"),
            "base_num_trees": info.num_trees,
            "base_num_leaves": info.num_leaves_total,
            "base_hit_round_cap": info.num_trees >= cfg.get("base_model", {}).get("num_boost_round", 5000),
            "retrain_num_trees": out.info.get("num_trees"),
            "runtime_s": out.runtime_s,
            "base_train_s": base_s,
            "llm_tokens_in": 0,
            "llm_tokens_out": 0,
            "llm_cost_usd": 0.0,
        }
        rows.append(row)
        for c in out.candidates:
            cands.append({"dataset": dataset, "seed": seed, "arm": out.arm, "chosen": c.index == out.chosen,
                          **c.to_dict()})
    return rows, cands


def run_grid(
    datasets: Iterable[str],
    seeds: Iterable[int],
    arms: Sequence[str] = NON_LLM_ARMS,
    cfg: Optional[dict[str, Any]] = None,
    out_dir: Path | str | None = None,
    artifacts_dir: Path | str | None = None,
    n_rows: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    cfg = cfg or load_config()
    all_rows, all_cands = [], []
    for ds in datasets:
        for seed in seeds:
            t = time.perf_counter()
            rows, cands = run_cell(ds, seed, arms, cfg, None if artifacts_dir is None else Path(artifacts_dir),
                                   n_rows=n_rows)
            all_rows += rows
            all_cands += cands
            if verbose:
                best = ", ".join(f"{r['arm']}={r['test_rmse']:.4f}" for r in rows)
                print(f"[{ds} seed {seed}] {time.perf_counter() - t:5.1f}s  test RMSE: {best}", flush=True)
    df = pd.DataFrame(all_rows)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "results.csv", index=False)
        with open(out / "candidates.jsonl", "w") as fh:
            for c in all_cands:
                fh.write(json.dumps(c, default=float) + "\n")
        (out / "summary.md").write_text(summarize(df))
    return df


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def _fmt(x: Any, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{nd}f}" if isinstance(x, (float, np.floating)) else str(x)


def arm_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = {"test_rmse": ["mean", "std"], "val_rmse": "mean", "cal_rmse": "mean", "gen_gap": "mean",
            "leaves_edited": "mean", "runtime_s": "mean"}
    if "test_rmse_vs_truth" in df:
        cols["test_rmse_vs_truth"] = "mean"
    t = df.groupby(["dataset", "arm"], sort=False).agg(cols)
    t.columns = ["_".join(c) if c[1] != "mean" else c[0] for c in t.columns]
    return t.reset_index()


def summarize(
    df: pd.DataFrame, comparisons: Sequence[tuple[str, str]] = DEFAULT_COMPARISONS
) -> str:
    lines = ["# Results summary", "", f"{df['seed'].nunique()} seeds per dataset. Lower RMSE is better.", ""]
    t = arm_table(df)
    has_truth = "test_rmse_vs_truth" in t
    hdr = "| dataset | arm | test RMSE (mean ± sd) | val RMSE | cal RMSE | gap (test−cal) |"
    hdr += " test RMSE vs truth |" if has_truth else ""
    hdr += " leaves edited | runtime s |"
    lines += ["## Per-arm means", "", hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for _, r in t.iterrows():
        line = (f"| {r.dataset} | {r.arm} | {r.test_rmse:.4f} ± {r.test_rmse_std:.4f} | {r.val_rmse:.4f} | "
                f"{r.cal_rmse:.4f} | {r.gen_gap:+.4f} |")
        if has_truth:
            line += f" {_fmt(r.test_rmse_vs_truth)} |"
        lines.append(line + f" {_fmt(r.leaves_edited, 1)} | {r.runtime_s:.2f} |")

    present = set(df["arm"])
    comps = [c for c in comparisons if set(c) <= present]
    if comps:
        lines += ["", "## Paired differences in test RMSE (a − b; negative = a better)", "",
                  "| dataset | a − b | mean diff | rel. % | 95% bootstrap CI | CI excl. 0 | a wins | Wilcoxon p |",
                  "|---|---|---|---|---|---|---|---|"]
        for a, b in comps:
            for _, r in compare_arms(df, a, b).iterrows():
                lines.append(
                    f"| {r.dataset} | {a} − {b} | {r.mean_diff:+.4f} | {r.mean_rel_diff_pct:+.2f}% | "
                    f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] | {'yes' if r.ci_excludes_0 else 'no'} | "
                    f"{r.a_wins}/{r.n} | {_fmt(r.wilcoxon_p, 3)} |"
                )
    if "base_hit_round_cap" in df:
        capped = df.drop_duplicates(["dataset", "seed"]).groupby("dataset")["base_hit_round_cap"].sum()
        trees = df.drop_duplicates(["dataset", "seed"]).groupby("dataset")["base_num_trees"].agg(["min", "max"])
        lines += ["", "## Base models", ""]
        for ds in trees.index:
            lines.append(f"- {ds}: {trees.loc[ds, 'min']}–{trees.loc[ds, 'max']} trees; "
                         f"seeds hitting the round cap: {int(capped.loc[ds])}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(cfg["datasets"]))
    ap.add_argument("--seeds", nargs="+", type=int, default=cfg["seeds"])
    ap.add_argument("--arms", nargs="+", default=list(NON_LLM_ARMS))
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "phase2"))
    ap.add_argument("--artifacts", default=str(REPO_ROOT / "artifacts" / "base_models"),
                    help="where base models are saved (git-ignored); '' to skip")
    args = ap.parse_args(argv)
    df = run_grid(args.datasets, args.seeds, args.arms, cfg, args.out, args.artifacts or None)
    print()
    print(summarize(df))


if __name__ == "__main__":
    main()
