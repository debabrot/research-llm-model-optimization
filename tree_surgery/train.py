"""Base model training, save and load (PLAN §3).

LightGBM regression trained on the train split only, early-stopped on validation.
The returned booster is truncated to ``best_iteration`` so its trees are exactly the
trees that make predictions: no dormant trees for an editor to touch.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import lightgbm as lgb
import numpy as np

from .datasets import SplitData
from .evaluation import regression_metrics

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 15,
    "learning_rate": 0.05,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 1,
    "verbose": -1,
}


@dataclass
class BaseModelInfo:
    dataset: str
    seed: int
    best_iteration: int
    num_trees: int
    num_leaves_total: int
    params: dict[str, Any]
    metrics: dict[str, dict[str, float]]  # split -> metric -> value


def train_base_model(
    data: SplitData,
    params: Optional[dict[str, Any]] = None,
    num_boost_round: int = 5000,
    early_stopping_rounds: int = 50,
) -> lgb.Booster:
    p = {**DEFAULT_PARAMS, **(params or {}), "seed": data.seed}
    tr, va = data["train"], data["val"]
    dtrain = lgb.Dataset(tr.X, label=tr.y, free_raw_data=False)
    dval = lgb.Dataset(va.X, label=va.y, reference=dtrain, free_raw_data=False)
    booster = lgb.train(
        p,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    best = booster.best_iteration or booster.current_iteration()
    # Rebuild from text with only the best trees. Leaf values are written at full
    # double precision, so this is exact.
    return lgb.Booster(model_str=booster.model_to_string(num_iteration=best))


def num_leaves_per_tree(booster: lgb.Booster) -> list[int]:
    return [t["num_leaves"] for t in booster.dump_model()["tree_info"]]


def describe_base_model(booster: lgb.Booster, data: SplitData, params: dict[str, Any]) -> BaseModelInfo:
    leaves = num_leaves_per_tree(booster)
    metrics = {}
    for name, split in data.splits.items():
        pred = booster.predict(split.X)
        m = regression_metrics(split.y, pred)
        if split.f_true is not None:
            m["rmse_vs_truth"] = float(np.sqrt(np.mean((pred - split.f_true) ** 2)))
        metrics[name] = m
    return BaseModelInfo(
        dataset=data.name,
        seed=data.seed,
        best_iteration=booster.num_trees(),
        num_trees=booster.num_trees(),
        num_leaves_total=int(sum(leaves)),
        params={**DEFAULT_PARAMS, **params, "seed": data.seed},
        metrics=metrics,
    )


def save_base_model(booster: lgb.Booster, info: BaseModelInfo, data: SplitData, out_dir: Path | str) -> Path:
    """Save model text, per-split predictions and metrics to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / "model.txt"))
    preds = {f"pred_{k}": booster.predict(s.X) for k, s in data.splits.items()}
    preds |= {f"idx_{k}": s.idx for k, s in data.splits.items()}
    np.savez_compressed(out / "predictions.npz", **preds)
    (out / "info.json").write_text(json.dumps(asdict(info), indent=2))
    return out


def load_model(path: Path | str) -> lgb.Booster:
    path = Path(path)
    if path.is_dir():
        path = path / "model.txt"
    return lgb.Booster(model_file=str(path))


def fit_base_model(
    data: SplitData, cfg: Optional[dict[str, Any]] = None, out_dir: Path | str | None = None
) -> tuple[lgb.Booster, BaseModelInfo]:
    """Train, describe and (optionally) save the base model for one (dataset, seed).

    ``cfg`` is the ``base_model`` block of ``configs/experiment.yaml``.
    """
    cfg = cfg or {}
    params = cfg.get("params", {})
    booster = train_base_model(
        data,
        params=params,
        num_boost_round=cfg.get("num_boost_round", 5000),
        early_stopping_rounds=cfg.get("early_stopping_rounds", 50),
    )
    info = describe_base_model(booster, data, params)
    if out_dir is not None:
        save_base_model(booster, info, data, out_dir)
    return booster, info
