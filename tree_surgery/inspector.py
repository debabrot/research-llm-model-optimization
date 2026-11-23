"""Leaf inspection: decision paths, readable rules and calibration residual stats (PLAN §5).

Each leaf is a conjunction of split conditions. ``LeafRule.conditions`` keeps the
full raw path (exact thresholds, missing-value direction); ``LeafRule.text()``
merges repeated conditions on one feature into an interval and renders it with
optional feature renaming (for the `anon` ablation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from .design_matrix import LeafIndex, leaf_design_matrix, leaf_values


# --------------------------------------------------------------------------- #
# Number formatting
# --------------------------------------------------------------------------- #
def sig(x: float, digits: int = 3, sign: bool = False) -> str:
    """Round to ``digits`` significant figures, positional notation, no trailing junk."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    if abs(float(x)) < 1e-12:
        # LightGBM encodes "split at zero" as kZeroThreshold = 1e-35; render it as 0.
        x = 0.0
    s = np.format_float_positional(float(x), precision=digits, unique=False, fractional=False, trim="-")
    if s in ("-0", "-0."):
        s = "0"
    if sign and not s.startswith("-"):
        s = "+" + s
    return s


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Condition:
    """One step on the path from root to leaf.

    For numerical splits LightGBM sends ``x <= threshold`` left. ``went_left``
    records which side this path took. ``missing_follows`` is True when a missing
    value would also take this side (only meaningful if ``missing_type != 'None'``).
    """

    feature: int
    feature_name: str
    decision_type: str          # "<=" (numerical) or "==" (categorical)
    threshold: Any              # float, or "a||b||c" for categorical
    went_left: bool
    missing_type: str           # "None" | "NaN" | "Zero"
    missing_follows: bool

    def holds(self, col: np.ndarray) -> np.ndarray:
        """Vectorised: does this condition hold for each value in ``col``?

        Handles missing values the way LightGBM does for the missing types we can meet.
        """
        col = np.asarray(col, dtype=np.float64)
        if self.decision_type == "<=":
            x = col
            is_missing = np.zeros(len(col), dtype=bool)
            if self.missing_type == "NaN":
                is_missing = np.isnan(col)
            elif self.missing_type == "Zero":
                is_missing = np.isnan(col) | (col == 0)
            else:  # "None": LightGBM maps NaN to 0.0
                x = np.where(np.isnan(col), 0.0, col)
            left = x <= float(self.threshold)
            side = left if self.went_left else ~left
            return np.where(is_missing, self.missing_follows, side)
        cats = {int(c) for c in str(self.threshold).split("||")}
        left = np.isin(col, list(cats))
        return left if self.went_left else ~left


@dataclass
class LeafRule:
    tree_id: int
    leaf_id: int
    conditions: list[Condition] = field(default_factory=list)

    @property
    def depth(self) -> int:
        return len(self.conditions)

    def mask(self, X: pd.DataFrame) -> np.ndarray:
        """Rows of X that satisfy every condition (i.e. land in this leaf)."""
        m = np.ones(len(X), dtype=bool)
        for c in self.conditions:
            m &= c.holds(X.iloc[:, c.feature].to_numpy())
        return m

    def text(self, name_map: Optional[dict[str, str]] = None, digits: int = 3) -> str:
        """Readable rule, e.g. ``distance_km > 12.3 AND 2.5 < rain_mm <= 4``."""
        if not self.conditions:
            return "(all samples)"
        rename = (lambda n: name_map.get(n, n)) if name_map else (lambda n: n)

        order: list[str] = []
        lo: dict[str, float] = {}
        hi: dict[str, float] = {}
        cat_in: dict[str, set[int]] = {}
        cat_out: dict[str, set[int]] = {}
        miss_ok: dict[str, bool] = {}
        has_missing: dict[str, bool] = {}

        for c in self.conditions:
            n = c.feature_name
            if n not in order:
                order.append(n)
            if c.missing_type != "None":
                has_missing[n] = True
                miss_ok[n] = miss_ok.get(n, True) and c.missing_follows
            if c.decision_type == "<=":
                t = float(c.threshold)
                if c.went_left:
                    hi[n] = min(hi.get(n, np.inf), t)
                else:
                    lo[n] = max(lo.get(n, -np.inf), t)
            else:
                cats = {int(v) for v in str(c.threshold).split("||")}
                if c.went_left:
                    cat_in[n] = cats if n not in cat_in else cat_in[n] & cats
                else:
                    cat_out[n] = cat_out.get(n, set()) | cats

        parts = []
        for n in order:
            name = rename(n)
            if n in cat_in or n in cat_out:
                allowed = cat_in.get(n)
                if allowed is not None:
                    allowed = allowed - cat_out.get(n, set())
                    s = f"{name} in {{{', '.join(map(str, sorted(allowed)))}}}"
                else:
                    s = f"{name} not in {{{', '.join(map(str, sorted(cat_out[n])))}}}"
            else:
                a, b = lo.get(n), hi.get(n)
                if a is not None and b is not None:
                    s = f"{sig(a, digits)} < {name} <= {sig(b, digits)}"
                elif a is not None:
                    s = f"{name} > {sig(a, digits)}"
                else:
                    s = f"{name} <= {sig(b, digits)}"
            if has_missing.get(n):
                s = f"({s} OR {name} missing)" if miss_ok.get(n) else f"{s} (missing excluded)"
            parts.append(s)
        return " AND ".join(parts)


def extract_rules(booster: lgb.Booster) -> dict[tuple[int, int], LeafRule]:
    """Decision path for every leaf, keyed by (tree_id, leaf_id)."""
    dump = booster.dump_model()
    names = dump["feature_names"]
    rules: dict[tuple[int, int], LeafRule] = {}

    for t, tree in enumerate(dump["tree_info"]):
        stack: list[tuple[dict, list[Condition]]] = [(tree["tree_structure"], [])]
        while stack:
            node, path = stack.pop()
            if "split_index" not in node:  # leaf (a single-leaf tree has no leaf_index)
                leaf_id = int(node.get("leaf_index", 0))
                rules[(t, leaf_id)] = LeafRule(t, leaf_id, path)
                continue
            f = int(node["split_feature"])
            common = dict(
                feature=f,
                feature_name=names[f],
                decision_type=node["decision_type"],
                threshold=node["threshold"],
                missing_type=node.get("missing_type", "None"),
            )
            default_left = bool(node.get("default_left", True))
            stack.append(
                (node["left_child"], path + [Condition(**common, went_left=True, missing_follows=default_left)])
            )
            stack.append(
                (node["right_child"], path + [Condition(**common, went_left=False, missing_follows=not default_left)])
            )
    return rules


def leaf_train_counts(booster: lgb.Booster) -> dict[tuple[int, int], int]:
    """Training-time sample count per leaf, as recorded by LightGBM."""
    out = {}
    for t, tree in enumerate(booster.dump_model()["tree_info"]):
        stack = [tree["tree_structure"]]
        while stack:
            node = stack.pop()
            if "split_index" in node:
                stack += [node["left_child"], node["right_child"]]
            else:
                out[(t, int(node.get("leaf_index", 0)))] = int(node.get("leaf_count", 0))
    return out


# --------------------------------------------------------------------------- #
# Leaf table
# --------------------------------------------------------------------------- #
def leaf_table(
    booster: lgb.Booster,
    X_cal: pd.DataFrame,
    y_cal: np.ndarray,
    index: Optional[LeafIndex] = None,
) -> pd.DataFrame:
    """One row per leaf: rule, value, train count and calibration residual stats.

    Residual = y - y_hat_base on the calibration split, so a positive mean residual
    means the model under-predicts in that leaf and a positive delta would help.
    std_resid is the population std (ddof=0); both stats are NaN when n_cal == 0.
    """
    index = index or LeafIndex.from_booster(booster)
    rules = extract_rules(booster)
    n_train = leaf_train_counts(booster)
    values = leaf_values(booster, index)

    X = leaf_design_matrix(booster, X_cal, index)
    resid = np.asarray(y_cal, dtype=np.float64) - booster.predict(X_cal)
    n_cal = np.asarray(X.sum(axis=0)).ravel()
    s1 = X.T @ resid
    s2 = X.T @ (resid**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n_cal > 0, s1 / n_cal, np.nan)
        var = np.where(n_cal > 0, s2 / n_cal - mean**2, np.nan)
    std = np.sqrt(np.clip(var, 0.0, None))

    keys = index.keys()
    # Note: rule objects are deliberately NOT stored in df.attrs: pandas deep-copies
    # attrs on every derived frame/row, which makes iterrows() quadratic-slow.
    return pd.DataFrame(
        {
            "tree_id": [k[0] for k in keys],
            "leaf_id": [k[1] for k in keys],
            "depth": [rules[k].depth for k in keys],
            "rule": [rules[k].text() for k in keys],
            "value": values,
            "n_train": [n_train[k] for k in keys],
            "n_cal": n_cal.astype(int),
            "mean_resid": mean,
            "std_resid": std,
        }
    )


def render_leaf_line(row: pd.Series | dict, rule_text: Optional[str] = None, digits: int = 3) -> str:
    """PLAN §5 line format:
    ``T{tree}/L{leaf} | rule: ... | n_cal=143 | value=0.021 | mean_resid=+0.84 | std_resid=1.9``
    """
    r = dict(row)
    rule = rule_text if rule_text is not None else r["rule"]
    return (
        f"T{int(r['tree_id'])}/L{int(r['leaf_id'])} | rule: {rule} | n_cal={int(r['n_cal'])} | "
        f"value={sig(r['value'], digits)} | mean_resid={sig(r['mean_resid'], digits, sign=True)} | "
        f"std_resid={sig(r['std_resid'], digits)}"
    )


def render_leaf_table(
    table: pd.DataFrame,
    rules: Optional[dict[tuple[int, int], LeafRule]] = None,
    name_map: Optional[dict[str, str]] = None,
    digits: int = 3,
) -> list[str]:
    """Render every leaf as one line.

    Pass ``rules`` (from ``extract_rules``) together with ``name_map`` to re-render
    rules under different feature names (anon ablation) or precision.
    """
    if (name_map is not None or digits != 3) and rules is None:
        raise ValueError("pass rules=extract_rules(booster) to re-render with name_map/digits")
    lines = []
    for row in table.to_dict("records"):
        text = None
        if rules is not None and (name_map is not None or digits != 3):
            text = rules[(int(row["tree_id"]), int(row["leaf_id"]))].text(name_map, digits)
        lines.append(render_leaf_line(row, text, digits))
    return lines
