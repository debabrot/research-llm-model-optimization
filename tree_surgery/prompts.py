"""What the LLM sees: static block, query tool, history block, output schemas (PLAN §5).

* **Static block** (system message, identical for every call within one
  dataset / seed / variant / approach, so providers can prompt-cache it): task and
  constraints, dataset description, base-model summary, leaf-table columns, output
  format. It contains summary counts only, never a list of leaves.
* **query_leaves tool**: the only way the LLM sees leaves. It filters and sorts the
  full leaf table and returns at most ``rows_per_query`` lines. No pre-filtered or
  residual-ranked shortlist is ever sent unasked (PLAN changelog 2026-11-17).
* **History block** (user message): previous candidates in this run with their
  *calibration* RMSE. Validation and test scores are never shown.

``named`` and ``anon`` share the exact same instructions. They differ only in the
dataset block and in feature names inside rules (``anon``: ``f0..fN`` in a per-seed
shuffled order, generic target name, no units or descriptions).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

from .datasets import SplitData, anon_mapping
from .design_matrix import LeafIndex
from .inspector import extract_rules, leaf_table, render_leaf_line, sig

Approach = Literal["select", "direct"]
Variant = Literal["named", "anon"]

SORTABLE = ("n_cal", "mean_resid", "abs_mean_resid", "std_resid", "value", "n_train", "depth", "tree_id")


# --------------------------------------------------------------------------- #
# Leaf view per variant
# --------------------------------------------------------------------------- #
@dataclass
class LeafView:
    """The full leaf table rendered for one variant. The query tool reads from this."""

    table: pd.DataFrame            # inspector.leaf_table + rendered rule text + abs_mean_resid
    variant: str
    name_map: dict[str, str]       # real -> shown feature name (identity for 'named')
    features_shown: list[str]      # features in display order

    @classmethod
    def build(cls, booster, data: SplitData, variant: Variant, index: Optional[LeafIndex] = None) -> "LeafView":
        cal = data["cal"]
        t = leaf_table(booster, cal.X, cal.y, index)
        names = data.feature_names
        if variant == "named":
            name_map = {n: n for n in names}
            shown = list(names)
        elif variant == "anon":
            name_map = anon_mapping(names, data.seed)
            shown = sorted(name_map.values(), key=lambda s: int(s[1:]))
            rules = extract_rules(booster)
            t["rule"] = [rules[(int(a), int(b))].text(name_map) for a, b in zip(t.tree_id, t.leaf_id)]
        else:
            raise ValueError(f"unknown variant {variant!r}")
        t["abs_mean_resid"] = t["mean_resid"].abs()
        return cls(t, variant, name_map, shown)

    def line(self, row: dict) -> str:
        return render_leaf_line(row)


# --------------------------------------------------------------------------- #
# query_leaves tool
# --------------------------------------------------------------------------- #
QUERY_TOOL_NAME = "query_leaves"


def query_tool_schema(rows_per_query: int) -> dict[str, Any]:
    num = {"type": "number"}
    return {
        "type": "function",
        "function": {
            "name": QUERY_TOOL_NAME,
            "description": (
                "Search the model's full leaf table. All filters are optional and combined with AND. "
                f"Returns at most {rows_per_query} leaves, one line each, plus the number of leaves matched."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "features": {"type": "array", "items": {"type": "string"},
                                 "description": "Only leaves whose rule uses ALL of these features."},
                    "rule_contains": {"type": "string",
                                      "description": "Only leaves whose rule text contains this substring."},
                    "tree_id": {"type": "integer", "description": "Only leaves of this tree."},
                    "min_n_cal": {"type": "integer"},
                    "max_n_cal": {"type": "integer"},
                    "min_mean_resid": num,
                    "max_mean_resid": num,
                    "min_abs_mean_resid": num,
                    "sort_by": {"type": "string", "enum": list(SORTABLE),
                                "description": "Column to sort by (leaves with no calibration rows sort last)."},
                    "descending": {"type": "boolean", "description": "Default true."},
                    "limit": {"type": "integer", "description": f"1..{rows_per_query}, default {rows_per_query}."},
                },
                "additionalProperties": False,
            },
        },
    }


class QueryArgs(BaseModel):
    model_config = {"extra": "forbid"}

    features: Optional[list[str]] = None
    rule_contains: Optional[str] = None
    tree_id: Optional[int] = None
    min_n_cal: Optional[int] = None
    max_n_cal: Optional[int] = None
    min_mean_resid: Optional[float] = None
    max_mean_resid: Optional[float] = None
    min_abs_mean_resid: Optional[float] = None
    sort_by: Optional[str] = None
    descending: bool = True
    limit: Optional[int] = None

    @field_validator("sort_by")
    @classmethod
    def _sortable(cls, v):
        if v is not None and v not in SORTABLE:
            raise ValueError(f"sort_by must be one of {list(SORTABLE)}")
        return v


def _feature_pattern(name: str) -> re.Pattern:
    # whole-token match so 'f1' doesn't match 'f10' and 'hour' doesn't match 'hours'
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")


def run_query(view: LeafView, raw_args: str | dict, rows_per_query: int) -> str:
    """Execute a ``query_leaves`` call. Bad arguments come back as an error message, not an exception."""
    try:
        args = QueryArgs.model_validate(json.loads(raw_args) if isinstance(raw_args, str) else raw_args)
    except (ValidationError, json.JSONDecodeError) as e:
        return f"ERROR: invalid query_leaves arguments: {e}".replace("\n", " ")[:600]
    t = view.table
    m = np.ones(len(t), dtype=bool)
    if args.features:
        unknown = [f for f in args.features if f not in view.features_shown]
        if unknown:
            return f"ERROR: unknown feature(s) {unknown}. Features: {view.features_shown}"
        for f in args.features:
            m &= t["rule"].str.contains(_feature_pattern(f), regex=True).to_numpy()
    if args.rule_contains:
        m &= t["rule"].str.contains(args.rule_contains, regex=False).to_numpy()
    if args.tree_id is not None:
        m &= (t["tree_id"] == args.tree_id).to_numpy()
    if args.min_n_cal is not None:
        m &= (t["n_cal"] >= args.min_n_cal).to_numpy()
    if args.max_n_cal is not None:
        m &= (t["n_cal"] <= args.max_n_cal).to_numpy()
    if args.min_mean_resid is not None:
        m &= (t["mean_resid"] >= args.min_mean_resid).to_numpy()
    if args.max_mean_resid is not None:
        m &= (t["mean_resid"] <= args.max_mean_resid).to_numpy()
    if args.min_abs_mean_resid is not None:
        m &= (t["abs_mean_resid"] >= args.min_abs_mean_resid).to_numpy()
    sel = t[m]
    if args.sort_by:
        sel = sel.sort_values(args.sort_by, ascending=not args.descending, na_position="last", kind="stable")
    limit = max(1, min(args.limit or rows_per_query, rows_per_query))
    shown = sel.head(limit)
    order = f", sorted by {args.sort_by} {'desc' if args.descending else 'asc'}" if args.sort_by else ""
    head = f"{len(sel)} leaves matched; showing {len(shown)}{order}."
    return "\n".join([head] + [view.line(r) for r in shown.to_dict("records")])


# --------------------------------------------------------------------------- #
# Output schemas
# --------------------------------------------------------------------------- #
class LeafPick(BaseModel):
    tree_id: int
    leaf_id: int
    reason: str = ""


class LeafEdit(LeafPick):
    delta: float


class SelectCandidate(BaseModel):
    leaves: list[LeafPick] = Field(min_length=1)
    rationale: str = ""


class DirectCandidate(BaseModel):
    leaves: list[LeafEdit] = Field(min_length=1)
    rationale: str = ""


class SelectProposal(BaseModel):
    candidates: list[SelectCandidate] = Field(min_length=1)


class DirectProposal(BaseModel):
    candidates: list[DirectCandidate] = Field(min_length=1)


class ProposalError(ValueError):
    """The model's answer can't be used. The message is sent back for the one retry."""


def extract_json(text: Optional[str]) -> Any:
    """Parse the JSON object in a reply (bare, or inside a ```json fence)."""
    if not text or not text.strip():
        raise ProposalError("empty answer; expected a JSON object")
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    cand = fence.group(1) if fence else text[text.find("{"): text.rfind("}") + 1]
    try:
        return json.loads(cand)
    except json.JSONDecodeError as e:
        raise ProposalError(f"answer is not valid JSON: {e}") from None


def parse_proposal(
    text: Optional[str],
    approach: Approach,
    index: LeafIndex,
    max_leaves: int,
    max_candidates: int,
    delta_bound: Optional[float] = None,
):
    """Validate a final answer. Raises :class:`ProposalError` with a message fit to send back."""
    obj = extract_json(text)
    schema = SelectProposal if approach == "select" else DirectProposal
    try:
        prop = schema.model_validate(obj)
    except ValidationError as e:
        raise ProposalError("answer does not match the schema: " + "; ".join(
            f"{'/'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()[:8])) from None
    errs = []
    if len(prop.candidates) > max_candidates:
        errs.append(f"{len(prop.candidates)} candidates given, at most {max_candidates} allowed")
    for i, c in enumerate(prop.candidates):
        keys = [(l.tree_id, l.leaf_id) for l in c.leaves]
        if len(keys) > max_leaves:
            errs.append(f"candidate {i}: {len(keys)} leaves, at most {max_leaves} allowed")
        if len(set(keys)) != len(keys):
            errs.append(f"candidate {i}: a leaf is listed twice")
        missing = [f"T{t}/L{l}" for t, l in keys if (t, l) not in index]
        if missing:
            errs.append(f"candidate {i}: leaves do not exist: {', '.join(missing[:5])}")
        if approach == "direct" and delta_bound is not None:
            big = [f"T{l.tree_id}/L{l.leaf_id} ({l.delta:g})" for l in c.leaves
                   if not np.isfinite(l.delta) or abs(l.delta) > delta_bound]
            if big:
                errs.append(f"candidate {i}: |delta| must be <= {sig(delta_bound)}: {', '.join(big[:5])}")
    if errs:
        raise ProposalError("; ".join(errs))
    return prop


# --------------------------------------------------------------------------- #
# Static block
# --------------------------------------------------------------------------- #
@dataclass
class PromptSpec:
    approach: Approach
    max_leaves: int
    candidates_per_session: int
    max_queries: int
    rows_per_query: int
    delta_bound: Optional[float] = None


def dataset_block(data: SplitData, view: LeafView) -> str:
    ranges = data.feature_ranges("train")
    y = data["train"].y
    if view.variant == "named":
        lines = [f"Target: {data.target_name} ({data.target_unit}); train mean {sig(y.mean())}, std {sig(y.std())}.",
                 "Features (name: description, unit, train range):"]
        for f in data.features:
            lo, hi = ranges[f.name]
            lines.append(f"- {f.name}: {f.description}; {f.unit}; [{sig(lo)}, {sig(hi)}]")
    else:
        inv = {v: k for k, v in view.name_map.items()}
        lines = [f"Target: y; train mean {sig(y.mean())}, std {sig(y.std())}.",
                 "Features (name: train range):"]
        for f in view.features_shown:
            lo, hi = ranges[inv[f]]
            lines.append(f"- {f}: [{sig(lo)}, {sig(hi)}]")
    return "\n".join(lines)


def model_block(booster, data: SplitData, view: LeafView, base_pred_cal: np.ndarray) -> str:
    t = view.table
    n_cal = t["n_cal"].to_numpy()
    tr = data["train"]
    train_rmse = float(np.sqrt(np.mean((tr.y - booster.predict(tr.X)) ** 2)))
    cal_rmse = float(np.sqrt(np.mean((data["cal"].y - base_pred_cal) ** 2)))
    q = np.percentile(n_cal, [50, 90, 99])
    per_tree = t.groupby("tree_id").size()
    return "\n".join([
        f"LightGBM regression: {booster.num_trees()} trees, {len(t)} leaves in total "
        f"(up to {int(per_tree.max())} per tree; rule depth {int(t.depth.min())}-{int(t.depth.max())}).",
        f"Prediction = sum over trees of the value of the leaf the row lands in.",
        f"RMSE: train {sig(train_rmse)}, calibration {sig(cal_rmse)}.",
        f"Calibration rows: {len(data['cal'].y)}. Leaves with at least 1 calibration row: {int((n_cal > 0).sum())}; "
        f"rows per leaf: median {sig(q[0])}, 90th percentile {sig(q[1])}, 99th percentile {sig(q[2])}, "
        f"max {int(n_cal.max())}.",
    ])


TASK_SELECT = """\
Choose which leaves to edit. For each candidate you pick up to {max_leaves} leaves; the edit \
values for those leaves are then fitted automatically by ridge regression on the calibration \
residuals, and the candidate is scored. You pick the leaves; you do not pick the values."""

TASK_DIRECT = """\
Choose which leaves to edit AND by how much. For each candidate you pick up to {max_leaves} \
leaves and a delta for each (added to that leaf's current value). Every |delta| must be <= \
{delta_bound}. The candidate is applied as you give it, then scored."""

STATIC_TEMPLATE = """\
You are improving an already-trained gradient-boosted tree model without retraining it. \
The tree structure is frozen; the only thing that can change is the output value of a few leaves.

# Task
{task}

You will see the effect of each candidate (its calibration RMSE) in the next round, and you \
can use that to propose better candidates. Lower RMSE is better. The goal is a model that is \
more accurate on new data from the same distribution, not just on the calibration rows.

# Data
{dataset}

# Model
{model}

# Leaf table
You cannot see the whole table at once. Use the `{tool}` tool (at most {max_queries} calls per \
round, at most {rows} leaves per call) to search it; filter and sort it however you think is useful. \
Each leaf is one line:
`T<tree>/L<leaf> | rule: <conditions on the path to this leaf> | n_cal=<calibration rows in the leaf> | value=<current leaf value> | mean_resid=<mean of (y - prediction) over those rows> | std_resid=<std of that residual>`
A positive mean_resid means the model under-predicts for rows in that leaf. Every row passes \
through exactly one leaf in every tree, so leaves of different trees overlap.
You may use your understanding of what the features mean.

# Answer
After your searches, reply with only a JSON object, no other text:
{schema}
Give exactly {n_cand} candidates, each with 1 to {max_leaves} distinct existing leaves. \
Different candidates may share leaves."""

SCHEMA_SELECT = ('{"candidates": [{"leaves": [{"tree_id": <int>, "leaf_id": <int>, "reason": "<short>"}, ...], '
                 '"rationale": "<one or two sentences>"}, ...]}')
SCHEMA_DIRECT = ('{"candidates": [{"leaves": [{"tree_id": <int>, "leaf_id": <int>, "delta": <number>, '
                 '"reason": "<short>"}, ...], "rationale": "<one or two sentences>"}, ...]}')


def static_block(spec: PromptSpec, data: SplitData, view: LeafView, booster, base_pred_cal: np.ndarray) -> str:
    task = (TASK_SELECT.format(max_leaves=spec.max_leaves) if spec.approach == "select"
            else TASK_DIRECT.format(max_leaves=spec.max_leaves, delta_bound=sig(spec.delta_bound)))
    return STATIC_TEMPLATE.format(
        task=task,
        dataset=dataset_block(data, view),
        model=model_block(booster, data, view, base_pred_cal),
        tool=QUERY_TOOL_NAME,
        max_queries=spec.max_queries,
        rows=spec.rows_per_query,
        schema=SCHEMA_SELECT if spec.approach == "select" else SCHEMA_DIRECT,
        n_cand=spec.candidates_per_session,
        max_leaves=spec.max_leaves,
    )


# --------------------------------------------------------------------------- #
# History block
# --------------------------------------------------------------------------- #
@dataclass
class HistoryItem:
    leaves: list[tuple[int, int]]
    deltas: Optional[list[float]]     # direct only
    cal_rmse: Optional[float]         # None for a failed candidate
    note: str = ""


def history_block(history: list[HistoryItem], base_cal_rmse: float, round_no: int, n_rounds: int,
                  n_cand: int) -> str:
    lines = [f"Round {round_no} of {n_rounds}. Base model calibration RMSE: {sig(base_cal_rmse, 4)}."]
    if not history:
        lines.append("No candidates evaluated yet.")
    else:
        lines.append("Candidates evaluated so far (calibration RMSE after the edit; lower is better):")
        for i, h in enumerate(history):
            if h.cal_rmse is None:
                lines.append(f"{i + 1}. FAILED ({h.note})")
                continue
            if h.deltas is None:
                edits = ", ".join(f"T{t}/L{l}" for t, l in h.leaves)
            else:
                edits = ", ".join(f"T{t}/L{l} {sig(d, 3, sign=True)}" for (t, l), d in zip(h.leaves, h.deltas))
            lines.append(f"{i + 1}. RMSE {sig(h.cal_rmse, 4)} | {edits}")
    lines.append(f"Search the leaf table as needed, then answer with {n_cand} new candidates.")
    return "\n".join(lines)
