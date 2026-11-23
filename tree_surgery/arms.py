"""Experiment arms behind one interface (PLAN §4).

Every arm receives an :class:`ArmContext` that holds the base model and the
train / cal / val splits. **The test split is not in the context**, so no arm,
optimizer or selection step can see it. The runner scores the arm's final model
on test afterwards.

Edit arms (``random_opt``, ``sparse_opt``, and in Phase 4 ``llm_select``) subclass
:class:`EditArm`. They only *propose leaf subsets*; the shared base class

1. enforces the edit budget (<= ``max_leaves`` leaves per candidate,
   <= ``max_candidates`` candidates per run), raising :class:`BudgetError` otherwise;
2. fits each candidate's deltas by ridge on **calibration**, with lambda chosen on
   validation (PLAN §6);
3. picks the candidate with the best **validation** RMSE;
4. materialises it with :class:`~tree_surgery.patcher.ModelPatcher` and checks that
   the patched booster's predictions equal the linear prediction ``y_base + X d``.

Reference ceilings (``refit``, ``retrain``) ignore the edit budget and are never
competitors.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .datasets import Split, SplitData
from .llm_client import Usage
from .design_matrix import LeafIndex, leaf_design_matrix, leaf_values
from .optimizers import DEFAULT_LAMBDAS, fit_ridge_select_lambda, lasso_supports, omp_path
from .patcher import ModelPatcher
from .train import train_base_model

VISIBLE_SPLITS = ("train", "cal", "val")
LINEARITY_TOL = 1e-8


class BudgetError(RuntimeError):
    """An edit arm proposed more leaves or more candidates than the budget allows."""


# --------------------------------------------------------------------------- #
# Context and outputs
# --------------------------------------------------------------------------- #
@dataclass
class ArmContext:
    """Everything an arm may look at. Deliberately has no test split."""

    dataset: str
    seed: int
    base: lgb.Booster
    patcher: ModelPatcher
    splits: dict[str, Split]                   # train / cal / val only
    X_cal: sp.csr_matrix                       # leaf design matrices
    X_val: sp.csr_matrix
    base_pred: dict[str, np.ndarray]           # base predictions on visible splits
    max_leaves: int = 10
    max_candidates: int = 10
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS
    base_cfg: dict[str, Any] = field(default_factory=dict)
    arm_cfg: dict[str, dict[str, Any]] = field(default_factory=dict)
    data: Optional[SplitData] = None           # metadata + train/cal/val; no test split
    llm_cfg: dict[str, Any] = field(default_factory=dict)
    delta_bound_std: float = 0.5
    _views: dict[str, Any] = field(default_factory=dict, repr=False)

    def leaf_view(self, variant: str):
        """Full leaf table rendered for ``variant`` ('named' | 'anon'), built once per context."""
        from .prompts import LeafView

        if variant not in self._views:
            self._views[variant] = LeafView.build(self.base, self.data, variant, self.index)
        return self._views[variant]

    @property
    def delta_bound(self) -> float:
        return self.delta_bound_std * float(np.std(self.splits["train"].y))

    @property
    def index(self) -> LeafIndex:
        return self.patcher.index

    @property
    def r_cal(self) -> np.ndarray:
        return self.splits["cal"].y - self.base_pred["cal"]

    @property
    def r_val(self) -> np.ndarray:
        return self.splits["val"].y - self.base_pred["val"]

    @property
    def n_cal_per_leaf(self) -> np.ndarray:
        return np.asarray(self.X_cal.sum(axis=0)).ravel().astype(np.int64)

    def cfg(self, arm: str) -> dict[str, Any]:
        return self.arm_cfg.get(arm, {}) or {}


def build_context(
    data: SplitData,
    base: lgb.Booster,
    cfg: Optional[dict[str, Any]] = None,
) -> ArmContext:
    """Build the arm context for one (dataset, seed). ``cfg`` is the full experiment config."""
    cfg = cfg or {}
    patcher = ModelPatcher(base)
    splits = {k: data[k] for k in VISIBLE_SPLITS}
    budget = cfg.get("edit_budget", {})
    return ArmContext(
        dataset=data.name,
        seed=data.seed,
        base=base,
        patcher=patcher,
        splits=splits,
        X_cal=leaf_design_matrix(base, splits["cal"].X, patcher.index),
        X_val=leaf_design_matrix(base, splits["val"].X, patcher.index),
        base_pred={k: base.predict(s.X) for k, s in splits.items()},
        max_leaves=int(budget.get("max_leaves", 10)),
        max_candidates=int(budget.get("max_candidates", 10)),
        lambdas=tuple(cfg.get("ridge_lambdas", DEFAULT_LAMBDAS)),
        base_cfg=cfg.get("base_model", {}),
        arm_cfg=cfg.get("arms", {}),
        data=SplitData(data.name, data.seed, data.target_name, data.target_unit, data.features, dict(splits)),
        llm_cfg=cfg.get("llm", {}),
        delta_bound_std=float(cfg.get("llm_direct_delta_bound_std", 0.5)),
    )


@dataclass
class CandidateRecord:
    """One evaluated candidate. Holds calibration / validation scores only, never test."""

    index: int
    leaves: list[tuple[int, int]]
    deltas: list[float]
    cal_rmse: float
    val_rmse: float
    lam: Optional[float] = None
    status: str = "ok"                         # "ok" | "failed" (Phase 4: invalid LLM output)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "leaves": [list(k) for k in self.leaves],
            "deltas": self.deltas,
            "cal_rmse": self.cal_rmse,
            "val_rmse": self.val_rmse,
            "lam": self.lam,
            "status": self.status,
            "meta": self.meta,
        }


@dataclass
class ArmOutput:
    arm: str
    model: lgb.Booster                         # final model; the runner scores it on every split
    candidates: list[CandidateRecord] = field(default_factory=list)
    chosen: Optional[int] = None               # index into candidates
    leaves_edited: Optional[int] = None        # None when structure changes (retrain)
    frac_params_changed: Optional[float] = None
    within_budget: bool = True
    info: dict[str, Any] = field(default_factory=dict)
    runtime_s: float = 0.0


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# --------------------------------------------------------------------------- #
# Arm interface
# --------------------------------------------------------------------------- #
class Arm:
    name: str = "arm"

    def run(self, ctx: ArmContext) -> ArmOutput:
        t0 = time.perf_counter()
        out = self._run(ctx)
        out.runtime_s = time.perf_counter() - t0
        return out

    def _run(self, ctx: ArmContext) -> ArmOutput:  # pragma: no cover - abstract
        raise NotImplementedError


class OriginalArm(Arm):
    """The unedited base model (floor)."""

    name = "original"

    def _run(self, ctx: ArmContext) -> ArmOutput:
        return ArmOutput(self.name, ctx.patcher.base_booster(), leaves_edited=0, frac_params_changed=0.0)


class EditArm(Arm):
    """Budgeted edit arm. Subclasses implement :meth:`propose` (leaf-column subsets)."""

    def propose(self, ctx: ArmContext) -> list[list[int]]:  # pragma: no cover - abstract
        raise NotImplementedError

    def annotate(self, records: list[CandidateRecord]) -> None:
        """Hook: attach per-candidate metadata (LLM reasons, session ids, failure causes)."""

    def extra_info(self) -> dict[str, Any]:
        """Hook: arm-level info merged into ``ArmOutput.info`` (e.g. LLM usage)."""
        return {}

    def check_budget(self, ctx: ArmContext, subsets: Sequence[Sequence[int]]) -> None:
        if len(subsets) > ctx.max_candidates:
            raise BudgetError(f"{self.name}: {len(subsets)} candidates, budget is {ctx.max_candidates}")
        for s in subsets:
            if len(s) > ctx.max_leaves:
                raise BudgetError(f"{self.name}: candidate edits {len(s)} leaves, budget is {ctx.max_leaves}")
            if len(set(s)) != len(s):
                raise BudgetError(f"{self.name}: candidate lists a leaf twice")
            for c in s:
                if not 0 <= int(c) < ctx.index.num_leaves:
                    raise BudgetError(f"{self.name}: leaf column {c} does not exist")

    def _run(self, ctx: ArmContext) -> ArmOutput:
        subsets = [list(map(int, s)) for s in self.propose(ctx)]
        self.check_budget(ctx, subsets)
        r_cal, r_val = ctx.r_cal, ctx.r_val
        records: list[CandidateRecord] = []
        for i, cols in enumerate(subsets):
            if not cols:
                records.append(CandidateRecord(i, [], [], _rmse(r_cal, 0), _rmse(r_val, 0), status="failed",
                                               meta={"reason": "empty subset"}))
                continue
            fit = fit_ridge_select_lambda(ctx.X_cal, r_cal, ctx.X_val, r_val, cols, ctx.lambdas)
            records.append(
                CandidateRecord(
                    index=i,
                    leaves=[ctx.index.key(c) for c in cols],
                    deltas=[float(d) for d in fit.deltas],
                    cal_rmse=fit.cal_rmse,
                    val_rmse=fit.val_rmse,
                    lam=fit.lam,
                    meta={"val_rmse_by_lambda": fit.val_rmse_by_lambda},
                )
            )
        self.annotate(records)
        ok = [r for r in records if r.status == "ok"]
        if not ok:
            return ArmOutput(self.name, ctx.patcher.base_booster(), records, None, 0, 0.0,
                             info={"note": "no valid candidate; base model reported", **self.extra_info()})
        best = min(ok, key=lambda r: (r.val_rmse, r.index))

        cand = ctx.patcher.candidate().apply(
            {k: d for k, d in zip(best.leaves, best.deltas)}, max_leaves=ctx.max_leaves
        )
        # Guard: the materialised booster must agree with the linear model used for selection.
        cols = [ctx.index.col(*k) for k in best.leaves]
        linear_val = ctx.base_pred["val"] + ctx.X_val[:, cols] @ np.asarray(best.deltas)
        err = float(np.max(np.abs(cand.predict(ctx.splits["val"].X) - linear_val)))
        if err > LINEARITY_TOL:
            raise AssertionError(f"{self.name}: patched model disagrees with linear prediction by {err:.2e}")

        n_edit = len(cand.edited_leaves)
        return ArmOutput(
            self.name,
            cand.booster,
            records,
            chosen=best.index,
            leaves_edited=n_edit,
            frac_params_changed=n_edit / ctx.index.num_leaves,
            info={"lam": best.lam, "linearity_err": err,
                  "chosen_beats_original_on_val": best.val_rmse < _rmse(r_val, 0), **self.extra_info()},
        )


class RandomOptArm(EditArm):
    """Random leaves, ridge values (sanity check).

    Each candidate is an independent draw of ``max_leaves`` distinct leaves from
    those with at least ``min_n_cal`` calibration rows (a leaf with no calibration
    rows cannot be fitted, so drawing it would only shrink the effective budget).
    """

    name = "random_opt"

    def propose(self, ctx: ArmContext) -> list[list[int]]:
        min_n = int(ctx.cfg(self.name).get("min_n_cal", 1))
        pool = np.flatnonzero(ctx.n_cal_per_leaf >= min_n)
        rng = np.random.default_rng([ctx.seed, 0x4A4D, sum(map(ord, ctx.dataset))])
        k = min(ctx.max_leaves, len(pool))
        return [sorted(rng.choice(pool, size=k, replace=False).tolist()) for _ in range(ctx.max_candidates)]


class SparseOptArm(EditArm):
    """Sparse regression on leaf indicators picks the leaves, ridge sets the values (main competitor).

    ``method: omp`` (default): one greedy OMP run to ``max_leaves`` leaves; candidate
    ``j`` is the first ``j + 1`` leaves of the path, so 10 candidates = supports of
    size 1..10. ``method: lasso``: distinct supports of size <= ``max_leaves`` along
    the Lasso path, thinned to ``max_candidates``.
    """

    name = "sparse_opt"

    def propose(self, ctx: ArmContext) -> list[list[int]]:
        method = ctx.cfg(self.name).get("method", "omp")
        if method == "omp":
            path = omp_path(ctx.X_cal, ctx.r_cal, ctx.max_leaves)
            sizes = np.unique(np.round(np.linspace(1, len(path), min(ctx.max_candidates, len(path)))).astype(int))
            return [path[:j] for j in sizes]
        if method == "lasso":
            return lasso_supports(ctx.X_cal, ctx.r_cal, ctx.max_leaves, ctx.max_candidates)
        raise ValueError(f"unknown sparse_opt method {method!r}")


class RefitArm(Arm):
    """``Booster.refit`` on calibration: every leaf re-estimated, structure fixed (ceiling, no budget).

    ``new_leaf = decay * old_leaf + (1 - decay) * refit_leaf``; decay chosen on validation.
    """

    name = "refit"

    def _run(self, ctx: ArmContext) -> ArmOutput:
        decays = ctx.cfg(self.name).get("decay_rates", [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        cal, val = ctx.splits["cal"], ctx.splits["val"]
        base = ctx.patcher.base_booster()
        records, models = [], []
        for i, decay in enumerate(decays):
            m = base.refit(cal.X, cal.y, decay_rate=float(decay))
            records.append(CandidateRecord(i, [], [], _rmse(cal.y, m.predict(cal.X)),
                                           _rmse(val.y, m.predict(val.X)), meta={"decay_rate": float(decay)}))
            models.append(m)
        best = min(records, key=lambda r: (r.val_rmse, r.index))
        model = models[best.index]
        changed = int(np.sum(leaf_values(model, ctx.index) != ctx.patcher.base_values))
        return ArmOutput(self.name, model, records, best.index, changed, changed / ctx.index.num_leaves,
                         within_budget=False, info={"decay_rate": best.meta["decay_rate"]})


class RetrainArm(Arm):
    """Retrain from scratch on train + calibration, same hyperparameters and early stopping (ceiling)."""

    name = "retrain"

    def _run(self, ctx: ArmContext) -> ArmOutput:
        tr, cal = ctx.splits["train"], ctx.splits["cal"]
        merged = Split(
            X=pd.concat([tr.X, cal.X], ignore_index=True),
            y=np.concatenate([tr.y, cal.y]),
            idx=np.concatenate([tr.idx, cal.idx]),
        )
        data = SplitData(ctx.dataset, ctx.seed, "", "", [], {"train": merged, "val": ctx.splits["val"]})
        model = train_base_model(
            data,
            params=ctx.base_cfg.get("params"),
            num_boost_round=ctx.base_cfg.get("num_boost_round", 5000),
            early_stopping_rounds=ctx.base_cfg.get("early_stopping_rounds", 50),
        )
        val = ctx.splits["val"]
        rec = CandidateRecord(0, [], [], _rmse(cal.y, model.predict(cal.X)), _rmse(val.y, model.predict(val.X)),
                              meta={"num_trees": model.num_trees()})
        return ArmOutput(self.name, model, [rec], 0, None, None, within_budget=False,
                         info={"num_trees": model.num_trees()})


# --------------------------------------------------------------------------- #
# LLM arms (PLAN §4, §5): llm_select (Approach B) and llm_direct (Approach A)
# --------------------------------------------------------------------------- #
class _LLMSessions:
    """Shared session loop: rounds of ``candidates_per_session`` candidates with calibration feedback.

    Feedback shown to the LLM is calibration-only. For ``select`` the calibration RMSE is
    that of a ridge fit at the smallest grid lambda, so it carries no validation information
    (the reported candidate still gets its lambda chosen on validation, as for every arm).
    """

    approach = "select"

    def __init__(self, client, variant: str = "named", run: int = 0):
        if variant not in ("named", "anon"):
            raise ValueError("variant must be 'named' or 'anon'")
        self.client, self.variant, self.run_id = client, variant, run
        self._meta: list[dict[str, Any]] = []
        self._info: dict[str, Any] = {}

    def _spec(self, ctx: ArmContext):
        from .prompts import PromptSpec

        sess = ctx.llm_cfg.get("session", {})
        return PromptSpec(
            approach=self.approach,
            max_leaves=ctx.max_leaves,
            candidates_per_session=int(sess.get("candidates_per_session", 5)),
            max_queries=int(sess.get("max_queries", 3)),
            rows_per_query=int(sess.get("rows_per_query", 30)),
            delta_bound=ctx.delta_bound if self.approach == "direct" else None,
        )

    def _feedback(self, ctx: ArmContext, cols: list[int], deltas: Optional[np.ndarray]) -> float:
        r = ctx.r_cal
        A = ctx.X_cal[:, cols]
        if deltas is None:
            from .optimizers import ridge_on_subset

            deltas = ridge_on_subset(ctx.X_cal, r, cols, min(ctx.lambdas))
        return _rmse(r, A @ np.asarray(deltas))

    def run_sessions(self, ctx: ArmContext) -> list[dict[str, Any]]:
        """Returns one dict per candidate slot: {cols, deltas, status, meta}."""
        from .llm_session import run_session
        from .prompts import HistoryItem, history_block, static_block

        spec = self._spec(ctx)
        view = ctx.leaf_view(self.variant)
        static = static_block(spec, ctx.data, view, ctx.base, ctx.base_pred["cal"])
        base_cal = _rmse(ctx.r_cal, 0)
        n_rounds = -(-ctx.max_candidates // spec.candidates_per_session)
        params = {"max_tokens": int(ctx.llm_cfg.get("max_tokens_per_call", 4000))}
        if ctx.llm_cfg.get("temperature") is not None:
            params["temperature"] = float(ctx.llm_cfg["temperature"])
        if hasattr(self.client, "cache_tag"):
            self.client.cache_tag = f"run{self.run_id}"
        log0 = len(getattr(self.client, "log", []))

        history: list[HistoryItem] = []
        slots: list[dict[str, Any]] = []
        sessions = []
        for rnd in range(n_rounds):
            room = ctx.max_candidates - len(slots)
            if room <= 0:
                break
            text = history_block(history, base_cal, rnd + 1, n_rounds, min(room, spec.candidates_per_session))
            spec_r = spec if room >= spec.candidates_per_session else type(spec)(**{**spec.__dict__,
                                                                                  "candidates_per_session": room})
            res = run_session(self.client, static, text, view, spec_r, ctx.index, params)
            sessions.append({"round": rnd, "ok": res.ok, "error": res.error, "n_queries": res.n_queries,
                             "n_calls": res.n_calls, "retried": res.retried, "queries": res.queries,
                             "usage": res.usage.__dict__, "cost_usd": res.cost_usd})
            if not res.ok:
                for _ in range(min(room, spec_r.candidates_per_session)):
                    slots.append({"cols": [], "deltas": None, "status": "failed",
                                  "meta": {"round": rnd, "reason": res.error}})
                    history.append(HistoryItem([], None, None, "invalid answer"))
                continue
            for c in res.proposal.candidates[:room]:
                keys = [(l.tree_id, l.leaf_id) for l in c.leaves]
                cols = [ctx.index.col(t, l) for t, l in keys]
                deltas = [float(l.delta) for l in c.leaves] if self.approach == "direct" else None
                fb = self._feedback(ctx, cols, None if deltas is None else np.asarray(deltas))
                slots.append({"cols": cols, "deltas": deltas, "status": "ok",
                              "meta": {"round": rnd, "rationale": c.rationale,
                                       "reasons": [l.reason for l in c.leaves], "feedback_cal_rmse": fb}})
                history.append(HistoryItem(keys, deltas, fb))

        log = getattr(self.client, "log", [])[log0:]
        tot = Usage()
        for e in log:
            tot = tot + Usage(**e["usage"])
        self._info = {
            "variant": self.variant, "run": self.run_id, "model": getattr(self.client, "model", "?"),
            "llm_calls": len(log), "llm_cache_hits": sum(e["from_cache"] for e in log),
            "llm_tokens_in": tot.prompt_tokens, "llm_tokens_out": tot.completion_tokens,
            "llm_tokens_cached": tot.cached_tokens, "llm_tokens_reasoning": tot.reasoning_tokens,
            "llm_cost_usd": float(sum(e["cost_usd"] for e in log)),
            "llm_billed_usd": float(sum(e["billed_usd"] for e in log)),
            "llm_latency_s": float(sum(e["latency_s"] for e in log)),
            "sessions": sessions,
            "static_prompt_chars": len(static),
        }
        return slots


class LLMSelectArm(_LLMSessions, EditArm):
    """Approach B: the LLM picks the leaves, ridge on calibration sets the values."""

    name = "llm_select"
    approach = "select"

    def propose(self, ctx: ArmContext) -> list[list[int]]:
        slots = self.run_sessions(ctx)
        self._meta = [s["meta"] | {"status": s["status"]} for s in slots]
        return [s["cols"] for s in slots]

    def annotate(self, records: list[CandidateRecord]) -> None:
        for rec, m in zip(records, self._meta):
            rec.meta.update(m)
            if m["status"] == "failed":
                rec.meta["reason"] = m.get("reason")

    def extra_info(self) -> dict[str, Any]:
        return dict(self._info)


class LLMDirectArm(_LLMSessions, Arm):
    """Approach A: the LLM picks the leaves and the deltas (|delta| <= bound)."""

    name = "llm_direct"
    approach = "direct"

    def _run(self, ctx: ArmContext) -> ArmOutput:
        slots = self.run_sessions(ctx)
        if len(slots) > ctx.max_candidates or any(len(s["cols"]) > ctx.max_leaves for s in slots):
            raise BudgetError(f"{self.name}: budget exceeded")
        r_cal, r_val = ctx.r_cal, ctx.r_val
        records = []
        for i, s in enumerate(slots):
            if s["status"] != "ok":
                records.append(CandidateRecord(i, [], [], _rmse(r_cal, 0), _rmse(r_val, 0), status="failed",
                                               meta=s["meta"]))
                continue
            d = np.asarray(s["deltas"])
            records.append(CandidateRecord(
                i, [ctx.index.key(c) for c in s["cols"]], [float(x) for x in d],
                _rmse(r_cal, ctx.X_cal[:, s["cols"]] @ d), _rmse(r_val, ctx.X_val[:, s["cols"]] @ d),
                meta=s["meta"]))
        ok = [r for r in records if r.status == "ok"]
        if not ok:
            return ArmOutput(self.name, ctx.patcher.base_booster(), records, None, 0, 0.0,
                             info={"note": "no valid candidate; base model reported", **self._info})
        best = min(ok, key=lambda r: (r.val_rmse, r.index))
        if any(abs(x) > ctx.delta_bound + 1e-12 for x in best.deltas):
            raise BudgetError(f"{self.name}: delta outside the bound")
        cand = ctx.patcher.candidate().apply(dict(zip(best.leaves, best.deltas)), max_leaves=ctx.max_leaves)
        cols = [ctx.index.col(*k) for k in best.leaves]
        linear_val = ctx.base_pred["val"] + ctx.X_val[:, cols] @ np.asarray(best.deltas)
        err = float(np.max(np.abs(cand.predict(ctx.splits["val"].X) - linear_val)))
        if err > LINEARITY_TOL:
            raise AssertionError(f"{self.name}: patched model disagrees with linear prediction by {err:.2e}")
        n = len(cand.edited_leaves)
        return ArmOutput(self.name, cand.booster, records, best.index, n, n / ctx.index.num_leaves,
                         info={"linearity_err": err, "chosen_beats_original_on_val": best.val_rmse < _rmse(r_val, 0),
                               **self._info})


LLM_ARMS = {"llm_select": LLMSelectArm, "llm_direct": LLMDirectArm}


ARMS: dict[str, type[Arm]] = {
    a.name: a for a in (OriginalArm, RandomOptArm, SparseOptArm, RefitArm, RetrainArm)
}
NON_LLM_ARMS = tuple(ARMS)


def get_arm(name: str, client=None, variant: str = "named", run: int = 0) -> Arm:
    if name in LLM_ARMS:
        if client is None:
            raise ValueError(f"{name} needs an LLM client (see tree_surgery.llm_client.make_client)")
        return LLM_ARMS[name](client, variant=variant, run=run)
    try:
        return ARMS[name]()
    except KeyError:
        raise ValueError(f"unknown arm {name!r}; known: {sorted(ARMS) + sorted(LLM_ARMS)}") from None
