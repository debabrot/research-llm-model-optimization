"""Leaf-value surgery on LightGBM boosters (PLAN §4, §11).

``ModelPatcher`` holds an immutable text snapshot of the base model. Every
``candidate()`` is a brand-new Booster parsed from that snapshot, so candidates
share no state with each other or with the base.

``Candidate.apply`` is atomic (all edits validated before any is written) and
logged. ``rollback`` restores the exact original doubles, so predictions return
bit-for-bit to where they were.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

import lightgbm as lgb
import numpy as np

from .design_matrix import LeafIndex, LeafKey, leaf_values


class EditError(ValueError):
    """Invalid edit request: unknown leaf, duplicate leaf, non-finite delta or budget exceeded."""


@dataclass(frozen=True)
class LeafEdit:
    tree_id: int
    leaf_id: int
    delta: float
    old_value: float
    new_value: float


def _normalise(edits: Mapping[LeafKey, float] | Iterable[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    if isinstance(edits, Mapping):
        return [(int(t), int(l), float(d)) for (t, l), d in edits.items()]
    return [(int(t), int(l), float(d)) for t, l, d in edits]


class Candidate:
    """An editable, independent copy of the base model."""

    def __init__(self, booster: lgb.Booster, index: LeafIndex, base_values: np.ndarray):
        self.booster = booster
        self.index = index
        self._base_values = base_values  # read-only reference, never mutated
        self.edit_log: list[LeafEdit] = []

    # ---- editing --------------------------------------------------------- #
    def apply(
        self,
        edits: Mapping[LeafKey, float] | Iterable[tuple[int, int, float]],
        max_leaves: Optional[int] = None,
        max_abs_delta: Optional[float] = None,
    ) -> "Candidate":
        """Add ``delta`` to each listed leaf. Validates everything first; on error nothing changes.

        ``max_leaves`` caps the number of *distinct* leaves edited over this candidate's
        lifetime (PLAN: <= 10). ``max_abs_delta`` bounds each delta (Approach A).
        """
        items = _normalise(edits)
        seen: set[LeafKey] = set()
        for t, l, d in items:
            if (t, l) not in self.index:
                raise EditError(f"leaf T{t}/L{l} does not exist")
            if (t, l) in seen:
                raise EditError(f"leaf T{t}/L{l} listed twice")
            if not math.isfinite(d):
                raise EditError(f"non-finite delta for T{t}/L{l}: {d}")
            if max_abs_delta is not None and abs(d) > max_abs_delta:
                raise EditError(f"|delta| {abs(d):.4g} for T{t}/L{l} exceeds bound {max_abs_delta:.4g}")
            seen.add((t, l))
        if max_leaves is not None:
            total = self.edited_leaves | seen
            if len(total) > max_leaves:
                raise EditError(f"{len(total)} distinct leaves edited, budget is {max_leaves}")

        for t, l, d in items:
            old = self.booster.get_leaf_output(t, l)
            new = old + d
            self.booster.set_leaf_output(t, l, new)
            self.edit_log.append(LeafEdit(t, l, d, old, new))
        return self

    def rollback(self, n: Optional[int] = None) -> "Candidate":
        """Undo the last ``n`` edits (all if None), restoring exact original values."""
        n = len(self.edit_log) if n is None else n
        for _ in range(n):
            e = self.edit_log.pop()
            self.booster.set_leaf_output(e.tree_id, e.leaf_id, e.old_value)
        return self

    # ---- inspection ------------------------------------------------------ #
    @property
    def edited_leaves(self) -> set[LeafKey]:
        return {(e.tree_id, e.leaf_id) for e in self.edit_log}

    def leaf_values(self) -> np.ndarray:
        return leaf_values(self.booster, self.index)

    def delta_vector(self) -> np.ndarray:
        """Current leaf values minus base leaf values, in design-matrix column order."""
        d = np.zeros(self.index.num_leaves)
        for t, l in self.edited_leaves:
            j = self.index.col(t, l)
            d[j] = self.booster.get_leaf_output(t, l) - self._base_values[j]
        return d

    def predict(self, X) -> np.ndarray:
        return self.booster.predict(X)

    # ---- persistence ----------------------------------------------------- #
    def save(self, out_dir: Path | str) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(out / "model.txt"))
        (out / "edits.json").write_text(json.dumps([asdict(e) for e in self.edit_log], indent=2))
        return out


class ModelPatcher:
    """Factory for independent edit candidates from one frozen base model."""

    def __init__(self, base: lgb.Booster):
        self._base_str = base.model_to_string()
        self.index = LeafIndex.from_booster(base)
        self.base_values = leaf_values(base, self.index)
        self.base_values.setflags(write=False)

    def candidate(self) -> Candidate:
        return Candidate(lgb.Booster(model_str=self._base_str), self.index, self.base_values)

    def base_booster(self) -> lgb.Booster:
        """A fresh copy of the unedited base model."""
        return lgb.Booster(model_str=self._base_str)


def load_candidate(path: Path | str, patcher: ModelPatcher) -> Candidate:
    """Load a saved candidate (model + edit log) against its base patcher."""
    path = Path(path)
    booster = lgb.Booster(model_file=str(path / "model.txt"))
    cand = Candidate(booster, patcher.index, patcher.base_values)
    log_path = path / "edits.json"
    if log_path.exists():
        cand.edit_log = [LeafEdit(**e) for e in json.loads(log_path.read_text())]
    return cand
