"""Sparse one-hot leaf design matrix (PLAN §1).

With structure fixed, a LightGBM regression model's prediction is linear in its
leaf values:

    y_hat(x) = sum_t leaf_value[t, leaf_t(x)] = X[x, :] @ v

where X is the (n x L) 0/1 matrix with exactly one 1 per tree per row, and v
stacks all leaf values in (tree, leaf) order. An edit is a delta vector d and the
edited model predicts y_hat + X @ d.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import lightgbm as lgb
import numpy as np
import scipy.sparse as sp

LeafKey = tuple[int, int]


@dataclass(frozen=True)
class LeafIndex:
    """Bijection between (tree_id, leaf_id) and a column index in the design matrix."""

    leaves_per_tree: tuple[int, ...]

    @classmethod
    def from_booster(cls, booster: lgb.Booster) -> "LeafIndex":
        info = booster.dump_model()["tree_info"]
        return cls(tuple(int(t["num_leaves"]) for t in info))

    @property
    def offsets(self) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(self.leaves_per_tree)]).astype(np.int64)

    @property
    def num_trees(self) -> int:
        return len(self.leaves_per_tree)

    @property
    def num_leaves(self) -> int:
        return int(sum(self.leaves_per_tree))

    def __len__(self) -> int:
        return self.num_leaves

    def __contains__(self, key: object) -> bool:
        if not (isinstance(key, tuple) and len(key) == 2):
            return False
        t, l = key
        return (
            isinstance(t, (int, np.integer))
            and isinstance(l, (int, np.integer))
            and 0 <= t < self.num_trees
            and 0 <= l < self.leaves_per_tree[t]
        )

    def col(self, tree_id: int, leaf_id: int) -> int:
        if (tree_id, leaf_id) not in self:
            raise KeyError(f"leaf T{tree_id}/L{leaf_id} does not exist")
        return int(self.offsets[tree_id] + leaf_id)

    def key(self, col: int) -> LeafKey:
        if not 0 <= col < self.num_leaves:
            raise IndexError(col)
        t = int(np.searchsorted(self.offsets, col, side="right") - 1)
        return t, int(col - self.offsets[t])

    def keys(self) -> list[LeafKey]:
        return [(t, l) for t, n in enumerate(self.leaves_per_tree) for l in range(n)]


def leaf_assignments(booster: lgb.Booster, X) -> np.ndarray:
    """(n, T) int array: the leaf each sample lands in, per tree."""
    leaves = booster.predict(X, pred_leaf=True)
    return np.asarray(leaves, dtype=np.int64).reshape(len(X), booster.num_trees())


def leaf_design_matrix(booster: lgb.Booster, X, index: LeafIndex | None = None) -> sp.csr_matrix:
    """Sparse (n x L) one-hot leaf membership matrix, float64."""
    index = index or LeafIndex.from_booster(booster)
    leaves = leaf_assignments(booster, X)
    n, T = leaves.shape
    cols = (leaves + index.offsets[:-1][None, :]).ravel()
    data = np.ones(n * T, dtype=np.float64)
    indptr = np.arange(0, n * T + 1, T, dtype=np.int64)
    return sp.csr_matrix((data, cols, indptr), shape=(n, index.num_leaves))


def leaf_values(booster: lgb.Booster, index: LeafIndex | None = None) -> np.ndarray:
    """Current leaf values in design-matrix column order (exact doubles)."""
    index = index or LeafIndex.from_booster(booster)
    return np.array([booster.get_leaf_output(t, l) for t, l in index.keys()], dtype=np.float64)


def delta_vector(edits: Mapping[LeafKey, float] | Iterable[tuple[int, int, float]], index: LeafIndex) -> np.ndarray:
    """Dense length-L delta vector from {(tree, leaf): delta} or [(tree, leaf, delta), ...]."""
    items = edits.items() if isinstance(edits, Mapping) else (((t, l), d) for t, l, d in edits)
    d = np.zeros(index.num_leaves, dtype=np.float64)
    for (t, l), delta in items:
        d[index.col(t, l)] += float(delta)
    return d
