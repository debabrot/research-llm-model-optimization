"""Datasets and seeded splits (PLAN §2).

D1 ``delivery``   semi-synthetic food-delivery time, known noise-free truth ``f_true``.
D2 ``california`` sklearn California Housing.

Every split is a random 50/15/15/20 partition into train / cal / val / test,
fully determined by the seed. For D1 the seed also determines the data draw.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

SPLIT_NAMES = ("train", "cal", "val", "test")
DEFAULT_FRACTIONS = {"train": 0.50, "cal": 0.15, "val": 0.15, "test": 0.20}


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureSpec:
    name: str
    unit: str
    description: str


@dataclass
class Dataset:
    """A full dataset before splitting."""

    name: str
    X: pd.DataFrame
    y: np.ndarray
    target_name: str
    target_unit: str
    features: list[FeatureSpec]
    f_true: Optional[np.ndarray] = None  # noise-free target, D1 only

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)


@dataclass
class Split:
    X: pd.DataFrame
    y: np.ndarray
    idx: np.ndarray                     # row indices into the full dataset
    f_true: Optional[np.ndarray] = None


@dataclass
class SplitData:
    """A dataset partitioned into train / cal / val / test for one seed."""

    name: str
    seed: int
    target_name: str
    target_unit: str
    features: list[FeatureSpec]
    splits: dict[str, Split] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Split:
        return self.splits[key]

    @property
    def feature_names(self) -> list[str]:
        return list(self.splits["train"].X.columns)

    def feature_ranges(self, split: str = "train") -> dict[str, tuple[float, float]]:
        X = self.splits[split].X
        return {c: (float(X[c].min()), float(X[c].max())) for c in X.columns}


# --------------------------------------------------------------------------- #
# D1: semi-synthetic delivery time
# --------------------------------------------------------------------------- #
DELIVERY_FEATURES = [
    FeatureSpec("distance_km", "km", "Road distance from restaurant to customer"),
    FeatureSpec("hour", "hour of day (0-23)", "Local hour when the order was placed"),
    FeatureSpec("rain_mm", "mm/h", "Rainfall intensity at order time"),
    FeatureSpec("traffic_index", "index 0-10", "Live congestion index along the route"),
    FeatureSpec("driver_experience_yrs", "years", "Courier's delivery experience"),
    FeatureSpec("order_size", "items", "Number of items in the order"),
    FeatureSpec("prep_time_min", "minutes", "Restaurant's estimated preparation time"),
    FeatureSpec("is_weekend", "0/1", "1 if the order was placed on Saturday or Sunday"),
]

# Order volume by hour (lunch and dinner peaks). Normalised in code.
_HOUR_WEIGHTS = np.array(
    [1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 6, 9, 8, 5, 4, 5, 7, 9, 9, 7, 5, 3, 2], dtype=float
)
# Noise scale chosen so that std(f_true) / RMS(noise sigma) ~= 3 (SNR ~= 3).
# Calibrated once on a 1M-row reference draw; fixed so the formula never changes across seeds.
_NOISE_SCALE = 2.47


def is_rush_hour(hour: np.ndarray) -> np.ndarray:
    hour = np.asarray(hour)
    return ((hour >= 8) & (hour <= 10)) | ((hour >= 17) & (hour <= 20))


def delivery_true_function(X: pd.DataFrame) -> np.ndarray:
    """Noise-free expected delivery time in minutes.

    NEVER shown to any LLM. Documented in the README for readers.

    Main effects, two interactions (rain x traffic, rush-hour x distance),
    a saturating experience effect and a threshold on order size.
    """
    d = X["distance_km"].to_numpy(float)
    hour = X["hour"].to_numpy()
    rain = X["rain_mm"].to_numpy(float)
    traffic = X["traffic_index"].to_numpy(float)
    exp_yrs = X["driver_experience_yrs"].to_numpy(float)
    order = X["order_size"].to_numpy(float)
    prep = X["prep_time_min"].to_numpy(float)
    weekend = X["is_weekend"].to_numpy(float)
    rush = is_rush_hour(hour).astype(float)

    return (
        8.0
        + 2.5 * d
        + 0.6 * prep
        + 1.2 * traffic
        + 0.15 * rain
        + 0.06 * rain * traffic            # interaction 1: rain x traffic
        + 0.9 * d * rush                   # interaction 2: hour x distance
        - 6.0 * (1.0 - np.exp(-exp_yrs / 3.0))
        + 0.5 * order
        + 5.0 * (order > 5)                # threshold: big orders need a second bag
        - 1.5 * weekend
    )


def delivery_noise_sigma(X: pd.DataFrame) -> np.ndarray:
    """Heteroscedastic noise: longer trips are less predictable."""
    return _NOISE_SCALE * (0.5 + X["distance_km"].to_numpy(float) / 5.0)


def generate_delivery(n_rows: int = 20_000, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    p_hour = _HOUR_WEIGHTS / _HOUR_WEIGHTS.sum()

    hour = rng.choice(24, size=n_rows, p=p_hour)
    rush = is_rush_hour(hour)
    distance = np.clip(rng.gamma(2.0, 2.5, n_rows), 0.3, 30.0)
    rain = np.where(rng.random(n_rows) < 0.3, np.clip(rng.exponential(6.0, n_rows), 0, 60), 0.0)
    traffic = np.clip(3.0 + 3.5 * rush + rng.normal(0, 1.5, n_rows), 0, 10)
    experience = np.clip(rng.exponential(3.0, n_rows), 0, 20)
    order = np.clip(1 + rng.poisson(2.5, n_rows), 1, 20)
    prep = np.clip(rng.normal(15.0, 5.0, n_rows), 3, 45)
    weekend = (rng.random(n_rows) < 2 / 7).astype(int)

    X = pd.DataFrame(
        {
            "distance_km": np.round(distance, 2),
            "hour": hour.astype(int),
            "rain_mm": np.round(rain, 1),
            "traffic_index": np.round(traffic, 2),
            "driver_experience_yrs": np.round(experience, 1),
            "order_size": order.astype(int),
            "prep_time_min": np.round(prep, 1),
            "is_weekend": weekend,
        }
    )
    f_true = delivery_true_function(X)
    y = f_true + rng.normal(0.0, 1.0, n_rows) * delivery_noise_sigma(X)
    return Dataset(
        name="delivery",
        X=X,
        y=y,
        target_name="delivery_time_min",
        target_unit="minutes",
        features=list(DELIVERY_FEATURES),
        f_true=f_true,
    )


# --------------------------------------------------------------------------- #
# D2: California Housing
# --------------------------------------------------------------------------- #
CALIFORNIA_FEATURES = [
    FeatureSpec("MedInc", "tens of thousands USD", "Median household income in the block group"),
    FeatureSpec("HouseAge", "years", "Median house age in the block group"),
    FeatureSpec("AveRooms", "rooms per household", "Average number of rooms per household"),
    FeatureSpec("AveBedrms", "bedrooms per household", "Average number of bedrooms per household"),
    FeatureSpec("Population", "people", "Block group population"),
    FeatureSpec("AveOccup", "people per household", "Average number of household members"),
    FeatureSpec("Latitude", "degrees", "Block group latitude"),
    FeatureSpec("Longitude", "degrees", "Block group longitude"),
]


def load_california(data_home: Optional[str] = None) -> Dataset:
    """sklearn's California Housing. Downloads once (~400 KB) into sklearn's data home."""
    from sklearn.datasets import fetch_california_housing

    bunch = fetch_california_housing(as_frame=True, data_home=data_home)
    X = bunch.data[[f.name for f in CALIFORNIA_FEATURES]].copy()
    y = bunch.target.to_numpy(float)
    return Dataset(
        name="california",
        X=X,
        y=y,
        target_name="MedHouseVal",
        target_unit="hundreds of thousands USD",
        features=list(CALIFORNIA_FEATURES),
    )


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def split_indices(
    n: int, seed: int, fractions: Optional[dict[str, float]] = None
) -> dict[str, np.ndarray]:
    """Random disjoint partition of range(n) into train / cal / val / test."""
    fractions = fractions or DEFAULT_FRACTIONS
    if set(fractions) != set(SPLIT_NAMES):
        raise ValueError(f"fractions must have keys {SPLIT_NAMES}, got {sorted(fractions)}")
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError("split fractions must sum to 1")

    # Separate stream from the D1 data draw so changing n_rows doesn't alias the two.
    rng = np.random.default_rng([seed, 0x5EED])
    perm = rng.permutation(n)
    out: dict[str, np.ndarray] = {}
    start = 0
    for i, name in enumerate(SPLIT_NAMES):
        size = n - start if i == len(SPLIT_NAMES) - 1 else int(round(fractions[name] * n))
        out[name] = np.sort(perm[start : start + size])
        start += size
    return out


def split_dataset(ds: Dataset, seed: int, fractions: Optional[dict[str, float]] = None) -> SplitData:
    idx = split_indices(len(ds.y), seed, fractions)
    splits = {
        name: Split(
            X=ds.X.iloc[ix].reset_index(drop=True),
            y=ds.y[ix],
            idx=ix,
            f_true=None if ds.f_true is None else ds.f_true[ix],
        )
        for name, ix in idx.items()
    }
    return SplitData(
        name=ds.name,
        seed=seed,
        target_name=ds.target_name,
        target_unit=ds.target_unit,
        features=ds.features,
        splits=splits,
    )


def load_dataset(
    name: str,
    seed: int,
    n_rows: Optional[int] = None,
    fractions: Optional[dict[str, float]] = None,
) -> SplitData:
    """Load and split a dataset for one seed. ``name`` is 'delivery' or 'california'."""
    if name == "delivery":
        ds = generate_delivery(n_rows or 20_000, seed=seed)
    elif name == "california":
        ds = load_california()
        if n_rows is not None and n_rows != len(ds.y):
            raise ValueError("california has a fixed size; leave n_rows unset")
    else:
        raise ValueError(f"unknown dataset {name!r}")
    return split_dataset(ds, seed, fractions)


# --------------------------------------------------------------------------- #
# Anonymisation (ablation, PLAN §4)
# --------------------------------------------------------------------------- #
def anon_mapping(feature_names: list[str], seed: int) -> dict[str, str]:
    """Map real feature names to f0..fN in a per-seed shuffled order.

    The model itself is untouched: the same booster is rendered with these names,
    so `named` vs `anon` differ only in what the LLM reads.
    """
    rng = np.random.default_rng([seed, 0xA707])
    order = rng.permutation(len(feature_names))
    return {feature_names[j]: f"f{k}" for k, j in enumerate(order)}
