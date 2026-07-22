"""The simulated production stream: weekly snapshots, optionally shifted on purpose.

Fresh listings cannot be collected legally (ADR 0002), so "this week's traffic" is a draw
from ``stream_pool`` — the split that is never trained on and never evaluated on. Drawing
**with replacement** makes each week an independent sample from the same reserve, which is
what turns an unshifted snapshot into a proper null: any alarm raised on it is a false one.

A scenario is a named, parameterised transformation applied after the draw. Each one breaks
something different, and the differences are the point:

``stable``
    nothing changes. The control case, and the input for measuring false alarms.
``price_shock``
    prices move, features do not. **Invisible to every feature-drift metric** — only the
    prediction distribution and the realised error can see it. Included precisely because
    it is the case that embarrasses a feature-only drift dashboard.
``fuel_mix_shift``
    the fleet tilts toward electric and hybrid: a genuine covariate shift in one column.
``mileage_shift``
    cars arrive with more kilometres on them — a shift in a numeric feature the model
    leans on heavily.
``unseen_makes``
    makes the model has never seen. The target encoder has to fall back, and the
    categorical drift metric should notice.
``missing_engine_volume``
    a data-quality break rather than a distribution shift: the field arrives empty.

Magnitudes are arguments, not constants, because session 4 sweeps them to measure how large
a shift has to be before a detector notices.

    python -m mlops_car_price.replay --week 1 --scenario fuel_mix_shift --magnitude 0.4
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mlops_car_price import dataset
from mlops_car_price.config import Config
from mlops_car_price.config import load as load_config

ScenarioFn = Callable[[pd.DataFrame, float, np.random.Generator], pd.DataFrame]

ELECTRIFIED_FUELS = ("Electric", "Hybrid")
# Mileage shift is expressed in reference standard deviations so the same magnitude means
# a comparable amount of drift regardless of the column's units.
UNSEEN_MAKE_PREFIX = "unseen-"


@dataclass(frozen=True)
class Scenario:
    """A named way for production data to stop looking like the training data."""

    name: str
    description: str
    default_magnitude: float
    apply: ScenarioFn


def _stable(frame: pd.DataFrame, magnitude: float, rng: np.random.Generator) -> pd.DataFrame:
    return frame.copy()


def _price_shock(frame: pd.DataFrame, magnitude: float, rng: np.random.Generator) -> pd.DataFrame:
    """Inflate every price. Features are untouched, so feature drift cannot see this."""
    out = frame.copy()
    out["price"] = out["price"] * (1.0 + magnitude)
    return out


def _fuel_mix_shift(
    frame: pd.DataFrame, magnitude: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Resample rows so electric and hybrid cars make up a larger share of the week."""
    weights = np.where(frame["fuel"].isin(ELECTRIFIED_FUELS), 1.0 + magnitude * 20.0, 1.0)
    chosen = rng.choice(len(frame), size=len(frame), replace=True, p=weights / weights.sum())
    return frame.iloc[chosen].reset_index(drop=True)


def _mileage_shift(frame: pd.DataFrame, magnitude: float, rng: np.random.Generator) -> pd.DataFrame:
    """Add ``magnitude`` standard deviations of mileage to every car."""
    out = frame.copy()
    out["mileage"] = (out["mileage"] + magnitude * out["mileage"].std()).clip(lower=0)
    return out


def _unseen_makes(frame: pd.DataFrame, magnitude: float, rng: np.random.Generator) -> pd.DataFrame:
    """Relabel a share of rows with makes the model has never been trained on."""
    out = frame.copy()
    affected = rng.random(len(out)) < magnitude
    out.loc[affected, "mark"] = UNSEEN_MAKE_PREFIX + pd.Series(
        rng.integers(0, 3, affected.sum()).astype(str), index=out.index[affected]
    )
    return out


def _missing_engine_volume(
    frame: pd.DataFrame, magnitude: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Blank out engine volume for a share of rows — a broken field, not a moved one."""
    out = frame.copy()
    out.loc[rng.random(len(out)) < magnitude, "vol_engine"] = np.nan
    return out


SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in (
        Scenario("stable", "no change; the control case", 0.0, _stable),
        Scenario("price_shock", "prices inflate, features unchanged", 0.30, _price_shock),
        Scenario("fuel_mix_shift", "more electric and hybrid cars", 0.40, _fuel_mix_shift),
        Scenario("mileage_shift", "higher mileage across the week", 0.50, _mileage_shift),
        Scenario("unseen_makes", "makes absent from the training data", 0.20, _unseen_makes),
        Scenario(
            "missing_engine_volume", "engine volume arrives empty", 0.30, _missing_engine_volume
        ),
    )
}


def snapshot_seed(config: Config, week: int) -> int:
    """A per-week seed derived from the project seed, so any week is reproducible alone."""
    return config.seed * 1_000 + week


def make_snapshot(
    config: Config,
    week: int,
    scenario: str = "stable",
    magnitude: float | None = None,
    n_rows: int | None = None,
) -> pd.DataFrame:
    """Draw one week of production traffic and apply a scenario to it."""
    if scenario not in SCENARIOS:
        raise KeyError(f"unknown scenario {scenario!r}; expected one of {sorted(SCENARIOS)}")

    pool = dataset.load_split("stream_pool", config)
    rows = n_rows if n_rows is not None else config.replay.snapshot_rows
    rng = np.random.default_rng(snapshot_seed(config, week))

    drawn = pool.iloc[rng.integers(0, len(pool), rows)].reset_index(drop=True)
    chosen = SCENARIOS[scenario]
    strength = chosen.default_magnitude if magnitude is None else magnitude
    return chosen.apply(drawn, strength, rng)


def snapshot_path(config: Config, week: int, scenario: str) -> Path:
    return config.paths.snapshots_dir / f"week_{week:03d}_{scenario}.parquet"


def write_snapshot(
    config: Config,
    week: int,
    scenario: str = "stable",
    magnitude: float | None = None,
    n_rows: int | None = None,
) -> Path:
    """Persist a snapshot so a monitoring run and a retraining run see the same week."""
    frame = make_snapshot(config, week, scenario, magnitude, n_rows)
    config.paths.snapshots_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_path(config, week, scenario)
    frame.to_parquet(path, index=False)
    return path


def load_snapshot(config: Config, week: int, scenario: str = "stable") -> pd.DataFrame:
    path = snapshot_path(config, week, scenario)
    if not path.exists():
        raise FileNotFoundError(f"no snapshot at {path} - run `python -m mlops_car_price.replay`")
    return pd.read_parquet(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--week", type=int, default=1)
    parser.add_argument("--scenario", default="stable", choices=sorted(SCENARIOS))
    parser.add_argument("--magnitude", type=float, default=None)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args(argv)

    config = load_config(args.config) if args.config else load_config()
    path = write_snapshot(config, args.week, args.scenario, args.magnitude, args.rows)
    frame = load_snapshot(config, args.week, args.scenario)

    chosen = SCENARIOS[args.scenario]
    strength = chosen.default_magnitude if args.magnitude is None else args.magnitude
    print(
        f"[replay] week {args.week}: {args.scenario} "
        f"(magnitude {strength:g}) - {chosen.description}"
    )
    print(f"[replay] {len(frame):,} rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
