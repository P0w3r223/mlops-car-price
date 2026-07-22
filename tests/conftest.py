"""Fixtures building a throwaway copy of the project around a synthetic dataset.

The temporary project uses the *real* ``configs/config.yaml`` rather than a hand-written
stand-in, so a key added to the shipped config without a matching loader change fails the
suite instead of failing on someone's machine.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mlops_car_price.config import DEFAULT_CONFIG_PATH, Config
from mlops_car_price.config import load as load_config

# Large enough that a weekly snapshot can be a small fraction of the stream reserve, the
# way it is on the real dataset (2 000 rows drawn from 23 573). Bootstrapping a snapshot
# that is a large share of the pool duplicates rows heavily, which inflates measured drift
# far past anything the real stream produces - a fixture artefact that reads as a bug.
SYNTHETIC_ROWS = 6_000

_MARKS = ("opel", "audi", "bmw", "ford", "volkswagen")
_FUELS = ("Gasoline", "Diesel", "Hybrid", "Electric", "LPG")
_PROVINCES = ("Mazowieckie", "Śląskie", "Wielkopolskie", "Małopolskie")


def synthetic_dataset(rows: int = SYNTHETIC_ROWS, seed: int = 7) -> pd.DataFrame:
    """A frame shaped like the Kaggle source, with a price that actually depends on it.

    Values stay inside the A3 cleaning rules (price, mileage, engine volume, age), so a
    build over this frame keeps all rows and the split sizes are predictable.
    """
    rng = np.random.default_rng(seed)
    year = rng.integers(1995, 2024, rows)
    mileage = rng.integers(1_000, 400_000, rows)
    vol_engine = rng.choice([0, 999, 1400, 1600, 1998, 2500, 3000], rows)
    mark = rng.choice(_MARKS, rows)

    age = 2024 - year
    price = 180_000 - age * 4_000 - mileage * 0.15 + vol_engine * 8
    price = np.clip(price + rng.normal(0, 5_000, rows), 2_000, 900_000)

    return pd.DataFrame(
        {
            "mark": mark,
            "model": [f"{m}-{i % 4}" for i, m in enumerate(mark)],
            "generation_name": rng.choice(["gen-a", "gen-b", None], rows),
            "year": year,
            "mileage": mileage,
            "vol_engine": vol_engine,
            "fuel": rng.choice(_FUELS, rows),
            "city": rng.choice(["Warszawa", "Katowice", "Poznań"], rows),
            "province": rng.choice(_PROVINCES, rows),
            "price": price.round(0),
        }
    )


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    """A temporary project root: real config, synthetic source CSV, isolated MLflow store."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    shutil.copy(DEFAULT_CONFIG_PATH, configs_dir / "config.yaml")

    config = load_config(configs_dir / "config.yaml")
    config.paths.raw_csv.parent.mkdir(parents=True, exist_ok=True)
    # index=True reproduces the unnamed index column of the real Kaggle export, which the
    # A3 loader is expected to drop.
    synthetic_dataset().to_csv(config.paths.raw_csv, index=True)

    # An ambient tracking URI would silently redirect runs to the developer's own store.
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    # MLflow resolves its default artifact root against the working directory.
    monkeypatch.chdir(tmp_path)
    return config


@pytest.fixture
def reconfigure():
    """Rewrite one section of a temporary project's config and reload it.

    Thresholds sized for 118k real rows (a 5 000-row minimum holdout, say) would reject
    everything on a 600-row synthetic frame; tests state the value they depend on instead
    of quietly working around it.
    """

    def apply(config: Config, section: str, **values) -> Config:
        path = config.root / "configs" / "config.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw[section].update(values)
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return load_config(path)

    return apply
