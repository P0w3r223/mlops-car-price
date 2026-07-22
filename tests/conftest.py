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

from mlops_car_price.config import DEFAULT_CONFIG_PATH, Config
from mlops_car_price.config import load as load_config

# Kept well above the 5 folds the target encoder cross-fits with, and large enough that a
# 20% holdout is still a meaningful frame.
SYNTHETIC_ROWS = 600

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
