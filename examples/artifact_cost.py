"""Measure what each candidate model costs to *operate*, not just how well it scores.

Offline accuracy is one column of the decision. A model that has to be retrained weekly,
stored in a registry version after version, pulled into a container image and loaded on
every service start also costs disk, bandwidth and startup time — and those columns decide
what a maintenance loop can actually carry.

    python examples/artifact_cost.py [--latency-samples 200] [--output reports/artifact_cost.md]

Regenerates the cost table in the README and in ADR 0003. Numbers come from this machine;
the ratios are what transfer, not the absolute milliseconds.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from car_price_ml import features as a3_features
from car_price_ml import model as a3_model

from mlops_car_price import dataset
from mlops_car_price.config import Config
from mlops_car_price.config import load as load_config

MODELS = ("Ridge", "LightGBM", "RandomForest")
DEFAULT_LATENCY_SAMPLES = 200
RETRAINS_PER_YEAR = 52  # the weekly loop this project is built around
_BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class CostProfile:
    """Everything about a model that a maintenance loop pays for, per model."""

    name: str
    mae: float
    train_seconds: float
    artifact_mb: float
    load_seconds: float
    latency_p50_ms: float
    latency_p95_ms: float
    batch_rows_per_second: float

    @property
    def registry_gb_per_year(self) -> float:
        """Storage a weekly retraining loop accumulates in a year of model versions."""
        return self.artifact_mb * RETRAINS_PER_YEAR / 1024


def _percentile_ms(samples: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(samples) * 1_000, percentile))


def measure(
    name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    models_dir: Path,
    seed: int,
    latency_samples: int,
) -> CostProfile:
    """Train once, then time the operations a served model actually performs."""
    started = time.perf_counter()
    fitted = a3_model.train(x_train, y_train, name=name, random_state=seed)
    train_seconds = time.perf_counter() - started

    metrics = a3_model.evaluate(y_holdout, fitted.predict(x_holdout))

    artifact_dir = models_dir / f"cost-{name}"
    path = a3_model.save_model(fitted, metadata={"model": name}, models_dir=artifact_dir)
    artifact_mb = path.stat().st_size / _BYTES_PER_MB

    started = time.perf_counter()
    reloaded = a3_model.load_model(models_dir=artifact_dir)["model"]
    load_seconds = time.perf_counter() - started

    # Single-row latency is what an API request pays; the batch figure is what a scoring
    # job pays. They are not the same number and the ratio differs sharply per model.
    single_row = x_holdout.iloc[[0]]
    durations = []
    for _ in range(latency_samples):
        started = time.perf_counter()
        reloaded.predict(single_row)
        durations.append(time.perf_counter() - started)

    started = time.perf_counter()
    reloaded.predict(x_holdout)
    batch_seconds = time.perf_counter() - started

    return CostProfile(
        name=name,
        mae=metrics["mae"],
        train_seconds=train_seconds,
        artifact_mb=artifact_mb,
        load_seconds=load_seconds,
        latency_p50_ms=_percentile_ms(durations, 50),
        latency_p95_ms=_percentile_ms(durations, 95),
        batch_rows_per_second=len(x_holdout) / batch_seconds,
    )


def render(profiles: list[CostProfile], n_train: int, n_holdout: int) -> str:
    """A markdown table — the artefact this script exists to produce."""
    header = (
        "| Model | Holdout MAE (PLN) | Train | Artifact | Load | Predict p50 | Predict p95 "
        "| Batch | Registry / year |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = "".join(
        f"| {p.name} | {p.mae:,.0f} | {p.train_seconds:.1f} s | {p.artifact_mb:,.1f} MB "
        f"| {p.load_seconds:.2f} s | {p.latency_p50_ms:.1f} ms | {p.latency_p95_ms:.1f} ms "
        f"| {p.batch_rows_per_second:,.0f} rows/s | {p.registry_gb_per_year:,.1f} GB |\n"
        for p in profiles
    )
    footer = (
        f"\nTrained on {n_train:,} rows, scored on the frozen holdout of {n_holdout:,} rows. "
        f"Registry column projects {RETRAINS_PER_YEAR} retrainings a year, one version each.\n"
    )
    return header + rows + footer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--latency-samples", type=int, default=DEFAULT_LATENCY_SAMPLES)
    parser.add_argument("--output", default="reports/artifact_cost.md")
    args = parser.parse_args(argv)

    config: Config = load_config(args.config) if args.config else load_config()
    x_train, y_train = a3_features.prepare(dataset.load_split("train_initial", config))
    x_holdout, y_holdout = a3_features.prepare(dataset.load_split("holdout_eval", config))

    profiles = []
    for name in MODELS:
        print(f"[cost] measuring {name} ...", flush=True)
        profiles.append(
            measure(
                name,
                x_train,
                y_train,
                x_holdout,
                y_holdout,
                models_dir=config.paths.models_dir,
                seed=config.seed,
                latency_samples=args.latency_samples,
            )
        )

    table = render(profiles, n_train=len(x_train), n_holdout=len(x_holdout))
    output = config.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"[cost] written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
