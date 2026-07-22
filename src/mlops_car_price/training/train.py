"""Training entrypoint: fit one model on ``train_initial``, score it on the frozen holdout.

Every training in this project goes through this module — never a notebook, never an ad-hoc
script — because a run that is not recorded cannot be compared, and comparison is the whole
point of the MLOps layer. Each run records the parameters, the holdout metrics, the exact
dataset (the manifest hash) and the version of the modelling package it borrowed the model
from, so any number in the README can be traced back to a run.

    python -m mlops_car_price.training.train --model LightGBM

Evaluation is a single frozen holdout, not cross-validation: a promotion decision must
compare champion and challenger on identical rows, and A3 already answers the "which model
family wins offline" question with CV.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from car_price_ml import features as a3_features
from car_price_ml import model as a3_model

from mlops_car_price import dataset, registry, tracking
from mlops_car_price.config import Config
from mlops_car_price.config import load as load_config

_BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class RunResult:
    """What a training run produced — returned so tests and callers need no MLflow queries."""

    run_id: str
    model_name: str
    metrics: dict[str, float]
    model_path: Path
    model_size_mb: float
    train_seconds: float
    dataset_hash: str
    registered_version: str | None = None


def _sample(frame: pd.DataFrame, sample_rows: int | None, seed: int) -> pd.DataFrame:
    """Optionally shrink the training frame (CI runs minutes, not hours)."""
    if sample_rows is None or sample_rows >= len(frame):
        return frame
    return frame.sample(n=sample_rows, random_state=seed).reset_index(drop=True)


def train_run(
    config: Config,
    model_name: str | None = None,
    sample_rows: int | None = None,
    run_name: str | None = None,
    log_model_artifact: bool = False,
    register: bool = False,
) -> RunResult:
    """Fit, evaluate on the frozen holdout, and record everything as one MLflow run.

    Args:
        model_name: One of the A3 bake-off models; defaults to ``training.default_model``.
        sample_rows: Rows drawn from ``train_initial``; defaults to ``training.sample_rows``.
        log_model_artifact: Copy the fitted model into MLflow's artifact store. Off by
            default — a RandomForest on this dataset serialises to hundreds of megabytes,
            so a run history of them would be measured in gigabytes (ADR 0003).
        register: Enter the run's model into the registry as a new version and label it
            ``challenger``. Implies ``log_model_artifact``: an unstored model cannot be a
            version. Exploration stays out of the registry; only candidates go in.
    """
    name = model_name or config.training.default_model
    known = tuple(a3_model.build_models().keys())
    if name not in known:
        raise ValueError(f"unknown model {name!r}; expected one of {known}")

    effective_sample = sample_rows if sample_rows is not None else config.training.sample_rows
    manifest = dataset.read_manifest(config)
    data_hash = dataset.dataset_hash(config)

    train_split = dataset.load_split("train_initial", config)
    train_frame = _sample(train_split, effective_sample, config.seed)
    holdout_frame = dataset.load_split("holdout_eval", config)
    x_train, y_train = a3_features.prepare(train_frame)
    x_holdout, y_holdout = a3_features.prepare(holdout_frame)

    tracking.ensure_experiment(config)

    with mlflow.start_run(run_name=run_name) as run:
        started = time.perf_counter()
        fitted = a3_model.train(x_train, y_train, name=name, random_state=config.seed)
        train_seconds = time.perf_counter() - started

        metrics = a3_model.evaluate(y_holdout, fitted.predict(x_holdout))

        run_models_dir = config.paths.models_dir / run.info.run_id
        model_path = a3_model.save_model(
            fitted,
            metadata={
                "model": name,
                "holdout_metrics": metrics,
                "n_train": len(x_train),
                "dataset_hash": data_hash,
                "features": list(a3_features.FEATURE_COLUMNS),
            },
            models_dir=run_models_dir,
        )
        model_size_mb = model_path.stat().st_size / _BYTES_PER_MB

        mlflow.log_params(
            {
                "model": name,
                "seed": config.seed,
                "n_train": len(x_train),
                "n_holdout": len(x_holdout),
                "sample_rows": effective_sample,
                "features": ",".join(a3_features.FEATURE_COLUMNS),
            }
        )
        mlflow.log_metrics(
            {
                **metrics,
                "train_seconds": round(train_seconds, 2),
                # Logged for every run because artifact size is a deployment constraint
                # here, not a footnote: it decides what a retraining loop can carry.
                "model_size_mb": round(model_size_mb, 2),
            }
        )
        mlflow.set_tags(
            {
                "dataset_hash": data_hash,
                "car_price_ml_version": manifest["car_price_ml_version"],
                "mlops_car_price_version": version("mlops-car-price"),
                "holdout": "frozen",
            }
        )
        if log_model_artifact or register:
            mlflow.sklearn.log_model(sk_model=fitted, artifact_path="model")

        run_id = run.info.run_id

    # Registration happens after the run is closed: a version points at a finished run.
    registered_version = None
    if register:
        # Not named `version`: that would shadow the module-level importlib helper for the
        # whole function, including the tag written above.
        model_version = registry.register_run(config, run_id)
        registry.set_alias(config, registry.CHALLENGER, model_version.version)
        registered_version = model_version.version

    return RunResult(
        run_id=run_id,
        model_name=name,
        metrics=metrics,
        model_path=model_path,
        model_size_mb=model_size_mb,
        train_seconds=train_seconds,
        dataset_hash=data_hash,
        registered_version=registered_version,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=None, help="model name (default: from config)")
    parser.add_argument(
        "--sample-rows", type=int, default=None, help="rows drawn from train_initial"
    )
    parser.add_argument("--run-name", default=None, help="MLflow run name")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument(
        "--log-model",
        action="store_true",
        help="also store the fitted model in MLflow's artifact store (large for RandomForest)",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="register the model as a new version and label it challenger",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config) if args.config else load_config()
    result = train_run(
        config,
        model_name=args.model,
        sample_rows=args.sample_rows,
        run_name=args.run_name,
        log_model_artifact=args.log_model,
        register=args.register,
    )

    print(f"[train] model:   {result.model_name}")
    print(f"[train] run:     {result.run_id}")
    print(f"[train] data:    {result.dataset_hash[:12]}")
    print("[train] holdout: " + "  ".join(f"{k}={v}" for k, v in result.metrics.items()))
    print(
        f"[train] cost:    {result.train_seconds:.1f}s train, "
        f"{result.model_size_mb:.1f} MB artifact"
    )
    if result.registered_version:
        print(f"[train] version: {result.registered_version} (alias: challenger)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
