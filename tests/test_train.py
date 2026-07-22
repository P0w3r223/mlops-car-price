"""A training run is only useful if it is recorded — and recorded completely."""

from __future__ import annotations

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from mlops_car_price import dataset
from mlops_car_price.config import Config
from mlops_car_price.training import train

REQUIRED_TAGS = ("dataset_hash", "car_price_ml_version", "mlops_car_price_version")
REQUIRED_METRICS = ("mae", "rmse", "mape", "r2", "train_seconds", "model_size_mb")


@pytest.fixture
def built(project: Config) -> Config:
    dataset.build(project)
    return project


def test_run_reports_holdout_metrics_and_persists_the_model(built: Config):
    result = train.train_run(built, model_name="LightGBM")

    assert result.metrics["mae"] > 0
    assert 0.0 <= result.metrics["r2"] <= 1.0
    assert result.model_path.exists()
    assert result.model_size_mb > 0
    assert result.dataset_hash == dataset.dataset_hash(built)


def test_run_records_parameters_metrics_and_provenance_tags(built: Config):
    result = train.train_run(built, model_name="LightGBM", run_name="unit-test")

    mlflow.set_tracking_uri(built.mlflow.resolve_tracking_uri())
    run = MlflowClient().get_run(result.run_id)

    assert run.data.params["model"] == "LightGBM"
    assert int(run.data.params["n_holdout"]) > 0
    for metric in REQUIRED_METRICS:
        assert metric in run.data.metrics
    for tag in REQUIRED_TAGS:
        assert run.data.tags[tag]
    # The dataset hash is what makes two runs comparable at all.
    assert run.data.tags["dataset_hash"] == result.dataset_hash


def test_sample_rows_limits_the_training_frame(built: Config):
    result = train.train_run(built, model_name="LightGBM", sample_rows=120)

    mlflow.set_tracking_uri(built.mlflow.resolve_tracking_uri())
    run = MlflowClient().get_run(result.run_id)

    assert int(run.data.params["n_train"]) == 120


def test_each_run_keeps_its_own_model_directory(built: Config):
    first = train.train_run(built, model_name="LightGBM", sample_rows=120)
    second = train.train_run(built, model_name="LightGBM", sample_rows=140)

    assert first.model_path != second.model_path
    assert first.model_path.exists() and second.model_path.exists()


def test_unknown_model_is_rejected_before_any_work_happens(built: Config):
    with pytest.raises(ValueError, match="unknown model"):
        train.train_run(built, model_name="XGBoost")


def test_training_without_a_dataset_says_what_to_run(project: Config):
    with pytest.raises(FileNotFoundError, match="dataset build"):
        train.train_run(project, model_name="LightGBM")
