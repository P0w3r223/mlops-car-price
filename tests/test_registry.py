"""Registration, aliases, and the provenance that has to travel with a version."""

from __future__ import annotations

import pytest

from mlops_car_price import dataset, registry
from mlops_car_price.config import Config
from mlops_car_price.training import train


@pytest.fixture
def built(project: Config) -> Config:
    dataset.build(project)
    return project


def test_registering_a_run_carries_its_provenance(built: Config):
    result = train.train_run(built, model_name="LightGBM", sample_rows=200, register=True)

    version = registry.get_version(built, result.registered_version)

    assert version.run_id == result.run_id
    assert version.tags["dataset_hash"] == result.dataset_hash
    assert version.tags["model"] == "LightGBM"
    assert version.tags["car_price_ml_version"]


def test_a_registered_run_is_labelled_challenger_not_champion(built: Config):
    result = train.train_run(built, model_name="LightGBM", sample_rows=200, register=True)

    challenger = registry.alias_version(built, registry.CHALLENGER)

    assert challenger.version == result.registered_version
    # Registering is proposing, not deploying.
    assert registry.alias_version(built, registry.CHAMPION) is None


def test_versions_increment_and_the_alias_follows_the_newest(built: Config):
    first = train.train_run(built, model_name="LightGBM", sample_rows=200, register=True)
    second = train.train_run(built, model_name="LightGBM", sample_rows=240, register=True)

    assert int(second.registered_version) == int(first.registered_version) + 1
    assert registry.alias_version(built, registry.CHALLENGER).version == second.registered_version


def test_an_unset_alias_reads_as_nothing_serving(built: Config):
    assert registry.alias_version(built, registry.CHAMPION) is None


def test_a_version_can_be_loaded_back_and_predicts(built: Config):
    result = train.train_run(built, model_name="LightGBM", sample_rows=200, register=True)
    holdout = dataset.load_split("holdout_eval", built)

    fitted = registry.load_version(built, result.registered_version)
    from car_price_ml import features as a3_features

    x, _ = a3_features.prepare(holdout)

    assert len(fitted.predict(x)) == len(holdout)


def test_serving_uri_points_at_an_alias_not_a_version(built: Config):
    assert registry.model_uri(built) == f"models:/{built.mlflow.registered_model}@champion"
