"""MLflow wiring — the one place that knows the tracking URI, experiment and client.

Kept separate from the training and registry code so those modules never repeat the
setup, and so a change of backend (SQLite file today, a tracking server in Docker later)
touches exactly one module.
"""

from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient

from mlops_car_price.config import Config


def configure(config: Config) -> str:
    """Point the MLflow fluent API at the configured backend; returns the URI used."""
    uri = config.mlflow.resolve_tracking_uri()
    mlflow.set_tracking_uri(uri)
    return uri


def client(config: Config) -> MlflowClient:
    """A client bound to the configured backend, independent of global fluent state."""
    return MlflowClient(tracking_uri=config.mlflow.resolve_tracking_uri())


def ensure_experiment(config: Config) -> str:
    """Return the experiment id, creating it with an explicit artifact root if needed.

    MLflow resolves its default artifact location against the working directory, which
    means running the same command from another directory would quietly start a second
    artifact store. The root comes from the config instead.
    """
    configure(config)
    mlflow_client = client(config)
    experiment = mlflow_client.get_experiment_by_name(config.mlflow.experiment)
    if experiment is None:
        config.mlflow.artifact_root.mkdir(parents=True, exist_ok=True)
        experiment_id = mlflow_client.create_experiment(
            config.mlflow.experiment,
            artifact_location=config.mlflow.artifact_root.as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id
