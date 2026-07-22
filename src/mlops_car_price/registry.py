"""The model registry: which version exists, which one serves, and where it came from.

Three words that are easy to confuse and worth keeping straight:

*run*
    one execution of training. Holds parameters, metrics and artifacts. Immutable history.
*model version*
    an artifact from a run, entered into the registry as version 1, 2, 3 … of a named
    model. A run only becomes a version when someone decides it is a candidate.
*alias*
    a movable label on a version — ``champion`` is what serves, ``challenger`` is what is
    being considered. Serving code asks for ``models:/car-price@champion`` and never needs
    to know a version number. Aliases replaced the deprecated stage mechanism in MLflow 2.9.

The provenance tags of the originating run (dataset hash, package versions) are copied onto
the version, so a deployed model can be traced without joining back to the run table.
"""

from __future__ import annotations

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from mlops_car_price import tracking
from mlops_car_price.config import Config

CHAMPION = "champion"
CHALLENGER = "challenger"

_PROVENANCE_TAGS = ("dataset_hash", "car_price_ml_version", "mlops_car_price_version")


def model_uri(config: Config, alias: str = CHAMPION) -> str:
    """The URI serving code should use — an alias, never a version number."""
    return f"models:/{config.mlflow.registered_model}@{alias}"


def register_run(config: Config, run_id: str) -> ModelVersion:
    """Enter a run's logged model into the registry as a new version.

    The run must have been trained with ``--register`` (or ``log_model_artifact=True``);
    a run without a stored artifact cannot become a version.
    """
    tracking.configure(config)
    name = config.mlflow.registered_model
    run = tracking.client(config).get_run(run_id)

    version = mlflow.register_model(f"runs:/{run_id}/model", name)

    mlflow_client = tracking.client(config)
    for tag in _PROVENANCE_TAGS:
        if tag in run.data.tags:
            mlflow_client.set_model_version_tag(name, version.version, tag, run.data.tags[tag])
    mlflow_client.set_model_version_tag(name, version.version, "model", run.data.params["model"])
    return mlflow_client.get_model_version(name, version.version)


def set_alias(config: Config, alias: str, version: str | int) -> None:
    """Move an alias onto a version. This is what "deploying" means here."""
    tracking.client(config).set_registered_model_alias(
        config.mlflow.registered_model, alias, str(version)
    )


def alias_version(config: Config, alias: str) -> ModelVersion | None:
    """The version an alias points at, or None when the alias was never set."""
    try:
        return tracking.client(config).get_model_version_by_alias(
            config.mlflow.registered_model, alias
        )
    except MlflowException:
        # Absent alias and absent registered model both surface as RESOURCE_DOES_NOT_EXIST;
        # for callers the answer is the same — there is nothing serving yet.
        return None


def get_version(config: Config, version: str | int) -> ModelVersion:
    return tracking.client(config).get_model_version(config.mlflow.registered_model, str(version))


def load_version(config: Config, version: str | int):
    """Load the fitted estimator behind a version number."""
    tracking.configure(config)
    return mlflow.sklearn.load_model(f"models:/{config.mlflow.registered_model}/{version}")


def load_alias(config: Config, alias: str = CHAMPION):
    """Load whatever the alias currently points at."""
    tracking.configure(config)
    return mlflow.sklearn.load_model(model_uri(config, alias))


def version_run(config: Config, version: ModelVersion):
    """The training run a version came from — for its metrics and parameters."""
    return tracking.client(config).get_run(version.run_id)
