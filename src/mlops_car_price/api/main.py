"""Serving the champion — whichever version that currently is.

The service never names a model version. It resolves ``models:/car-price@champion`` at
startup, and a promotion elsewhere becomes a deployment here by restarting the process. That
indirection is the point of the registry: nothing in this file changes when the model does.

Three endpoints, one of which exists purely so the system can be interrogated:

``POST /predict``
    a valuation, with the serving version echoed in the ``X-Model-Version`` header so a
    prediction can be traced to the model that made it.
``GET /model-info``
    which version is serving, what it scored on the frozen holdout, which data it was trained
    against, and why it was promoted. The answer to "what is running right now?".
``GET /health``
    liveness and readiness. A service with no champion starts anyway and reports itself
    **degraded** rather than crash-looping (ADR 0008).

    uvicorn mlops_car_price.api.main:app
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import pandas as pd
from car_price_ml import config as a3_config
from car_price_ml import data as a3_data
from car_price_ml import features as a3_features
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field, field_validator

from mlops_car_price import registry
from mlops_car_price.config import Config
from mlops_car_price.config import load as load_config

MODEL_VERSION_HEADER = "X-Model-Version"
# Years the model was actually trained on; beyond that it would be extrapolating silently.
_MIN_YEAR = a3_config.REFERENCE_YEAR - a3_config.AGE_MAX


@dataclass
class ServedModel:
    """The champion as the service sees it: the estimator plus its provenance."""

    estimator: Any
    version: str
    model_name: str
    holdout_mae: float | None
    dataset_hash: str | None
    car_price_ml_version: str | None
    promotion_reason: str | None
    trained_at: str | None


class CarFeatures(BaseModel):
    """A valuation request.

    The contract takes ``year`` because that is what a person knows about a car; the service
    derives ``age`` exactly the way training did, using the same code.
    """

    mark: str = Field(max_length=40, examples=["opel"])
    model: str = Field(max_length=60, examples=["combo"])
    fuel: str = Field(examples=["Diesel"])
    province: str = Field(max_length=60, examples=["Mazowieckie"])
    year: int = Field(ge=_MIN_YEAR, le=a3_config.REFERENCE_YEAR, examples=[2015])
    mileage: int = Field(ge=0, le=int(a3_config.MILEAGE_MAX), examples=[139568])
    vol_engine: int = Field(ge=0, le=int(a3_config.VOL_ENGINE_MAX), examples=[1248])

    @field_validator("fuel")
    @classmethod
    def fuel_must_be_known(cls, value: str) -> str:
        if value not in a3_config.KNOWN_FUELS:
            raise ValueError(f"fuel must be one of {a3_config.KNOWN_FUELS}, got {value!r}")
        return value


class Prediction(BaseModel):
    price_pln: float
    model_version: str
    model_name: str


class ModelInfo(BaseModel):
    version: str
    model_name: str
    holdout_mae: float | None
    dataset_hash: str | None
    car_price_ml_version: str | None
    promotion_reason: str | None
    trained_at: str | None


def load_champion(config: Config) -> ServedModel | None:
    """Resolve the champion alias into a served model, or None when nothing is promoted."""
    version = registry.alias_version(config, registry.CHAMPION)
    if version is None:
        return None

    run = registry.version_run(config, version)
    started = run.info.start_time
    return ServedModel(
        estimator=registry.load_version(config, version.version),
        # Coerced at the boundary: MLflow hands back a version that is not always a string,
        # and an HTTP header that is not a string fails deep inside the server.
        version=str(version.version),
        model_name=run.data.params.get("model", "unknown"),
        holdout_mae=run.data.metrics.get("mae"),
        dataset_hash=run.data.tags.get("dataset_hash"),
        car_price_ml_version=run.data.tags.get("car_price_ml_version"),
        promotion_reason=version.tags.get("promotion_reason"),
        trained_at=pd.Timestamp(started, unit="ms", tz="UTC").isoformat() if started else None,
    )


def build_app(config: Config | None = None) -> FastAPI:
    """Assemble the service. Takes a config so tests can point it at their own registry."""
    settings = config or load_config()
    state: dict[str, ServedModel | None] = {"champion": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Loading at startup rather than per request: a request should not pay for
        # deserialising a model, and a service that cannot serve should say so immediately.
        state["champion"] = load_champion(settings)
        yield

    app = FastAPI(
        title="mlops-car-price",
        description="Used-car valuation served from the model registry's champion alias.",
        version="2.0",
        lifespan=lifespan,
    )

    def champion() -> ServedModel:
        served = state["champion"]
        if served is None:
            raise HTTPException(
                status_code=503,
                detail="no champion in the registry - promote a model version first",
            )
        return served

    @app.get("/health")
    def health() -> dict:
        """Ready only when a champion is actually loaded; degraded is not an error."""
        served = state["champion"]
        if served is None:
            return {"status": "degraded", "reason": "no champion in the registry"}
        return {"status": "ok", "model_version": served.version}

    @app.get("/model-info", response_model=ModelInfo)
    def model_info() -> ModelInfo:
        served = champion()
        return ModelInfo(
            version=served.version,
            model_name=served.model_name,
            holdout_mae=served.holdout_mae,
            dataset_hash=served.dataset_hash,
            car_price_ml_version=served.car_price_ml_version,
            promotion_reason=served.promotion_reason,
            trained_at=served.trained_at,
        )

    @app.post("/predict", response_model=Prediction)
    def predict(features: CarFeatures, response: Response) -> Prediction:
        served = champion()
        frame = a3_data.add_age(pd.DataFrame([features.model_dump()]))
        price = float(served.estimator.predict(frame[list(a3_features.FEATURE_COLUMNS)])[0])

        # The version travels with the answer, so a logged prediction can be traced to the
        # model that produced it even after the alias has moved on.
        response.headers[MODEL_VERSION_HEADER] = served.version
        return Prediction(
            price_pln=round(price, 2), model_version=served.version, model_name=served.model_name
        )

    return app


app = build_app()
