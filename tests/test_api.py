"""The serving contract: what it answers, what it refuses, and what it says when empty."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mlops_car_price import dataset, registry
from mlops_car_price.api.main import MODEL_VERSION_HEADER, build_app
from mlops_car_price.config import Config
from mlops_car_price.training import promote, train

VALID_CAR = {
    "mark": "opel",
    "model": "opel-1",
    "fuel": "Diesel",
    "province": "Mazowieckie",
    "year": 2015,
    "mileage": 139_568,
    "vol_engine": 1_248,
}


@pytest.fixture
def served(project: Config, reconfigure) -> Config:
    config = reconfigure(project, "promotion", min_holdout_rows=50)
    dataset.build(config)
    result = train.train_run(config, model_name="LightGBM", register=True)
    promote.promote(config, result.registered_version)
    return config


@pytest.fixture
def client(served: Config) -> TestClient:
    with TestClient(build_app(served)) as test_client:
        yield test_client


@pytest.fixture
def empty_client(project: Config) -> TestClient:
    dataset.build(project)
    with TestClient(build_app(project)) as test_client:
        yield test_client


def test_a_valuation_comes_back_with_the_version_that_made_it(client: TestClient):
    response = client.post("/predict", json=VALID_CAR)

    assert response.status_code == 200
    body = response.json()
    assert body["price_pln"] > 0
    assert response.headers[MODEL_VERSION_HEADER] == body["model_version"]


def test_model_info_answers_what_is_running_right_now(client: TestClient, served: Config):
    response = client.get("/model-info")

    body = response.json()
    champion = registry.alias_version(served, registry.CHAMPION)
    assert body["version"] == str(champion.version)
    assert body["model_name"] == "LightGBM"
    assert body["holdout_mae"] > 0
    assert body["dataset_hash"] == dataset.dataset_hash(served)
    assert body["trained_at"]


def test_health_reports_the_serving_version(client: TestClient):
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["model_version"]


def test_a_service_without_a_champion_starts_and_says_so(empty_client: TestClient):
    """Degraded, not dead: an orchestrator can route around it and a human can ask why."""
    body = empty_client.get("/health").json()

    assert body["status"] == "degraded"
    assert "no champion" in body["reason"]


def test_predicting_without_a_champion_is_unavailable_not_broken(empty_client: TestClient):
    response = empty_client.post("/predict", json=VALID_CAR)

    assert response.status_code == 503
    assert "promote a model version" in response.json()["detail"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fuel", "Plutonium"),
        ("year", 1800),
        ("year", 2100),
        ("mileage", -1),
        ("vol_engine", 99_999),
    ],
)
def test_nonsense_input_is_rejected_rather_than_priced(client: TestClient, field, value):
    """A valuation for an impossible car is worse than no valuation."""
    response = client.post("/predict", json={**VALID_CAR, field: value})

    assert response.status_code == 422


def test_a_missing_field_is_rejected(client: TestClient):
    payload = {key: value for key, value in VALID_CAR.items() if key != "mileage"}

    assert client.post("/predict", json=payload).status_code == 422


def test_an_unknown_make_is_still_priced(client: TestClient):
    """The target encoder falls back for unseen categories; the service must not 500."""
    response = client.post("/predict", json={**VALID_CAR, "mark": "delorean"})

    assert response.status_code == 200
    assert response.json()["price_pln"] > 0


def test_serving_follows_the_alias_rather_than_a_version_number(served: Config):
    """Promoting a different version changes what the next process serves - no code change."""
    first = str(registry.alias_version(served, registry.CHAMPION).version)
    challenger = train.train_run(served, model_name="RandomForest", register=True)
    registry.set_alias(served, registry.CHAMPION, challenger.registered_version)

    with TestClient(build_app(served)) as moved:
        body = moved.get("/model-info").json()

    assert body["version"] != first
    assert body["model_name"] == "RandomForest"
