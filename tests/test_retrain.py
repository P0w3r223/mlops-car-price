"""The statistical gate and the maintenance loop, including the case where it does nothing."""

from __future__ import annotations

import numpy as np
import pytest

from mlops_car_price import dataset, registry, replay
from mlops_car_price.config import Config
from mlops_car_price.training import promote, retrain, train

SNAPSHOT_ROWS = 200
HOLDOUT_ROWS_FOR_TESTS = 50


@pytest.fixture
def looped(project: Config, reconfigure) -> Config:
    config = reconfigure(project, "promotion", min_holdout_rows=HOLDOUT_ROWS_FOR_TESTS)
    config = reconfigure(config, "drift", min_snapshot_rows=50)
    config = reconfigure(config, "replay", snapshot_rows=SNAPSHOT_ROWS)
    dataset.build(config)
    return config


@pytest.fixture
def with_champion(looped: Config) -> Config:
    result = train.train_run(looped, model_name="LightGBM", register=True)
    promote.promote(looped, result.registered_version)
    return looped


def scored(version: str, errors: np.ndarray, mae: float | None = None) -> promote.Scored:
    return promote.Scored(
        version=version,
        model_name="LightGBM",
        mae=float(np.mean(errors)) if mae is None else mae,
        logged_mae=float(np.mean(errors)) if mae is None else mae,
        dataset_hash="abc123",
        artifact_mb=3.3,
        absolute_errors=errors,
    )


def test_a_real_improvement_survives_the_bootstrap(looped: Config):
    """Errors that are lower on almost every row are not a sampling accident."""
    rng = np.random.default_rng(1)
    champion_errors = np.abs(rng.normal(9_000, 5_000, 4_000))
    candidate_errors = champion_errors * 0.9

    decision = promote.decide(
        looped, scored("2", candidate_errors), scored("1", champion_errors), 4_000
    )

    assert decision.promoted
    assert decision.evidence.ci.low > 0
    assert any("holds at 95% confidence" in reason for reason in decision.reasons)


def test_an_improvement_indistinguishable_from_noise_is_refused(looped: Config):
    """The point estimate clears the margin; the interval says it could be either model."""
    rng = np.random.default_rng(2)
    champion_errors = np.abs(rng.normal(9_000, 12_000, 300))
    candidate_errors = np.abs(rng.normal(9_000, 12_000, 300))
    # Force a point improvement past the 100 PLN margin while the pairing is pure noise.
    candidate_errors = candidate_errors - 150.0

    decision = promote.decide(
        looped, scored("2", candidate_errors), scored("1", champion_errors), 300
    )

    assert not decision.promoted
    assert any("spans zero" in reason for reason in decision.reasons)
    assert decision.evidence is not None


def test_the_comparison_is_paired_not_independent(looped: Config):
    """Pairing is what makes a small, consistent improvement visible at all."""
    rng = np.random.default_rng(3)
    champion_errors = np.abs(rng.normal(9_000, 8_000, 2_000))
    candidate_errors = champion_errors - 200.0  # same rows, uniformly a little better

    evidence = promote.compare_errors(
        looped, scored("2", candidate_errors), scored("1", champion_errors)
    )

    assert evidence.estimate == pytest.approx(200.0, abs=1.0)
    # Between-car variance is huge next to a 200 PLN effect; only the pairing removes it.
    assert evidence.ci.low > 0
    assert any("same unit" in assumption for assumption in evidence.assumptions)


def test_a_candidate_that_fails_the_margin_never_reaches_the_bootstrap(with_champion: Config):
    """The cheap checks run first: 10 000 resamples are not spent on a settled question."""
    weaker = train.train_run(
        with_champion, model_name="LightGBM", sample_rows=300, register=True
    )

    decision = promote.promote(with_champion, weaker.registered_version)

    assert not decision.promoted
    assert any("short of the required" in reason for reason in decision.reasons)
    assert decision.evidence is None


def test_a_quiet_week_does_not_trigger_retraining(with_champion: Config):
    replay.write_snapshot(with_champion, week=1, scenario="stable", n_rows=SNAPSHOT_ROWS)

    outcome = retrain.run_cycle(with_champion, (1,), "stable")

    assert not outcome.retrained
    assert "no drift" in outcome.reason
    assert outcome.decision is None


def test_a_drifted_week_produces_a_challenger_that_the_gate_judges(with_champion: Config):
    replay.write_snapshot(
        with_champion, week=2, scenario="mileage_shift", magnitude=1.5, n_rows=SNAPSHOT_ROWS
    )

    outcome = retrain.run_cycle(with_champion, (2,), "mileage_shift")

    assert outcome.retrained
    assert outcome.challenger_version is not None
    assert outcome.decision is not None
    # Promoted or refused, the registry must hold a champion either way.
    assert registry.alias_version(with_champion, registry.CHAMPION) is not None


def test_a_dry_run_reports_drift_without_training(with_champion: Config):
    replay.write_snapshot(
        with_champion, week=3, scenario="mileage_shift", magnitude=1.5, n_rows=SNAPSHOT_ROWS
    )

    outcome = retrain.run_cycle(with_champion, (3,), "mileage_shift", dry_run=True)

    assert not outcome.retrained
    assert "dry run" in outcome.reason
    assert outcome.drifted_weeks


def test_force_retrains_a_quiet_week(with_champion: Config):
    replay.write_snapshot(with_champion, week=4, scenario="stable", n_rows=SNAPSHOT_ROWS)

    outcome = retrain.run_cycle(with_champion, (4,), "stable", force=True)

    assert outcome.retrained
    assert outcome.reason == "forced"


def test_the_challenger_trains_on_the_arrived_weeks_as_well(with_champion: Config):
    replay.write_snapshot(with_champion, week=5, scenario="stable", n_rows=SNAPSHOT_ROWS)
    original = dataset.load_split("train_initial", with_champion)

    extended = retrain.extended_training_frame(with_champion, (5,), "stable")

    assert len(extended) == len(original) + SNAPSHOT_ROWS


def test_the_frozen_holdout_never_enters_training(with_champion: Config):
    """The one invariant that makes every comparison in this project meaningful."""
    replay.write_snapshot(with_champion, week=6, scenario="stable", n_rows=SNAPSHOT_ROWS)
    holdout = dataset.load_split("holdout_eval", with_champion)

    extended = retrain.extended_training_frame(with_champion, (6,), "stable")

    key = ["mark", "model", "year", "mileage", "price"]
    merged = extended[key].merge(holdout[key], how="inner")
    assert merged.empty
