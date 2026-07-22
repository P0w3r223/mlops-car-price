"""Promotion rules — first as pure decisions, then as their effect on the registry."""

from __future__ import annotations

import numpy as np
import pytest

from mlops_car_price import dataset, registry
from mlops_car_price.config import Config
from mlops_car_price.training import promote, train

HOLDOUT_ROWS_FOR_TESTS = 50


def scored(
    version: str,
    mae: float,
    logged_mae: float | None = None,
    data_hash: str = "abc123",
    artifact_mb: float = 3.3,
):
    """A stand-in for a re-scored registry version; errors are irrelevant to the rules today."""
    return promote.Scored(
        version=version,
        model_name="LightGBM",
        mae=mae,
        logged_mae=mae if logged_mae is None else logged_mae,
        dataset_hash=data_hash,
        artifact_mb=artifact_mb,
        absolute_errors=np.full(10, mae),
    )


@pytest.fixture
def small_holdout(project: Config, reconfigure) -> Config:
    return reconfigure(project, "promotion", min_holdout_rows=HOLDOUT_ROWS_FOR_TESTS)


@pytest.fixture
def built(small_holdout: Config) -> Config:
    dataset.build(small_holdout)
    return small_holdout


def test_first_candidate_wins_by_default(small_holdout: Config):
    decision = promote.decide(small_holdout, scored("1", 9000.0), None, n_holdout=1_000)

    assert decision.promoted
    assert "no champion" in decision.reasons[0]
    assert decision.delta_mae is None


def test_a_clear_improvement_is_promoted(small_holdout: Config):
    decision = promote.decide(small_holdout, scored("2", 8_500.0), scored("1", 9_000.0), 1_000)

    assert decision.promoted
    assert decision.delta_mae == pytest.approx(500.0)


def test_an_improvement_below_the_margin_is_refused(small_holdout: Config):
    # 40 PLN better, against a configured margin of 100 PLN.
    decision = promote.decide(small_holdout, scored("2", 8_960.0), scored("1", 9_000.0), 1_000)

    assert not decision.promoted
    assert "short of the required" in decision.reasons[0]


def test_a_worse_candidate_is_refused(small_holdout: Config):
    decision = promote.decide(small_holdout, scored("2", 9_500.0), scored("1", 9_000.0), 1_000)

    assert not decision.promoted


def test_a_holdout_too_small_to_judge_blocks_the_decision(small_holdout: Config):
    decision = promote.decide(small_holdout, scored("2", 1.0), scored("1", 9_000.0), n_holdout=10)

    assert not decision.promoted
    assert "fewer than the required" in decision.reasons[0]


def test_an_artifact_that_does_not_reproduce_its_metrics_is_refused(small_holdout: Config):
    candidate = scored("2", mae=8_000.0, logged_mae=8_500.0)

    decision = promote.decide(small_holdout, candidate, scored("1", 9_000.0), 1_000)

    assert not decision.promoted
    assert "does not reproduce" in decision.reasons[0]


def test_models_scored_on_different_data_versions_are_not_compared(small_holdout: Config):
    candidate = scored("2", 8_000.0, data_hash="ffffff")
    champion = scored("1", 9_000.0, data_hash="aaaaaa")

    decision = promote.decide(small_holdout, candidate, champion, 1_000)

    assert not decision.promoted
    assert "rebuild before comparing" in decision.reasons[0]


def test_a_model_over_the_deployment_budget_is_refused_however_good_it_is(small_holdout: Config):
    """Accuracy does not buy its way past the operational limit — ADR 0003 in code."""
    candidate = scored("2", 5_000.0, artifact_mb=338.5)

    decision = promote.decide(small_holdout, candidate, scored("1", 9_000.0), 1_000)

    assert not decision.promoted
    assert "deployment budget" in decision.reasons[0]


def test_the_budget_also_blocks_a_first_champion(small_holdout: Config):
    decision = promote.decide(small_holdout, scored("1", 5_000.0, artifact_mb=338.5), None, 1_000)

    assert not decision.promoted


def test_the_first_registered_version_takes_the_champion_alias(built: Config):
    result = train.train_run(built, model_name="LightGBM", sample_rows=200, register=True)

    decision = promote.promote(built, result.registered_version)

    assert decision.promoted
    assert registry.alias_version(built, registry.CHAMPION).version == result.registered_version


def test_a_weaker_challenger_does_not_take_over(built: Config):
    champion = train.train_run(built, model_name="LightGBM", register=True)
    promote.promote(built, champion.registered_version)

    weaker = train.train_run(built, model_name="LightGBM", sample_rows=40, register=True)
    decision = promote.promote(built, weaker.registered_version)

    assert not decision.promoted
    assert registry.alias_version(built, registry.CHAMPION).version == champion.registered_version
    # A refusal stays inspectable rather than being erased.
    assert registry.alias_version(built, registry.CHALLENGER).version == weaker.registered_version


def test_a_dry_run_decides_without_touching_the_aliases(built: Config):
    result = train.train_run(built, model_name="LightGBM", sample_rows=200, register=True)

    decision = promote.promote(built, result.registered_version, dry_run=True)

    assert decision.promoted
    assert registry.alias_version(built, registry.CHAMPION) is None


def test_the_verdict_is_written_onto_the_version(built: Config):
    result = train.train_run(built, model_name="LightGBM", sample_rows=200, register=True)

    promote.promote(built, result.registered_version)
    version = registry.get_version(built, result.registered_version)

    assert version.tags["promotion"] == "accepted"
    assert version.tags["promotion_reason"]


def test_scoring_recomputes_metrics_and_keeps_per_row_errors(built: Config):
    result = train.train_run(built, model_name="LightGBM", sample_rows=200, register=True)
    decision = promote.evaluate_candidate(built, result.registered_version)

    # The errors are what session 5's paired bootstrap will consume.
    assert len(decision.candidate.absolute_errors) == decision.n_holdout
    assert decision.candidate.mae == pytest.approx(result.metrics["mae"], abs=0.1)
