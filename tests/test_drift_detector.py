"""The verdict layer: which signal fires for which failure, and what must never fire."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from mlops_car_price import dataset, registry, replay
from mlops_car_price.config import Config
from mlops_car_price.drift import detector, metrics, report
from mlops_car_price.training import promote, train

SNAPSHOT_ROWS = 200


@pytest.fixture
def monitored(project: Config, reconfigure) -> Config:
    """A built project with thresholds sized for a 600-row synthetic dataset."""
    config = reconfigure(project, "drift", min_snapshot_rows=50)
    config = reconfigure(config, "promotion", min_holdout_rows=50)
    config = reconfigure(config, "replay", snapshot_rows=SNAPSHOT_ROWS)
    dataset.build(config)
    return config


@pytest.fixture
def with_champion(monitored: Config) -> Config:
    result = train.train_run(monitored, model_name="LightGBM", register=True)
    promote.promote(monitored, result.registered_version)
    return monitored


def analyse(config: Config, scenario: str, week: int = 1, magnitude: float | None = None):
    reference = dataset.load_split("train_initial", config)
    snapshot = replay.make_snapshot(config, week, scenario, magnitude, SNAPSHOT_ROWS)
    return detector.analyse(config, reference, snapshot, f"test/{scenario}")


def test_an_unshifted_snapshot_raises_no_alarm(monitored: Config):
    """The control case. A monitor that cries wolf here is worse than no monitor."""
    verdict = analyse(monitored, "stable")

    assert verdict.judged
    assert not verdict.drifted, verdict.reasons


def test_a_high_cardinality_column_does_not_alarm_on_sampling_noise(monitored: Config):
    """Regression guard for the failure that fixed thresholds produce.

    200 categories sampled at 500 rows score PSI ~0.36 against their own source - well past
    the textbook 0.2, and entirely noise. The calibrated floor has to absorb that.
    """
    reference = pd.Series([f"m{i % 200}" for i in range(20_000)], name="mark")
    current = reference.sample(500, random_state=7)

    raw = metrics.categorical_drift("mark", reference, current)
    psi_floor, shift_floor = metrics.noise_floor(
        reference, len(current), "categorical", resamples=40, quantile=0.99, seed=1
    )
    calibrated = dataclasses.replace(
        raw, psi_noise_floor=psi_floor, shift_noise_floor=shift_floor
    )

    # The trap is real: judged against the fixed threshold this column looks drifted ...
    assert raw.psi > monitored.drift.psi_threshold
    assert detector.feature_reasons(monitored, (raw,)) != []
    # ... and against its own noise floor it does not.
    assert detector.feature_reasons(monitored, (calibrated,)) == []


def test_a_shift_in_one_numeric_feature_is_localised(monitored: Config):
    verdict = analyse(monitored, "mileage_shift", week=2, magnitude=1.5)

    assert verdict.drifted
    assert "mileage" in verdict.flagged_features


def test_a_field_that_arrives_empty_is_caught_by_the_missing_rate(monitored: Config):
    verdict = analyse(monitored, "missing_engine_volume", week=3, magnitude=0.5)

    assert verdict.drifted
    assert "vol_engine" in verdict.flagged_features
    assert any("missing rate" in reason for reason in verdict.reasons)


def test_unseen_categories_are_caught(monitored: Config):
    verdict = analyse(monitored, "unseen_makes", week=4, magnitude=0.5)

    assert verdict.drifted
    assert "mark" in verdict.flagged_features


def test_a_snapshot_too_small_to_judge_is_not_judged(monitored: Config):
    reference = dataset.load_split("train_initial", monitored)
    snapshot = replay.make_snapshot(monitored, week=5, n_rows=10)

    verdict = detector.analyse(monitored, reference, snapshot, "tiny")

    assert not verdict.judged
    assert not verdict.drifted
    assert "not judged" in verdict.reasons[0]


def test_a_price_shock_is_invisible_to_features_and_predictions(with_champion: Config):
    """Only the realised error sees a shift that moves the target and nothing else."""
    reference = dataset.load_split("train_initial", with_champion)
    snapshot = replay.make_snapshot(with_champion, 6, "price_shock", 0.4, SNAPSHOT_ROWS)
    champion = registry.alias_version(with_champion, registry.CHAMPION)
    model = registry.load_version(with_champion, champion.version)
    reference_mae = registry.version_run(with_champion, champion).data.metrics["mae"]

    verdict = detector.analyse(
        with_champion, reference, snapshot, "shock", model, reference_mae
    )

    assert verdict.flagged_features == ()
    assert not any(reason.startswith("predictions:") for reason in verdict.reasons)
    assert verdict.drifted
    assert any(reason.startswith("error:") for reason in verdict.reasons)
    assert verdict.mae_increase_pct > 10.0


def test_the_error_signal_needs_both_labels_and_a_baseline(monitored: Config):
    """Without a champion there is no reference error, so no error verdict is claimed."""
    verdict = analyse(monitored, "price_shock", week=7, magnitude=0.4)

    assert verdict.current_mae is None
    assert verdict.mae_increase_pct is None


def test_the_report_renders_the_verdict_and_the_diagnostics(monitored: Config):
    verdict = analyse(monitored, "mileage_shift", week=8, magnitude=1.5)

    rendered = report.render(verdict)

    assert "# Drift report" in rendered
    assert "**Verdict: DRIFT**" in rendered
    assert "mileage" in rendered
    assert "never a gate" in rendered


def test_p_values_are_reported_but_never_decide(monitored: Config):
    """Every feature carries a p-value; none of the reasons is phrased in terms of one."""
    verdict = analyse(monitored, "mileage_shift", week=9, magnitude=1.5)

    assert any(feature.p_value is not None for feature in verdict.features)
    assert not any("p-value" in reason or "p=" in reason for reason in verdict.reasons)


def test_analysing_a_stored_snapshot_uses_the_champion(with_champion: Config):
    replay.write_snapshot(with_champion, week=11, scenario="stable", n_rows=SNAPSHOT_ROWS)

    verdict = detector.analyse_snapshot(with_champion, week=11, scenario="stable")

    assert verdict.judged
    assert verdict.prediction is not None
    assert verdict.current_mae is not None
    assert np.isfinite(verdict.current_mae)
