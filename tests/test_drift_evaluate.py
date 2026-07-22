"""The harness that judges the detector — and the claims it exists to support."""

from __future__ import annotations

import dataclasses

import pytest

from mlops_car_price import dataset, replay
from mlops_car_price.config import Config
from mlops_car_price.drift import detector, evaluate, metrics

TRIALS = 30
SNAPSHOT_ROWS = 200


@pytest.fixture
def built(project: Config, reconfigure) -> Config:
    config = reconfigure(project, "drift", min_snapshot_rows=50, calibration_resamples=100)
    dataset.build(config)
    return config


@pytest.fixture
def reference(built: Config):
    return dataset.load_split("train_initial", built)


def drifts_for(config: Config, reference, scenario: str, magnitude: float, n_rows: int):
    snapshot = replay.make_snapshot(config, 500, scenario, magnitude, n_rows)
    floors = metrics.noise_floors(
        reference,
        n_rows,
        detector.NUMERIC_FEATURES,
        detector.CATEGORICAL_FEATURES,
        config.drift.calibration_resamples,
        config.drift.calibration_quantile,
        config.seed,
    )
    return metrics.compare(
        reference,
        snapshot,
        detector.NUMERIC_FEATURES,
        detector.CATEGORICAL_FEATURES,
        floors=floors,
    )


def test_the_three_detectors_read_the_same_measurements(built: Config, reference):
    measured = drifts_for(built, reference, "mileage_shift", 1.5, SNAPSHOT_ROWS)

    verdicts = {name: evaluate.flags(built, measured, name) for name in evaluate.DETECTORS}

    # A shift this large is not subtle; every detector should see it.
    assert all(verdicts.values()), verdicts


def test_an_unknown_detector_is_rejected(built: Config, reference):
    measured = drifts_for(built, reference, "stable", 0.0, SNAPSHOT_ROWS)

    with pytest.raises(KeyError, match="unknown detector"):
        evaluate.flags(built, measured, "vibes")


def test_the_fixed_detector_ignores_the_noise_floor(built: Config, reference):
    """The two effect-size detectors differ only in whether the floor is consulted."""
    measured = drifts_for(built, reference, "stable", 0.0, SNAPSHOT_ROWS)
    inflated = tuple(
        dataclasses.replace(m, psi_noise_floor=99.0, shift_noise_floor=99.0) for m in measured
    )

    # An absurd floor silences the calibrated detector and leaves the fixed one untouched.
    assert not evaluate.flags(built, inflated, "calibrated")
    assert evaluate.flags(built, inflated, "fixed") == evaluate.flags(built, measured, "fixed")


def test_power_rises_with_the_size_of_the_shift(built: Config, reference):
    curve = evaluate.power_curve(
        built, reference, "mileage_shift", (0.05, 0.5, 1.5), SNAPSHOT_ROWS, TRIALS
    )

    rates = [point.rate("calibrated") for point in curve]

    assert rates[0] <= rates[1] <= rates[2]
    assert rates[-1] > 0.9


def test_the_calibrated_detector_stays_quiet_on_unshifted_weeks(built: Config, reference):
    rates = evaluate.false_alarm_rates(built, reference, SNAPSHOT_ROWS, TRIALS)

    # The whole point of calibration. Loose bound: 30 trials cannot resolve 1%.
    assert rates.rate("calibrated") <= 0.2


def test_the_sweep_reports_one_result_per_sample_size(built: Config, reference):
    """Plumbing only.

    The statistical claim this sweep exists to demonstrate - that a p-value detector grows
    more certain about an unchanged, negligible shift as `n` rises - needs sample sizes a
    600-row fixture cannot produce. It is asserted directly on distributions in
    `test_drift_metrics.py`, and measured on the real 70 715-row reference in
    `reports/detector_evaluation.md`. Faking it here would only test the fixture.
    """
    sizes = (100, 300)

    sweep = evaluate.sample_size_sweep(built, reference, "mileage_shift", 0.05, sizes, TRIALS)

    assert [point.n_rows for point in sweep] == list(sizes)
    assert all(set(point.rates) == set(evaluate.DETECTORS) for point in sweep)
    assert all(0.0 <= point.rate(name) <= 1.0 for point in sweep for name in evaluate.DETECTORS)


def test_floors_are_computed_once_per_sample_size(built: Config, reference):
    cache = evaluate.FloorCache(built, reference)

    first = cache.get(SNAPSHOT_ROWS)
    again = cache.get(SNAPSHOT_ROWS)

    assert first is again
    assert set(first) == set(detector.NUMERIC_FEATURES) | set(detector.CATEGORICAL_FEATURES)


def test_the_report_level_tail_is_split_across_columns():
    """A report fires if any column fires, so each column is held to a tighter bar."""
    assert metrics.per_column_quantile(0.99, 1) == pytest.approx(0.99)
    assert metrics.per_column_quantile(0.99, 10) == pytest.approx(0.999)


def test_every_trial_is_a_different_week(built: Config, reference):
    """Repeated trials must be independent draws, not one snapshot counted many times."""
    first_week, trials = 700, 3
    floors = metrics.noise_floors(
        reference,
        SNAPSHOT_ROWS,
        detector.NUMERIC_FEATURES,
        detector.CATEGORICAL_FEATURES,
        built.drift.calibration_resamples,
        built.drift.calibration_quantile,
        built.seed,
    )

    measured = evaluate.measure_trials(
        built, reference, "mileage_shift", 0.15, SNAPSHOT_ROWS, trials, first_week, floors
    )

    by_hand = 0
    for offset in range(trials):
        snapshot = replay.make_snapshot(
            built, first_week + offset, "mileage_shift", 0.15, SNAPSHOT_ROWS
        )
        drifts = metrics.compare(
            reference,
            snapshot,
            detector.NUMERIC_FEATURES,
            detector.CATEGORICAL_FEATURES,
            floors=floors,
        )
        by_hand += int(evaluate.flags(built, drifts, "calibrated"))

    assert measured.rate("calibrated") == pytest.approx(by_hand / trials)
