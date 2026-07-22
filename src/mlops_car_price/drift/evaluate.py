"""Judge the drift detector the way a test is judged: how often it is wrong, and when.

A monitor is a classifier with two error rates, and shipping one without measuring them is
how alerting becomes noise. Three detectors are compared on identical snapshots, so the
differences between them are not sampling luck:

``fixed``
    the textbook rule — PSI over 0.2, or normalised shift over 0.1.
``calibrated``
    the same thresholds, but a column must also clear what it scores against itself at this
    sample size (ADR 0006).
``p_value``
    flag when any column's KS test rejects at the configured level. The arm that exists to
    be beaten: a p-value answers "could this be chance?", and with enough rows the answer is
    always no, however small and irrelevant the shift.

Three things get measured:

*false alarm rate* on unshifted weeks — the number that decides whether anyone still reads
the alerts after a month; *power* as a function of shift size — how big a change has to be
before it is noticed; and *sensitivity to sample size* at a fixed, deliberately trivial
shift, which is where the p-value detector falls apart.

Every trial computes the metrics once and lets each detector interpret them, so the
comparison is paired.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from mlops_car_price import replay
from mlops_car_price.config import Config
from mlops_car_price.drift import detector, metrics
from mlops_car_price.drift.metrics import FeatureDrift

DETECTORS = ("fixed", "calibrated", "p_value")
# Trials start well past the weeks the demo scripts write, so an evaluation never reuses a
# snapshot that already appears in a report.
FIRST_EVALUATION_WEEK = 1_000


class FloorCache:
    """Noise floors keyed by snapshot size, computed at most once each.

    A floor depends only on the reference column and ``n``. Recomputing it per scenario, per
    magnitude and per trial is the difference between a script that runs in minutes and one
    nobody waits for.
    """

    def __init__(self, config: Config, reference: pd.DataFrame) -> None:
        self._config = config
        self._reference = reference
        self._cache: dict[int, dict[str, tuple[float, float]]] = {}

    def get(self, n_rows: int) -> dict[str, tuple[float, float]]:
        if n_rows not in self._cache:
            self._cache[n_rows] = metrics.noise_floors(
                self._reference,
                n_rows,
                detector.NUMERIC_FEATURES,
                detector.CATEGORICAL_FEATURES,
                self._config.drift.calibration_resamples,
                self._config.drift.calibration_quantile,
                self._config.seed,
            )
        return self._cache[n_rows]


@dataclass(frozen=True)
class DetectionRates:
    """How often each detector fired, over the same set of trials."""

    scenario: str
    magnitude: float
    n_rows: int
    trials: int
    rates: dict[str, float]

    def rate(self, detector_name: str) -> float:
        return self.rates[detector_name]


def flags(config: Config, drifts: tuple[FeatureDrift, ...], detector_name: str) -> bool:
    """Would this detector call these measurements drift?"""
    if detector_name == "p_value":
        return any(
            drift.p_value is not None and drift.p_value < config.drift.ks_p_value_threshold
            for drift in drifts
        )
    if detector_name == "fixed":
        drifts = tuple(
            replace(drift, psi_noise_floor=None, shift_noise_floor=None) for drift in drifts
        )
    elif detector_name != "calibrated":
        raise KeyError(f"unknown detector {detector_name!r}; expected one of {DETECTORS}")
    return bool(detector.feature_reasons(config, drifts))


def measure_trials(
    config: Config,
    reference: pd.DataFrame,
    scenario: str,
    magnitude: float | None,
    n_rows: int,
    trials: int,
    first_week: int = FIRST_EVALUATION_WEEK,
    floors: dict[str, tuple[float, float]] | None = None,
    cache: FloorCache | None = None,
) -> DetectionRates:
    """Run one scenario ``trials`` times and record what each detector did.

    The noise floors are computed once for this sample size, because a floor is a property
    of the reference column and ``n``, not of the week being judged.
    """
    if floors is None:
        cache = cache or FloorCache(config, reference)
        floors = cache.get(n_rows)

    fired = dict.fromkeys(DETECTORS, 0)
    for trial in range(trials):
        snapshot = replay.make_snapshot(
            config, first_week + trial, scenario, magnitude, n_rows
        )
        drifts = metrics.compare(
            reference,
            snapshot,
            detector.NUMERIC_FEATURES,
            detector.CATEGORICAL_FEATURES,
            floors=floors,
        )
        for name in DETECTORS:
            fired[name] += int(flags(config, drifts, name))

    return DetectionRates(
        scenario=scenario,
        magnitude=magnitude if magnitude is not None else 0.0,
        n_rows=n_rows,
        trials=trials,
        rates={name: fired[name] / trials for name in DETECTORS},
    )


def false_alarm_rates(
    config: Config,
    reference: pd.DataFrame,
    n_rows: int,
    trials: int,
    cache: FloorCache | None = None,
) -> DetectionRates:
    """How often each detector cries wolf on a week where nothing happened."""
    return measure_trials(config, reference, "stable", 0.0, n_rows, trials, cache=cache)


def power_curve(
    config: Config,
    reference: pd.DataFrame,
    scenario: str,
    magnitudes: tuple[float, ...],
    n_rows: int,
    trials: int,
    cache: FloorCache | None = None,
) -> list[DetectionRates]:
    """Detection rate as a function of how large the shift actually is."""
    cache = cache or FloorCache(config, reference)
    floors = cache.get(n_rows)
    return [
        measure_trials(
            config,
            reference,
            scenario,
            magnitude,
            n_rows,
            trials,
            first_week=FIRST_EVALUATION_WEEK + 100 * index,
            floors=floors,
        )
        for index, magnitude in enumerate(magnitudes)
    ]


def sample_size_sweep(
    config: Config,
    reference: pd.DataFrame,
    scenario: str,
    magnitude: float,
    sizes: tuple[int, ...],
    trials: int,
    cache: FloorCache | None = None,
) -> list[DetectionRates]:
    """The same negligible shift, judged at growing sample sizes.

    The shift never changes; only ``n`` does. A detector reading effect sizes should be flat
    here, and one reading p-values should climb to certainty about a difference nobody would
    act on.
    """
    cache = cache or FloorCache(config, reference)
    return [
        measure_trials(
            config,
            reference,
            scenario,
            magnitude,
            size,
            trials,
            first_week=FIRST_EVALUATION_WEEK + 200 * index,
            cache=cache,
        )
        for index, size in enumerate(sizes)
    ]
