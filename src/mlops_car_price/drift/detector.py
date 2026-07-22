"""Turn distribution distances into a verdict, on three signals that fail independently.

**Feature drift** — did the inputs move? Cheap, available immediately, and blind to
anything that happens on the target side.

**Prediction drift** — did the model's output distribution move? Catches an input shift the
per-column view can miss (an interaction between columns), and needs no labels.

**Realised error** — is the model actually worse? The only signal that sees a shift in the
target itself, and in a real system the one that arrives last, because labels are late.
Here labels come with the replayed traffic, which is exactly why a price shock is worth
replaying: it moves prices while leaving every feature and every prediction untouched, so
the first two signals stay silent and only this one fires.

Gates run on effect sizes (PSI, normalised Wasserstein) and never on p-values: on samples
this size a KS test rejects shifts far too small to act on. The p-value is carried in the
report as a diagnostic so the claim can be checked rather than believed.

    python -m mlops_car_price.drift.detector --week 1 --scenario fuel_mix_shift
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass

import pandas as pd
from car_price_ml import config as a3_config
from car_price_ml import features as a3_features
from car_price_ml import model as a3_model

from mlops_car_price import dataset, registry, replay
from mlops_car_price.config import Config
from mlops_car_price.config import load as load_config
from mlops_car_price.drift import metrics
from mlops_car_price.drift.metrics import FeatureDrift

NUMERIC_FEATURES = a3_config.NUMERIC_FEATURES
CATEGORICAL_FEATURES = (*a3_config.HIGH_CARD_CATEGORICAL, *a3_config.LOW_CARD_CATEGORICAL)
PREDICTION_COLUMN = "prediction"


@dataclass(frozen=True)
class DriftReport:
    """Everything measured about one snapshot, and the verdict that follows from it."""

    label: str
    judged: bool
    drifted: bool
    reasons: tuple[str, ...]
    features: tuple[FeatureDrift, ...]
    flagged_features: tuple[str, ...]
    prediction: FeatureDrift | None
    current_mae: float | None
    reference_mae: float | None
    n_current: int

    @property
    def mae_increase_pct(self) -> float | None:
        """How much worse the model is on this week than on the frozen holdout."""
        if self.current_mae is None or not self.reference_mae:
            return None
        return (self.current_mae - self.reference_mae) / self.reference_mae * 100.0


def effective_thresholds(config: Config, drift: FeatureDrift) -> tuple[float, float]:
    """The bar a column has to clear: the larger of "matters" and "more than noise".

    The configured thresholds say what size of shift is worth acting on. The noise floor,
    measured per column at the snapshot's own sample size, says what that column scores
    against itself when nothing happened. A signal has to beat both — using only the first
    alarms on the control case, using only the second alarms on shifts too small to care
    about.
    """
    psi_floor = drift.psi_noise_floor or 0.0
    shift_floor = drift.shift_noise_floor or 0.0
    return (
        max(config.drift.psi_threshold, psi_floor),
        max(config.drift.wasserstein_threshold, shift_floor),
    )


def feature_reasons(config: Config, drifts: tuple[FeatureDrift, ...]) -> list[tuple[str, str]]:
    """Reasons a column is considered drifted, as (feature, reason) pairs.

    The normalised-shift rule applies to **numeric columns only**: it is a Wasserstein
    distance in units of the reference standard deviation, while the categorical figure
    reported next to it is a total variation distance on a different scale. Categorical
    columns are judged on PSI alone.
    """
    found: list[tuple[str, str]] = []
    for drift in drifts:
        if drift.missing_rate_increase > config.drift.max_missing_rate_increase:
            found.append(
                (
                    drift.feature,
                    f"feature '{drift.feature}': missing rate rose by "
                    f"{drift.missing_rate_increase:.1%} (limit "
                    f"{config.drift.max_missing_rate_increase:.1%})",
                )
            )
            continue

        psi_bar, shift_bar = effective_thresholds(config, drift)
        if drift.psi > psi_bar:
            found.append(
                (
                    drift.feature,
                    f"feature '{drift.feature}': PSI {drift.psi:.3f} over {psi_bar:.3f}",
                )
            )
        elif drift.kind == "numeric" and drift.normalised_shift > shift_bar:
            found.append(
                (
                    drift.feature,
                    f"feature '{drift.feature}': normalised shift "
                    f"{drift.normalised_shift:.3f} over {shift_bar:.3f}",
                )
            )
    return found


def _prediction_reasons(config: Config, drift: FeatureDrift) -> list[str]:
    """The same effect-size rules, applied to the model's own output distribution."""
    psi_bar, shift_bar = effective_thresholds(config, drift)
    if drift.psi > psi_bar:
        return [f"predictions: PSI {drift.psi:.3f} over {psi_bar:.3f}"]
    if drift.normalised_shift > shift_bar:
        return [f"predictions: normalised shift {drift.normalised_shift:.3f} over {shift_bar:.3f}"]
    return []


def analyse(
    config: Config,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    label: str,
    model=None,
    reference_mae: float | None = None,
) -> DriftReport:
    """Compare a snapshot with the training distribution. Pure apart from model inference."""
    if len(current) < config.drift.min_snapshot_rows:
        return DriftReport(
            label=label,
            judged=False,
            drifted=False,
            reasons=(
                f"snapshot has {len(current)} rows, fewer than the required "
                f"{config.drift.min_snapshot_rows} - not judged",
            ),
            features=(),
            flagged_features=(),
            prediction=None,
            current_mae=None,
            reference_mae=reference_mae,
            n_current=len(current),
        )

    features = metrics.compare(
        reference,
        current,
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
        calibration_resamples=config.drift.calibration_resamples,
        calibration_quantile=config.drift.calibration_quantile,
        seed=config.seed,
    )
    flagged = feature_reasons(config, features)
    reasons = [reason for _, reason in flagged]

    prediction_drift: FeatureDrift | None = None
    current_mae: float | None = None
    if model is not None:
        columns = list(a3_features.FEATURE_COLUMNS)
        reference_predictions = pd.Series(model.predict(reference[columns]))
        current_predictions = pd.Series(model.predict(current[columns]))
        prediction_drift = metrics.numeric_drift(
            PREDICTION_COLUMN, reference_predictions, current_predictions
        )
        if config.drift.calibration_resamples > 0:
            psi_floor, shift_floor = metrics.noise_floor(
                reference_predictions,
                len(current),
                "numeric",
                config.drift.calibration_resamples,
                config.drift.calibration_quantile,
                config.seed,
            )
            prediction_drift = dataclasses.replace(
                prediction_drift, psi_noise_floor=psi_floor, shift_noise_floor=shift_floor
            )
        reasons.extend(_prediction_reasons(config, prediction_drift))

        if a3_config.TARGET in current.columns:
            current_mae = a3_model.evaluate(current[a3_config.TARGET], current_predictions)["mae"]

    if current_mae is not None and reference_mae:
        increase = (current_mae - reference_mae) / reference_mae * 100.0
        if increase > config.drift.max_mae_increase_pct:
            reasons.append(
                f"error: MAE up {increase:.1f}% on the week "
                f"(limit {config.drift.max_mae_increase_pct:.1f}%)"
            )

    return DriftReport(
        label=label,
        judged=True,
        drifted=bool(reasons),
        reasons=tuple(reasons),
        features=features,
        flagged_features=tuple(name for name, _ in flagged),
        prediction=prediction_drift,
        current_mae=current_mae,
        reference_mae=reference_mae,
        n_current=len(current),
    )


def analyse_snapshot(
    config: Config, week: int, scenario: str = "stable", use_champion: bool = True
) -> DriftReport:
    """Load a stored snapshot and judge it against the training distribution."""
    reference = dataset.load_split("train_initial", config)
    current = replay.load_snapshot(config, week, scenario)

    model = None
    reference_mae = None
    if use_champion:
        champion = registry.alias_version(config, registry.CHAMPION)
        if champion is not None:
            model = registry.load_version(config, champion.version)
            reference_mae = registry.version_run(config, champion).data.metrics.get("mae")

    return analyse(
        config, reference, current, f"week {week} / {scenario}", model, reference_mae
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--week", type=int, default=1)
    parser.add_argument("--scenario", default="stable", choices=sorted(replay.SCENARIOS))
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    report = analyse_snapshot(config, args.week, args.scenario)

    from mlops_car_price.drift import report as rendering

    print(rendering.render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
