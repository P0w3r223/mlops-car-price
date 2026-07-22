"""The maintenance loop: notice, retrain, judge, and usually do nothing.

    python -m mlops_car_price.training.retrain --weeks 1 2 3 [--force] [--dry-run]

One pass of the cycle a scheduled job would run:

1. **look at the traffic** — judge the given weeks against the training distribution;
2. **stop unless something happened** — no drift means no retraining, because retraining on
   a quiet week burns compute and hands the promotion gate a coin flip to judge;
3. **train a challenger** on the original training data *plus* the weeks that have arrived,
   labels included;
4. **register it** as a candidate, never as the champion;
5. **let the gate decide** — and the expected outcome is refusal. A loop whose challengers
   are always promoted is not a quality bar, it is a deployment script with extra steps.

The whole pass is recorded: which weeks were examined, what drifted, what the challenger
scored, and why it was accepted or refused.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd

from mlops_car_price import dataset, registry, replay, tracking
from mlops_car_price.config import Config
from mlops_car_price.config import load as load_config
from mlops_car_price.drift import detector
from mlops_car_price.drift.detector import DriftReport
from mlops_car_price.training import promote, train
from mlops_car_price.training.promote import PromotionDecision


@dataclass(frozen=True)
class RetrainOutcome:
    """What one pass of the loop did, including the case where it correctly did nothing."""

    weeks: tuple[int, ...]
    drift_reports: tuple[DriftReport, ...]
    retrained: bool
    reason: str
    challenger_version: str | None = None
    decision: PromotionDecision | None = None

    @property
    def drifted_weeks(self) -> tuple[str, ...]:
        return tuple(report.label for report in self.drift_reports if report.drifted)

    def summary(self) -> str:
        if not self.retrained:
            return f"no retraining - {self.reason}"
        if self.decision is None:
            return f"challenger v{self.challenger_version} trained, not judged"
        return f"challenger v{self.challenger_version}: {self.decision.summary()}"


def extended_training_frame(config: Config, weeks: tuple[int, ...], scenario: str) -> pd.DataFrame:
    """The original training data plus the weeks that have since arrived.

    The snapshots carry their prices, so they are training data as soon as they land. In a
    system where labels are late this is the step that waits — which is worth saying out
    loud, because a replay makes it look free.
    """
    frames = [dataset.load_split("train_initial", config)]
    frames.extend(replay.load_snapshot(config, week, scenario) for week in weeks)
    return pd.concat(frames, ignore_index=True)


def run_cycle(
    config: Config,
    weeks: tuple[int, ...],
    scenario: str = "stable",
    force: bool = False,
    dry_run: bool = False,
    model_name: str | None = None,
) -> RetrainOutcome:
    """Judge the weeks, retrain if they warrant it, and let the gate rule on the result."""
    reports = tuple(
        detector.analyse_snapshot(config, week, scenario) for week in weeks
    )
    drifted = [report for report in reports if report.drifted]

    if not drifted and not force:
        return RetrainOutcome(
            weeks=weeks,
            drift_reports=reports,
            retrained=False,
            reason=f"no drift across {len(weeks)} week(s) - the champion still fits the traffic",
        )

    if dry_run:
        return RetrainOutcome(
            weeks=weeks,
            drift_reports=reports,
            retrained=False,
            reason=f"drift in {len(drifted)} week(s), but this is a dry run",
        )

    extended = extended_training_frame(config, weeks, scenario)
    result = train.train_run(
        config,
        model_name=model_name,
        run_name=f"challenger-weeks-{'-'.join(str(week) for week in weeks)}",
        register=True,
        training_frame=extended,
    )
    decision = promote.promote(config, result.registered_version)

    return RetrainOutcome(
        weeks=weeks,
        drift_reports=reports,
        retrained=True,
        reason="forced" if not drifted else f"drift in {len(drifted)} week(s)",
        challenger_version=result.registered_version,
        decision=decision,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weeks", type=int, nargs="+", required=True)
    parser.add_argument("--scenario", default="stable", choices=sorted(replay.SCENARIOS))
    parser.add_argument("--model", default=None, help="model to train (default: from config)")
    parser.add_argument(
        "--force", action="store_true", help="retrain even when no week looks drifted"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report the drift verdict and stop"
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args(argv)

    config = load_config(args.config) if args.config else load_config()
    tracking.configure(config)

    outcome = run_cycle(
        config,
        tuple(args.weeks),
        args.scenario,
        force=args.force,
        dry_run=args.dry_run,
        model_name=args.model,
    )

    for report in outcome.drift_reports:
        verdict = "DRIFT" if report.drifted else "clean"
        print(f"[retrain] {report.label}: {verdict}")
        for reason in report.reasons:
            print(f"[retrain]   {reason}")

    if outcome.decision is not None:
        champion = registry.alias_version(config, registry.CHAMPION)
        print(f"[retrain] challenger v{outcome.challenger_version}: "
              f"MAE {outcome.decision.candidate.mae:.1f} PLN")
        for reason in outcome.decision.reasons:
            print(f"[retrain]   {reason}")
        print(f"[retrain] champion is now v{champion.version if champion else 'none'}")

    print(f"[retrain] {outcome.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
