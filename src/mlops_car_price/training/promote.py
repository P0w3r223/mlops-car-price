"""Champion/challenger: decide whether a candidate version may take over serving.

The decision is deliberately boring and explicit. Every refusal names its reason, and the
whole decision is written back to MLflow, so the registry answers "why is *this* model
serving?" months later without anyone remembering.

Four checks run today:

1. the holdout is large enough for a comparison to mean anything,
2. the candidate artifact **reproduces the MAE its own run recorded** — if loading the
   stored model gives a different number, the artifact or the data is wrong and nothing
   downstream should be trusted,
3. candidate and champion were scored on the **same dataset version**,
4. the improvement clears the configured margin.

One check is deliberately missing: whether the improvement is larger than noise. A margin
in złoty is not a significance test, and 23 571 paired rows deserve one. That arrives in
session 5 as a paired bootstrap over the per-row errors, which is why scoring already
returns them.

    python -m mlops_car_price.training.promote --version 2
    python -m mlops_car_price.training.promote --run-id <id> --register --dry-run
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import mlflow
import numpy as np
import pandas as pd
from car_price_ml import features as a3_features
from car_price_ml import model as a3_model
from mlflow.entities.model_registry import ModelVersion
from numpy.typing import NDArray

from mlops_car_price import dataset, registry, tracking
from mlops_car_price.config import Config
from mlops_car_price.config import load as load_config


@dataclass(frozen=True)
class Scored:
    """A registry version re-scored on the frozen holdout, with its per-row errors."""

    version: str
    model_name: str
    mae: float
    logged_mae: float | None
    dataset_hash: str | None
    artifact_mb: float | None
    absolute_errors: NDArray[np.float64] = field(repr=False)


@dataclass(frozen=True)
class PromotionDecision:
    """The verdict and the reasons behind it — both are recorded, not just the verdict."""

    promoted: bool
    reasons: tuple[str, ...]
    candidate: Scored
    champion: Scored | None
    n_holdout: int

    @property
    def delta_mae(self) -> float | None:
        """Champion MAE minus candidate MAE: positive means the candidate is better."""
        if self.champion is None:
            return None
        return self.champion.mae - self.candidate.mae

    def summary(self) -> str:
        verdict = "PROMOTED" if self.promoted else "REJECTED"
        delta = "n/a" if self.delta_mae is None else f"{self.delta_mae:+.1f} PLN"
        return f"{verdict} - candidate v{self.candidate.version}, delta {delta}"


def _holdout(config: Config) -> tuple[pd.DataFrame, pd.Series]:
    return a3_features.prepare(dataset.load_split("holdout_eval", config))


def score_version(
    config: Config, version: ModelVersion, x: pd.DataFrame, y: pd.Series
) -> Scored:
    """Load a registered version and score it on the holdout rows it is judged on.

    The metrics are recomputed rather than read from the run: a stored artifact that no
    longer reproduces its own numbers is exactly the failure this step exists to catch.
    """
    fitted = registry.load_version(config, version.version)
    predictions = fitted.predict(x)
    metrics = a3_model.evaluate(y, predictions)
    run = registry.version_run(config, version)

    return Scored(
        version=version.version,
        model_name=run.data.params.get("model", "unknown"),
        mae=metrics["mae"],
        logged_mae=run.data.metrics.get("mae"),
        dataset_hash=run.data.tags.get("dataset_hash"),
        artifact_mb=run.data.metrics.get("model_size_mb"),
        absolute_errors=np.abs(np.asarray(y, dtype=float) - np.asarray(predictions, dtype=float)),
    )


def decide(
    config: Config, candidate: Scored, champion: Scored | None, n_holdout: int
) -> PromotionDecision:
    """Apply the promotion rules. Pure — no registry writes, so it is trivially testable."""
    reasons: list[str] = []

    if n_holdout < config.promotion.min_holdout_rows:
        reasons.append(
            f"holdout has {n_holdout} rows, fewer than the required "
            f"{config.promotion.min_holdout_rows}"
        )

    tolerance = config.promotion.mae_reproduction_tolerance_pln
    if candidate.logged_mae is not None and abs(candidate.mae - candidate.logged_mae) > tolerance:
        reasons.append(
            f"candidate artifact scores {candidate.mae:.1f} PLN but its run logged "
            f"{candidate.logged_mae:.1f} PLN - the artifact does not reproduce its own metrics"
        )

    budget = config.promotion.max_artifact_mb
    if candidate.artifact_mb is not None and candidate.artifact_mb > budget:
        # Accuracy is not the only requirement a served model has to meet. A model the
        # retraining loop cannot carry is refused however well it scores (ADR 0003).
        reasons.append(
            f"artifact is {candidate.artifact_mb:,.1f} MB, over the {budget:,.1f} MB "
            f"deployment budget"
        )

    if reasons:
        return PromotionDecision(False, tuple(reasons), candidate, champion, n_holdout)

    if champion is None:
        return PromotionDecision(
            True,
            ("no champion registered yet - the first valid candidate takes the alias",),
            candidate,
            champion,
            n_holdout,
        )

    if candidate.dataset_hash != champion.dataset_hash:
        reasons.append(
            f"candidate was trained against dataset {str(candidate.dataset_hash)[:12]} and the "
            f"champion against {str(champion.dataset_hash)[:12]} - rebuild before comparing"
        )
        return PromotionDecision(False, tuple(reasons), candidate, champion, n_holdout)

    delta = champion.mae - candidate.mae
    margin = config.promotion.min_mae_improvement_pln
    if delta < margin:
        reasons.append(
            f"MAE improves by {delta:+.1f} PLN, short of the required {margin:.1f} PLN"
        )
        return PromotionDecision(False, tuple(reasons), candidate, champion, n_holdout)

    return PromotionDecision(
        True,
        (f"MAE improves by {delta:.1f} PLN, clearing the {margin:.1f} PLN margin",),
        candidate,
        champion,
        n_holdout,
    )


def evaluate_candidate(config: Config, version: str | int) -> PromotionDecision:
    """Score a candidate and the current champion on the frozen holdout, then decide."""
    tracking.configure(config)
    x, y = _holdout(config)

    candidate_version = registry.get_version(config, version)
    candidate = score_version(config, candidate_version, x, y)

    champion_version = registry.alias_version(config, registry.CHAMPION)
    champion = None
    if champion_version is not None and champion_version.version != candidate_version.version:
        champion = score_version(config, champion_version, x, y)

    return decide(config, candidate, champion, n_holdout=len(x))


def apply_decision(config: Config, decision: PromotionDecision) -> None:
    """Move the aliases and record the decision on the version and in a run.

    A rejected candidate keeps the ``challenger`` alias: a refusal is a state worth being
    able to inspect, not something to erase.
    """
    alias = registry.CHAMPION if decision.promoted else registry.CHALLENGER
    registry.set_alias(config, alias, decision.candidate.version)

    mlflow_client = tracking.client(config)
    name = config.mlflow.registered_model
    mlflow_client.set_model_version_tag(
        name,
        decision.candidate.version,
        "promotion",
        "accepted" if decision.promoted else "rejected",
    )
    mlflow_client.set_model_version_tag(
        name, decision.candidate.version, "promotion_reason", " | ".join(decision.reasons)
    )

    tracking.ensure_experiment(config)
    with mlflow.start_run(run_name=f"promotion-v{decision.candidate.version}"):
        mlflow.log_params(
            {
                "candidate_version": decision.candidate.version,
                "champion_version": decision.champion.version if decision.champion else "none",
                "decision": "promoted" if decision.promoted else "rejected",
            }
        )
        mlflow.log_metrics(
            {
                "candidate_mae": decision.candidate.mae,
                "n_holdout": decision.n_holdout,
                **({"champion_mae": decision.champion.mae} if decision.champion else {}),
                **({"delta_mae": decision.delta_mae} if decision.delta_mae is not None else {}),
            }
        )
        mlflow.set_tag("promotion_reason", " | ".join(decision.reasons))


def promote(config: Config, version: str | int, dry_run: bool = False) -> PromotionDecision:
    """Evaluate a candidate and, unless this is a dry run, act on the verdict."""
    decision = evaluate_candidate(config, version)
    if not dry_run:
        apply_decision(config, decision)
    return decision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--version", help="registry version to judge")
    source.add_argument("--run-id", help="run to register first, then judge")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument(
        "--dry-run", action="store_true", help="decide and report, but move no aliases"
    )
    args = parser.parse_args(argv)

    config = load_config(args.config) if args.config else load_config()
    version = args.version or registry.register_run(config, args.run_id).version

    decision = promote(config, version, dry_run=args.dry_run)

    champion = decision.champion
    print(f"[promote] candidate v{decision.candidate.version} "
          f"({decision.candidate.model_name}): MAE {decision.candidate.mae:.1f} PLN")
    if champion is not None:
        print(f"[promote] champion  v{champion.version} "
              f"({champion.model_name}): MAE {champion.mae:.1f} PLN")
    print(f"[promote] holdout: {decision.n_holdout:,} rows")
    for reason in decision.reasons:
        print(f"[promote] reason: {reason}")
    print(f"[promote] {decision.summary()}{' (dry run)' if args.dry_run else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
