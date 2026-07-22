"""Which signal catches which failure — one row per drift scenario.

Three monitors watch the same weekly snapshot and disagree by design. Feature drift is
cheap and immediate; prediction drift needs the model but no labels; realised error needs
labels and therefore, in a real system, patience. The table this script produces shows that
no one of them subsumes the others, which is the argument for paying for all three.

    python examples/drift_scenarios.py [--rows 2000] [--output reports/drift_scenarios.md]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from mlops_car_price import dataset, registry, replay
from mlops_car_price.config import Config
from mlops_car_price.config import load as load_config
from mlops_car_price.drift import detector

# One week per scenario, so each snapshot is drawn with its own seed.
WEEKS = {name: index + 1 for index, name in enumerate(replay.SCENARIOS)}


@dataclass(frozen=True)
class ScenarioOutcome:
    scenario: str
    description: str
    feature_drift: tuple[str, ...]
    prediction_flagged: bool
    mae_increase_pct: float | None
    drifted: bool


def run(config: Config, rows: int) -> list[ScenarioOutcome]:
    reference = dataset.load_split("train_initial", config)

    champion = registry.alias_version(config, registry.CHAMPION)
    if champion is None:
        raise RuntimeError("no champion registered - train and promote a model first")
    model = registry.load_version(config, champion.version)
    reference_mae = registry.version_run(config, champion).data.metrics.get("mae")

    outcomes = []
    for name, scenario in replay.SCENARIOS.items():
        snapshot = replay.make_snapshot(config, WEEKS[name], name, n_rows=rows)
        report = detector.analyse(
            config, reference, snapshot, f"scenario {name}", model, reference_mae
        )
        outcomes.append(
            ScenarioOutcome(
                scenario=name,
                description=scenario.description,
                feature_drift=report.flagged_features,
                prediction_flagged=any(r.startswith("predictions:") for r in report.reasons),
                mae_increase_pct=report.mae_increase_pct,
                drifted=report.drifted,
            )
        )
    return outcomes


def render(outcomes: list[ScenarioOutcome], rows: int) -> str:
    header = (
        "| Scenario | What breaks | Feature drift | Prediction drift | MAE change | Alert |\n"
        "|---|---|---|---|---:|---|\n"
    )
    body = ""
    for outcome in outcomes:
        features = ", ".join(outcome.feature_drift) if outcome.feature_drift else "-"
        change = "-" if outcome.mae_increase_pct is None else f"{outcome.mae_increase_pct:+.1f}%"
        body += (
            f"| `{outcome.scenario}` | {outcome.description} | {features} "
            f"| {'yes' if outcome.prediction_flagged else 'no'} | {change} "
            f"| {'**yes**' if outcome.drifted else 'no'} |\n"
        )
    footer = (
        f"\nSnapshots of {rows:,} rows drawn from `stream_pool`, compared against "
        f"`train_initial`. Scenario magnitudes are the defaults in `replay.SCENARIOS`.\n"
    )
    return header + body + footer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--output", default="reports/drift_scenarios.md")
    args = parser.parse_args(argv)

    config = load_config(args.config) if args.config else load_config()
    rows = args.rows or config.replay.snapshot_rows

    table = render(run(config, rows), rows)
    output = config.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table, encoding="utf-8")
    print(table)
    print(f"[scenarios] written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
