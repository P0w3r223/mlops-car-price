"""Measure the drift detector: false alarms, power, and what sample size does to it.

Produces the three tables the README leans on. Runs a few thousand snapshot comparisons, so
it takes minutes rather than seconds.

    python examples/detector_evaluation.py [--trials 200] [--output reports/detector_evaluation.md]
"""

from __future__ import annotations

import argparse

from mlops_car_price import dataset
from mlops_car_price.config import Config
from mlops_car_price.config import load as load_config
from mlops_car_price.drift import evaluate
from mlops_car_price.drift.evaluate import DETECTORS, DetectionRates

# Scenarios whose magnitude means something on a continuous scale, so a power curve reads.
POWER_GRID = {
    "mileage_shift": (0.05, 0.1, 0.2, 0.35, 0.5),
    "unseen_makes": (0.01, 0.02, 0.05, 0.1, 0.2),
    "fuel_mix_shift": (0.02, 0.05, 0.1, 0.2, 0.4),
}
# A shift small enough that nobody would retrain for it - the point is what n does to it.
NEGLIGIBLE_SHIFT = 0.05
SAMPLE_SIZES = (250, 500, 1_000, 2_000, 5_000)

DETECTOR_LABELS = {
    "fixed": "fixed thresholds",
    "calibrated": "calibrated (this project)",
    "p_value": "KS p-value",
}


def _table(first_column: str, rows: list[tuple[str, DetectionRates]]) -> str:
    columns = " | ".join(DETECTOR_LABELS[name] for name in DETECTORS)
    dashes = "|".join("---:" for _ in DETECTORS)
    lines = [f"| {first_column} | {columns} |", f"|---|{dashes}|"]
    lines += [
        f"| {label} | " + " | ".join(f"{rates.rate(name):.1%}" for name in DETECTORS) + " |"
        for label, rates in rows
    ]
    return "\n".join(lines)


def render(
    false_alarms: list[DetectionRates],
    curves: dict[str, list[DetectionRates]],
    sweep: list[DetectionRates],
    trials: int,
    power_rows: int,
) -> str:
    lines = [
        "# Detector evaluation",
        "",
        f"Every figure is a detection rate over {trials} independent snapshots, judged by "
        "three detectors on identical data.",
        "",
        "## False alarms",
        "",
        "Unshifted weeks at growing sample sizes. Nothing changed, so every figure here is "
        "an error. This is where a fixed threshold fails: PSI's null distribution grows as "
        "the sample shrinks, and the high-cardinality `model` column crosses 0.2 on noise "
        "alone.",
        "",
        _table("Rows in snapshot", [(f"{point.n_rows:,}", point) for point in false_alarms]),
        "",
        "## Power",
        "",
        f"Detection rate against the size of the shift, on snapshots of {power_rows:,} rows.",
        "",
    ]

    for scenario, points in curves.items():
        lines += [
            f"### `{scenario}`",
            "",
            _table("Magnitude", [(f"{point.magnitude:g}", point) for point in points]),
            "",
        ]

    lines += [
        "## Sensitivity to sample size",
        "",
        f"The same negligible shift (`mileage_shift`, magnitude {NEGLIGIBLE_SHIFT:g} standard "
        "deviations) at growing sample sizes. The shift never changes - only `n` does, and "
        "only one detector notices.",
        "",
        _table("Rows in snapshot", [(f"{point.n_rows:,}", point) for point in sweep]),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--output", default="reports/detector_evaluation.md")
    args = parser.parse_args(argv)

    config: Config = load_config(args.config) if args.config else load_config()
    rows = args.rows or config.replay.snapshot_rows
    reference = dataset.load_split("train_initial", config)
    # One cache across every table: the floors depend on the sample size and nothing else.
    cache = evaluate.FloorCache(config, reference)

    false_alarms = []
    for size in SAMPLE_SIZES:
        print(f"[eval] false alarms at {size:,} rows ...", flush=True)
        false_alarms.append(evaluate.false_alarm_rates(config, reference, size, args.trials, cache))

    curves = {}
    for scenario, magnitudes in POWER_GRID.items():
        print(f"[eval] power curve for {scenario} ...", flush=True)
        curves[scenario] = evaluate.power_curve(
            config, reference, scenario, magnitudes, rows, args.trials, cache
        )

    print("[eval] sample-size sweep ...", flush=True)
    sweep = evaluate.sample_size_sweep(
        config, reference, "mileage_shift", NEGLIGIBLE_SHIFT, SAMPLE_SIZES, args.trials, cache
    )

    table = render(false_alarms, curves, sweep, args.trials, rows)
    output = config.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"[eval] written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
