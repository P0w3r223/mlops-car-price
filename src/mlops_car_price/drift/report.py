"""Render a drift verdict as markdown.

Markdown rather than a dashboard: the report is a CI artifact and a file in a pull request,
where a diff between two weeks is readable. Rendering is kept out of `detector` so the
decision logic can be evaluated ten thousand times (session 4) without formatting anything.
"""

from __future__ import annotations

from mlops_car_price.drift.detector import DriftReport
from mlops_car_price.drift.metrics import FeatureDrift


def _format_p_value(value: float | None) -> str:
    if value is None:
        return "-"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _feature_row(drift: FeatureDrift, flagged: bool) -> str:
    return (
        f"| {'**' if flagged else ''}{drift.feature}{'**' if flagged else ''} "
        f"| {drift.kind} | {drift.psi:.3f} | {drift.normalised_shift:.3f} "
        f"| {_format_p_value(drift.p_value)} | {drift.missing_rate_current:.1%} "
        f"| {'yes' if flagged else 'no'} |"
    )


def render(report: DriftReport) -> str:
    """A self-contained markdown section for one snapshot."""
    verdict = "DRIFT" if report.drifted else ("NOT JUDGED" if not report.judged else "no drift")
    lines = [
        f"# Drift report - {report.label}",
        "",
        f"**Verdict: {verdict}** on {report.n_current:,} rows.",
        "",
    ]

    if report.reasons:
        lines.append("## Why")
        lines.append("")
        lines.extend(f"- {reason}" for reason in report.reasons)
        lines.append("")

    if report.judged:
        lines += [
            "## Features",
            "",
            "| Feature | Kind | PSI | Normalised shift | p-value | Missing | Flagged |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        lines.extend(
            _feature_row(drift, drift.feature in report.flagged_features)
            for drift in report.features
        )
        lines.append("")
        lines.append(
            "_The p-value column is a diagnostic, never a gate: on samples this size a KS "
            "test rejects shifts far too small to act on._"
        )
        lines.append("")

    if report.prediction is not None:
        lines += [
            "## Predictions",
            "",
            f"- PSI {report.prediction.psi:.3f}, normalised shift "
            f"{report.prediction.normalised_shift:.3f}",
        ]
        if report.current_mae is not None:
            increase = report.mae_increase_pct
            change = "-" if increase is None else f"{increase:+.1f}% vs the frozen holdout"
            lines.append(f"- MAE on this week: {report.current_mae:,.0f} PLN ({change})")
        lines.append("")

    return "\n".join(lines)
