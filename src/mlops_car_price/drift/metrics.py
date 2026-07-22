"""Distribution-comparison metrics, one column at a time.

Four numbers are computed per feature because they fail differently:

**PSI** (population stability index) compares binned shares and is the one the thresholds
gate on. It is an *effect size*: it does not care how many rows produced it.

**Wasserstein distance**, normalised by the reference standard deviation, answers "how far
did the distribution move, in units of its own spread?" — the same question PSI answers,
from a direction that does not depend on binning.

**The KS statistic and its p-value** are reported and deliberately *not* gated on. At
100 000 rows a KS test rejects for shifts far too small to matter; the p-value there measures
sample size more than drift. Session 4 puts a number on that claim rather than asserting it.

**The missing rate** catches the failure the other three cannot see at all: a field that
arrives empty. Dropping nulls and comparing what is left can leave every distribution metric
perfectly calm while half the column is gone.

PSI's formula divides by the reference share, so an empty bin would send it to infinity. The
standard fix is used: a zero share becomes half an observation, which keeps a genuinely new
category finite but large.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_BINS = 10
# A bin nobody landed in is treated as "half an observation", the usual correction.
_ZERO_SHARE_FLOOR = 0.5


@dataclass(frozen=True)
class FeatureDrift:
    """How far one column moved between the reference and the current sample."""

    feature: str
    kind: str  # "numeric" or "categorical"
    psi: float
    normalised_shift: float  # Wasserstein / reference sigma, or total variation distance
    ks_statistic: float | None
    p_value: float | None
    missing_rate_reference: float
    missing_rate_current: float
    n_reference: int
    n_current: int
    # What this column scores against itself when nothing has drifted, at this sample size.
    # None when calibration is switched off.
    psi_noise_floor: float | None = None
    shift_noise_floor: float | None = None

    @property
    def missing_rate_increase(self) -> float:
        return self.missing_rate_current - self.missing_rate_reference


def _shares(counts: np.ndarray, total: int) -> np.ndarray:
    """Convert counts to shares, floored so an empty bin cannot make PSI infinite."""
    floored = np.where(counts == 0, _ZERO_SHARE_FLOOR, counts.astype(float))
    return floored / max(total, 1)


def population_stability_index(
    reference_shares: np.ndarray, current_shares: np.ndarray
) -> float:
    """PSI between two share vectors: sum (current - reference) * ln(current / reference)."""
    ratio = np.log(current_shares / reference_shares)
    return float(np.sum((current_shares - reference_shares) * ratio))


def numeric_bins(reference: np.ndarray, bins: int = DEFAULT_BINS) -> np.ndarray:
    """Quantile bin edges taken from the reference, open at both ends.

    Quantiles rather than equal width: the price and mileage distributions are skewed, and
    equal-width bins would put almost every row in one of them.
    """
    quantiles = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges = np.unique(quantiles)
    if len(edges) < 2:  # a constant column has no interior structure to bin
        edges = np.array([edges[0], edges[0] + 1.0])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def numeric_drift(
    feature: str, reference: pd.Series, current: pd.Series, bins: int = DEFAULT_BINS
) -> FeatureDrift:
    """Compare a numeric column: PSI, normalised Wasserstein, KS, and the missing rate."""
    reference_values = reference.dropna().to_numpy(dtype=float)
    current_values = current.dropna().to_numpy(dtype=float)

    if len(reference_values) == 0 or len(current_values) == 0:
        return FeatureDrift(
            feature=feature,
            kind="numeric",
            psi=float("nan"),
            normalised_shift=float("nan"),
            ks_statistic=None,
            p_value=None,
            missing_rate_reference=float(reference.isna().mean()),
            missing_rate_current=float(current.isna().mean()),
            n_reference=len(reference),
            n_current=len(current),
        )

    edges = numeric_bins(reference_values, bins)
    reference_counts, _ = np.histogram(reference_values, bins=edges)
    current_counts, _ = np.histogram(current_values, bins=edges)
    psi = population_stability_index(
        _shares(reference_counts, len(reference_values)),
        _shares(current_counts, len(current_values)),
    )

    spread = float(np.std(reference_values))
    distance = float(stats.wasserstein_distance(reference_values, current_values))
    ks = stats.ks_2samp(reference_values, current_values)

    return FeatureDrift(
        feature=feature,
        kind="numeric",
        psi=psi,
        # A constant reference column has no spread to normalise by; any movement at all is
        # then total, so report it as such rather than dividing by zero.
        normalised_shift=distance / spread if spread > 0 else (0.0 if distance == 0 else np.inf),
        ks_statistic=float(ks.statistic),
        p_value=float(ks.pvalue),
        missing_rate_reference=float(reference.isna().mean()),
        missing_rate_current=float(current.isna().mean()),
        n_reference=len(reference),
        n_current=len(current),
    )


def categorical_drift(feature: str, reference: pd.Series, current: pd.Series) -> FeatureDrift:
    """Compare a categorical column: PSI, total variation distance, chi-square p-value."""
    reference_values = reference.dropna().astype(str)
    current_values = current.dropna().astype(str)
    categories = sorted(set(reference_values.unique()) | set(current_values.unique()))

    reference_counts = reference_values.value_counts().reindex(categories, fill_value=0).to_numpy()
    current_counts = current_values.value_counts().reindex(categories, fill_value=0).to_numpy()

    reference_shares = _shares(reference_counts, len(reference_values))
    current_shares = _shares(current_counts, len(current_values))
    psi = population_stability_index(reference_shares, current_shares)
    total_variation = float(np.abs(current_shares - reference_shares).sum() / 2)

    p_value: float | None = None
    table = np.vstack([reference_counts, current_counts])
    keep = table.sum(axis=0) > 0
    if keep.sum() >= 2 and table[:, keep].sum() > 0:
        p_value = float(stats.chi2_contingency(table[:, keep]).pvalue)

    return FeatureDrift(
        feature=feature,
        kind="categorical",
        psi=psi,
        normalised_shift=total_variation,
        ks_statistic=None,
        p_value=p_value,
        missing_rate_reference=float(reference.isna().mean()),
        missing_rate_current=float(current.isna().mean()),
        n_reference=len(reference),
        n_current=len(current),
    )


def noise_floor(
    reference: pd.Series,
    n_current: int,
    kind: str,
    resamples: int,
    quantile: float,
    seed: int,
    bins: int = DEFAULT_BINS,
) -> tuple[float, float]:
    """What this column scores against itself when nothing has drifted.

    Both PSI and the distances are biased upward by small samples and by many categories:
    a 200-category column compared on 2 000 rows scores PSI ≈ 0.36 against its own source,
    which is well past the textbook "0.2 means drift". Comparing against a fixed number
    would therefore alarm on the control case — the failure mode that trains people to
    ignore the monitor.

    So the reference is resampled at the size of the actual snapshot, scored against
    itself, and a high quantile of that null distribution becomes the floor a real signal
    has to clear. Returns ``(psi_floor, shift_floor)``.
    """
    values = reference.dropna()
    if len(values) == 0 or resamples <= 0 or n_current <= 0:
        return 0.0, 0.0

    rng = np.random.default_rng(seed)
    psis, shifts = np.empty(resamples), np.empty(resamples)

    # Only PSI and the distance are needed here, and everything that does not depend on the
    # draw (bin edges, reference shares, spread, category order) is computed once. The KS
    # and chi-square tests are skipped entirely: nothing gates on them, and they dominate
    # the cost of a null loop run hundreds of times per column.
    if kind == "numeric":
        reference_values = values.to_numpy(dtype=float)
        edges = numeric_bins(reference_values, bins)
        reference_shares = _shares(
            np.histogram(reference_values, bins=edges)[0], len(reference_values)
        )
        spread = float(np.std(reference_values))
        for index in range(resamples):
            drawn = reference_values[rng.integers(0, len(reference_values), n_current)]
            current_shares = _shares(np.histogram(drawn, bins=edges)[0], n_current)
            psis[index] = population_stability_index(reference_shares, current_shares)
            distance = float(stats.wasserstein_distance(reference_values, drawn))
            shifts[index] = distance / spread if spread > 0 else 0.0
    else:
        codes, categories = pd.factorize(values.astype(str))
        reference_counts = np.bincount(codes, minlength=len(categories))
        reference_shares = _shares(reference_counts, len(codes))
        for index in range(resamples):
            drawn = codes[rng.integers(0, len(codes), n_current)]
            current_shares = _shares(
                np.bincount(drawn, minlength=len(categories)), n_current
            )
            psis[index] = population_stability_index(reference_shares, current_shares)
            shifts[index] = float(np.abs(current_shares - reference_shares).sum() / 2)

    return float(np.quantile(psis, quantile)), float(np.quantile(shifts, quantile))


def per_column_quantile(report_quantile: float, n_columns: int) -> float:
    """Split a report-level confidence across the columns that could each raise the alarm.

    A report fires if *any* column fires, so calibrating every column at the 99th percentile
    of its own null gives roughly seven chances to be wrong, not one. The tail is therefore
    divided across the columns (a Bonferroni split): conservative, because the columns are
    correlated, and simple enough to state out loud.
    """
    if n_columns <= 0:
        return report_quantile
    return 1.0 - (1.0 - report_quantile) / n_columns


def noise_floors(
    reference: pd.DataFrame,
    n_current: int,
    numeric_features: tuple[str, ...],
    categorical_features: tuple[str, ...],
    resamples: int,
    quantile: float,
    seed: int,
    bins: int = DEFAULT_BINS,
) -> dict[str, tuple[float, float]]:
    """Noise floors for every column at one sample size.

    ``quantile`` is the confidence for the **report**; each column is calibrated at the
    tighter quantile that follows from it. Split out from ``compare`` because a floor
    depends only on the reference column and the snapshot size — never on the snapshot
    itself — so the evaluation harness computes them once and reuses them across thousands
    of trials.
    """
    columns = (
        *((column, "numeric") for column in numeric_features),
        *((column, "categorical") for column in categorical_features),
    )
    column_quantile = per_column_quantile(quantile, len(columns))
    return {
        column: noise_floor(
            reference[column], n_current, kind, resamples, column_quantile, seed, bins
        )
        for column, kind in columns
    }


def compare(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_features: tuple[str, ...],
    categorical_features: tuple[str, ...],
    bins: int = DEFAULT_BINS,
    calibration_resamples: int = 0,
    calibration_quantile: float = 0.99,
    seed: int = 0,
    floors: dict[str, tuple[float, float]] | None = None,
) -> tuple[FeatureDrift, ...]:
    """Run the per-column comparison over every feature the model actually consumes.

    Args:
        floors: Precomputed noise floors keyed by column. When given, they are used as they
            are and ``calibration_resamples`` is ignored — the caller has already paid for
            the measurement.
    """
    missing = [
        column
        for column in (*numeric_features, *categorical_features)
        if column not in reference.columns or column not in current.columns
    ]
    if missing:
        raise KeyError(f"columns missing from reference or current frame: {missing}")

    results = []
    for column, kind in (
        *((column, "numeric") for column in numeric_features),
        *((column, "categorical") for column in categorical_features),
    ):
        drift = (
            numeric_drift(column, reference[column], current[column], bins)
            if kind == "numeric"
            else categorical_drift(column, reference[column], current[column])
        )
        if floors is not None:
            if column in floors:
                psi_floor, shift_floor = floors[column]
                drift = replace(drift, psi_noise_floor=psi_floor, shift_noise_floor=shift_floor)
        elif calibration_resamples > 0:
            psi_floor, shift_floor = noise_floor(
                reference[column],
                len(current),
                kind,
                calibration_resamples,
                per_column_quantile(
                    calibration_quantile, len(numeric_features) + len(categorical_features)
                ),
                seed,
                bins,
            )
            drift = replace(drift, psi_noise_floor=psi_floor, shift_noise_floor=shift_floor)
        results.append(drift)
    return tuple(results)
