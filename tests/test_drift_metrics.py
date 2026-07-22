"""Drift metrics, checked against hand arithmetic and against known distributions.

PSI is the one metric implemented here rather than taken from SciPy, so it is pinned to a
worked example: two bins, shares 0.5/0.5 against 0.25/0.75, gives
``(0.25-0.5)*ln(0.25/0.5) + (0.75-0.5)*ln(0.75/0.5) ≈ 0.2747``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlops_car_price.drift import metrics

HAND_COMPUTED_PSI = 0.25 * np.log(0.5) * -1 + 0.25 * np.log(1.5)


def test_psi_matches_hand_arithmetic():
    reference = np.array([0.5, 0.5])
    current = np.array([0.25, 0.75])

    assert metrics.population_stability_index(reference, current) == pytest.approx(
        HAND_COMPUTED_PSI
    )


def test_psi_is_zero_for_identical_distributions():
    shares = np.array([0.2, 0.3, 0.5])

    assert metrics.population_stability_index(shares, shares) == pytest.approx(0.0)


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(11)
    reference = pd.Series(rng.normal(0, 1, 5_000))

    psis = [
        metrics.numeric_drift("x", reference, pd.Series(rng.normal(shift, 1, 5_000))).psi
        for shift in (0.0, 0.25, 0.5, 1.0)
    ]

    assert psis == sorted(psis)
    assert psis[0] < 0.05 < psis[-1]


def test_normalised_shift_reads_in_standard_deviations():
    """A one-sigma mean shift should register as roughly one unit of normalised distance."""
    rng = np.random.default_rng(3)
    reference = pd.Series(rng.normal(0, 2.0, 20_000))
    current = pd.Series(rng.normal(2.0, 2.0, 20_000))

    drift = metrics.numeric_drift("x", reference, current)

    assert drift.normalised_shift == pytest.approx(1.0, abs=0.05)


def test_the_ks_p_value_collapses_with_sample_size_on_a_negligible_shift():
    """The reason gates use effect sizes: significance here is a statement about n."""
    rng = np.random.default_rng(5)
    tiny_shift = 0.05

    small = metrics.numeric_drift(
        "x", pd.Series(rng.normal(0, 1, 500)), pd.Series(rng.normal(tiny_shift, 1, 500))
    )
    large = metrics.numeric_drift(
        "x", pd.Series(rng.normal(0, 1, 100_000)), pd.Series(rng.normal(tiny_shift, 1, 100_000))
    )

    assert small.p_value > 0.05
    assert large.p_value < 0.001
    # The shift itself never changed - only the sample size did.
    assert large.normalised_shift == pytest.approx(small.normalised_shift, abs=0.1)


def test_a_brand_new_category_keeps_psi_finite():
    reference = pd.Series(["a"] * 100 + ["b"] * 100)
    current = pd.Series(["a"] * 100 + ["b"] * 50 + ["c"] * 50)

    drift = metrics.categorical_drift("mark", reference, current)

    assert np.isfinite(drift.psi)
    assert drift.psi > 0


def test_disjoint_categories_give_total_variation_of_one():
    reference = pd.Series(["a"] * 50)
    current = pd.Series(["z"] * 50)

    drift = metrics.categorical_drift("mark", reference, current)

    assert drift.normalised_shift == pytest.approx(1.0, abs=0.02)


def test_a_constant_column_does_not_divide_by_zero():
    constant = pd.Series([5.0] * 100)

    same = metrics.numeric_drift("x", constant, constant)
    moved = metrics.numeric_drift("x", constant, pd.Series([7.0] * 100))

    assert same.normalised_shift == 0.0
    assert np.isinf(moved.normalised_shift)


def test_missing_values_are_reported_and_excluded_from_the_distribution():
    reference = pd.Series([1.0, 2.0, 3.0, 4.0])
    current = pd.Series([1.0, 2.0, np.nan, np.nan])

    drift = metrics.numeric_drift("x", reference, current)

    assert drift.missing_rate_reference == 0.0
    assert drift.missing_rate_current == 0.5
    assert drift.missing_rate_increase == 0.5
    assert np.isfinite(drift.psi)


def test_an_all_missing_column_reports_rather_than_crashes():
    drift = metrics.numeric_drift("x", pd.Series([1.0, 2.0]), pd.Series([np.nan, np.nan]))

    assert drift.missing_rate_current == 1.0
    assert np.isnan(drift.psi)
    assert drift.p_value is None


def test_compare_names_the_columns_it_cannot_find():
    frame = pd.DataFrame({"age": [1, 2, 3]})

    with pytest.raises(KeyError, match="mileage"):
        metrics.compare(frame, frame, ("age", "mileage"), ())
