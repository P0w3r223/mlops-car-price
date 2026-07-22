"""The simulated stream: reproducible weeks, and scenarios that break what they claim to."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlops_car_price import dataset, replay
from mlops_car_price.config import Config

FEATURE_COLUMNS = ("mark", "model", "fuel", "province", "year", "mileage", "vol_engine")
SNAPSHOT_ROWS = 300


@pytest.fixture
def built(project: Config) -> Config:
    dataset.build(project)
    return project


def test_a_snapshot_has_the_requested_size(built: Config):
    snapshot = replay.make_snapshot(built, week=1, n_rows=SNAPSHOT_ROWS)

    assert len(snapshot) == SNAPSHOT_ROWS


def test_a_week_is_reproducible_and_weeks_differ(built: Config):
    first = replay.make_snapshot(built, week=1, n_rows=SNAPSHOT_ROWS)
    again = replay.make_snapshot(built, week=1, n_rows=SNAPSHOT_ROWS)
    later = replay.make_snapshot(built, week=2, n_rows=SNAPSHOT_ROWS)

    pd.testing.assert_frame_equal(first, again)
    assert not first.equals(later)


def test_price_shock_moves_the_target_and_nothing_else(built: Config):
    """The defining property: this scenario is invisible to any feature-only monitor."""
    stable = replay.make_snapshot(built, week=3, scenario="stable", n_rows=SNAPSHOT_ROWS)
    shocked = replay.make_snapshot(
        built, week=3, scenario="price_shock", magnitude=0.3, n_rows=SNAPSHOT_ROWS
    )

    pd.testing.assert_frame_equal(stable[list(FEATURE_COLUMNS)], shocked[list(FEATURE_COLUMNS)])
    assert shocked["price"].mean() == pytest.approx(stable["price"].mean() * 1.3, rel=1e-9)


def test_fuel_mix_shift_raises_the_electrified_share(built: Config):
    stable = replay.make_snapshot(built, week=4, scenario="stable", n_rows=SNAPSHOT_ROWS)
    shifted = replay.make_snapshot(
        built, week=4, scenario="fuel_mix_shift", magnitude=0.5, n_rows=SNAPSHOT_ROWS
    )

    share = lambda frame: frame["fuel"].isin(replay.ELECTRIFIED_FUELS).mean()  # noqa: E731
    assert share(shifted) > share(stable)


def test_mileage_shift_moves_the_column_up(built: Config):
    stable = replay.make_snapshot(built, week=5, scenario="stable", n_rows=SNAPSHOT_ROWS)
    shifted = replay.make_snapshot(
        built, week=5, scenario="mileage_shift", magnitude=1.0, n_rows=SNAPSHOT_ROWS
    )

    assert shifted["mileage"].mean() > stable["mileage"].mean()
    assert (shifted["mileage"] >= 0).all()


def test_unseen_makes_introduces_labels_absent_from_the_pool(built: Config):
    pool = dataset.load_split("stream_pool", built)

    snapshot = replay.make_snapshot(
        built, week=6, scenario="unseen_makes", magnitude=0.5, n_rows=SNAPSHOT_ROWS
    )

    new_labels = set(snapshot["mark"]) - set(pool["mark"])
    assert new_labels
    assert all(label.startswith(replay.UNSEEN_MAKE_PREFIX) for label in new_labels)


def test_missing_engine_volume_blanks_the_field(built: Config):
    snapshot = replay.make_snapshot(
        built, week=7, scenario="missing_engine_volume", magnitude=0.4, n_rows=SNAPSHOT_ROWS
    )

    assert snapshot["vol_engine"].isna().mean() == pytest.approx(0.4, abs=0.1)


def test_stable_really_changes_nothing(built: Config):
    pool = dataset.load_split("stream_pool", built)
    snapshot = replay.make_snapshot(built, week=8, scenario="stable", n_rows=SNAPSHOT_ROWS)

    # Every drawn row exists in the pool untouched.
    merged = snapshot.merge(pool, how="left", indicator=True)
    assert (merged["_merge"] == "both").all()


def test_an_unknown_scenario_is_rejected(built: Config):
    with pytest.raises(KeyError, match="unknown scenario"):
        replay.make_snapshot(built, week=1, scenario="apocalypse")


def test_snapshots_round_trip_through_disk(built: Config):
    path = replay.write_snapshot(built, week=9, scenario="stable", n_rows=SNAPSHOT_ROWS)

    assert path.exists()
    loaded = replay.load_snapshot(built, week=9, scenario="stable")
    pd.testing.assert_frame_equal(
        loaded, replay.make_snapshot(built, week=9, scenario="stable", n_rows=SNAPSHOT_ROWS)
    )


def test_loading_a_snapshot_that_was_never_written_says_what_to_run(built: Config):
    with pytest.raises(FileNotFoundError, match=r"mlops_car_price\.replay"):
        replay.load_snapshot(built, week=99, scenario="stable")


def test_every_scenario_produces_a_usable_frame(built: Config):
    """Whatever a scenario breaks, it must still return the columns the pipeline expects."""
    for name in replay.SCENARIOS:
        snapshot = replay.make_snapshot(built, week=10, scenario=name, n_rows=SNAPSHOT_ROWS)

        assert len(snapshot) == SNAPSHOT_ROWS, name
        assert set(FEATURE_COLUMNS) <= set(snapshot.columns), name
        assert np.isfinite(snapshot["price"]).all(), name
