"""The split has to be reproducible and disjoint — every later comparison rests on it."""

from __future__ import annotations

import numpy as np
import pytest

from mlops_car_price import dataset
from mlops_car_price.config import Config

PROPORTIONS = {"train_initial": 0.6, "holdout_eval": 0.2, "stream_pool": 0.2}


def test_split_indices_partition_every_row():
    indices = dataset.split_indices(1_000, PROPORTIONS, seed=42)

    combined = np.concatenate(list(indices.values()))
    assert sorted(combined.tolist()) == list(range(1_000))
    assert len(indices["train_initial"]) == 600
    assert len(indices["holdout_eval"]) == 200


def test_split_indices_are_deterministic_for_a_seed_and_move_with_it():
    first = dataset.split_indices(1_000, PROPORTIONS, seed=42)
    again = dataset.split_indices(1_000, PROPORTIONS, seed=42)
    other = dataset.split_indices(1_000, PROPORTIONS, seed=43)

    assert np.array_equal(first["holdout_eval"], again["holdout_eval"])
    assert not np.array_equal(first["holdout_eval"], other["holdout_eval"])


def test_split_rejects_a_frame_too_small_to_fill_three_parts():
    with pytest.raises(ValueError, match="too few"):
        dataset.split_indices(4, PROPORTIONS, seed=42)


def test_build_writes_all_splits_and_a_matching_manifest(project: Config):
    manifest = dataset.build(project)

    total = sum(split["rows"] for split in manifest["splits"].values())
    assert total == manifest["source"]["clean_rows"]
    for name in dataset.SPLIT_NAMES:
        path = dataset.split_path(name, project)
        assert path.exists()
        assert manifest["splits"][name]["sha256"] == dataset.sha256_file(path)
        assert len(dataset.load_split(name, project)) == manifest["splits"][name]["rows"]


def test_splits_share_no_rows(project: Config):
    dataset.build(project)

    frames = {name: dataset.load_split(name, project) for name in dataset.SPLIT_NAMES}
    keys = {
        name: set(map(tuple, frame[["year", "mileage", "price", "mark"]].to_numpy().tolist()))
        for name, frame in frames.items()
    }
    assert not keys["train_initial"] & keys["holdout_eval"]
    assert not keys["train_initial"] & keys["stream_pool"]
    assert not keys["holdout_eval"] & keys["stream_pool"]


def test_dataset_hash_is_stable_across_rebuilds(project: Config):
    dataset.build(project)
    first = dataset.dataset_hash(project)

    dataset.build(project)

    assert dataset.dataset_hash(project) == first


def test_build_ignores_the_installed_package_default_paths(project: Config, monkeypatch):
    """Regression guard: A3's module defaults resolve inside site-packages once installed.

    Nothing here may fall back to them — the failure would be silent (wrong data) rather
    than loud, so the default is pointed at a non-existent file on purpose.
    """
    from car_price_ml import config as a3_config

    monkeypatch.setattr(a3_config, "DATASET_CSV", project.root / "nowhere.csv")

    manifest = dataset.build(project)

    assert manifest["source"]["clean_rows"] > 0


def test_reading_a_split_before_building_says_what_to_run(project: Config):
    with pytest.raises(FileNotFoundError, match="dataset build"):
        dataset.load_split("holdout_eval", project)
