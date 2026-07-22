"""The config is the contract: a bad value must fail at load time, not mid-run."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mlops_car_price import config as config_module
from mlops_car_price.config import ConfigError, load


def write_config(tmp_path: Path, mutate) -> Path:
    """Copy the shipped config, apply a mutation, and return the temporary path."""
    raw = yaml.safe_load(config_module.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(raw)
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    path = configs_dir / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_shipped_config_loads_with_absolute_paths():
    config = load()

    assert config.root == config_module.PACKAGE_ROOT
    assert config.paths.raw_csv.is_absolute()
    assert config.paths.manifest.name == config_module.MANIFEST_FILENAME
    assert sum(config.splits.as_dict().values()) == pytest.approx(1.0)


def test_relative_paths_follow_the_config_file(tmp_path: Path):
    path = write_config(tmp_path, lambda raw: None)

    config = load(path)

    assert config.root == tmp_path
    assert config.paths.processed_dir.is_relative_to(tmp_path)


def test_splits_that_do_not_sum_to_one_are_rejected(tmp_path: Path):
    def inflate(raw):
        raw["splits"]["train_initial"] = 0.7

    with pytest.raises(ConfigError, match=r"sum to 1\.0"):
        load(write_config(tmp_path, inflate))


def test_missing_key_names_the_offender(tmp_path: Path):
    def drop(raw):
        del raw["promotion"]["alpha"]

    with pytest.raises(ConfigError, match=r"promotion\.alpha"):
        load(write_config(tmp_path, drop))


def test_out_of_range_alpha_is_rejected(tmp_path: Path):
    def bad_alpha(raw):
        raw["promotion"]["alpha"] = 1.5

    with pytest.raises(ConfigError, match=r"promotion\.alpha"):
        load(write_config(tmp_path, bad_alpha))


def test_tracking_uri_falls_back_to_local_sqlite(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    config = load(write_config(tmp_path, lambda raw: None))

    uri = config.mlflow.resolve_tracking_uri()

    # SQLite, not a file store: the model registry does not exist on a file backend.
    assert uri.startswith("sqlite:///")
    assert uri.endswith("mlflow.db")


def test_environment_overrides_the_configured_tracking_uri(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    config = load(write_config(tmp_path, lambda raw: None))

    assert config.mlflow.resolve_tracking_uri() == "http://localhost:5000"


def test_the_config_path_follows_the_deployment_not_the_package(tmp_path: Path, monkeypatch):
    """Regression guard, learned the hard way from a container.

    ``PACKAGE_ROOT`` is derived from the module's own location, so in an installed package
    it points inside site-packages and the shipped config is nowhere near it. The working
    directory and an environment variable both have to win over that.
    """
    written = write_config(tmp_path, lambda raw: None)
    monkeypatch.delenv(config_module.CONFIG_PATH_VARIABLE, raising=False)
    monkeypatch.chdir(tmp_path)

    assert config_module.default_config_path() == written
    assert load().root == tmp_path

    elsewhere = write_config(tmp_path / "other", lambda raw: None)
    monkeypatch.setenv(config_module.CONFIG_PATH_VARIABLE, str(elsewhere))

    assert config_module.default_config_path() == elsewhere
