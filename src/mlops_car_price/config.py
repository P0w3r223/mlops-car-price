"""Typed configuration loaded from ``configs/config.yaml``.

Nothing in this package hardcodes a threshold, a proportion or a path: the YAML file is
the single source of truth and it arrives here as frozen dataclasses that validate
themselves on construction. A bad config therefore fails at load time with a message
naming the offending key, rather than halfway through a training run.

Relative paths are resolved against the **repository root**, defined as the parent of the
directory holding the config file. That rule keeps a config copied into a temporary
directory self-contained (its paths follow it), which is what the tests rely on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "config.yaml"
CONFIG_PATH_VARIABLE = "MLOPS_CAR_PRICE_CONFIG"

MANIFEST_FILENAME = "manifest.json"
_PROPORTION_TOLERANCE = 1e-9


class ConfigError(ValueError):
    """Raised when the config file is missing a key or carries an impossible value."""


@dataclass(frozen=True)
class SplitConfig:
    """Proportions of the three-way dataset split; must add up to one."""

    train_initial: float
    holdout_eval: float
    stream_pool: float

    def __post_init__(self) -> None:
        for name in ("train_initial", "holdout_eval", "stream_pool"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ConfigError(f"splits.{name} must be in (0, 1), got {value}")
        total = self.train_initial + self.holdout_eval + self.stream_pool
        if abs(total - 1.0) > _PROPORTION_TOLERANCE:
            raise ConfigError(f"splits must sum to 1.0, got {total}")

    def as_dict(self) -> dict[str, float]:
        return {
            "train_initial": self.train_initial,
            "holdout_eval": self.holdout_eval,
            "stream_pool": self.stream_pool,
        }


@dataclass(frozen=True)
class PathsConfig:
    """Absolute paths derived from the repository root."""

    raw_csv: Path
    processed_dir: Path
    snapshots_dir: Path
    models_dir: Path
    reports_dir: Path

    @property
    def manifest(self) -> Path:
        """Content manifest of the split — the dataset version of the whole project."""
        return self.processed_dir / MANIFEST_FILENAME


@dataclass(frozen=True)
class MlflowConfig:
    """Where runs are recorded and where model versions live."""

    experiment: str
    registered_model: str
    backend_store: Path
    artifact_root: Path
    tracking_uri: str | None

    def resolve_tracking_uri(self) -> str:
        """Environment override → explicit URI → local SQLite file.

        SQLite rather than a file store because MLflow's model registry is unavailable on
        a file backend — a constraint that only surfaces at ``register_model`` time.
        """
        from_env = os.environ.get("MLFLOW_TRACKING_URI")
        if from_env:
            return from_env
        if self.tracking_uri:
            return self.tracking_uri
        return f"sqlite:///{self.backend_store.as_posix()}"


@dataclass(frozen=True)
class TrainingConfig:
    """Defaults for a training run."""

    default_model: str
    sample_rows: int | None

    def __post_init__(self) -> None:
        if self.sample_rows is not None and self.sample_rows <= 0:
            raise ConfigError(
                f"training.sample_rows must be positive or null, got {self.sample_rows}"
            )


@dataclass(frozen=True)
class ReplayConfig:
    """How the simulated production stream is drawn."""

    snapshot_rows: int

    def __post_init__(self) -> None:
        _require_positive(self.snapshot_rows, "replay.snapshot_rows")


@dataclass(frozen=True)
class DriftConfig:
    """Thresholds that turn drift metrics into an alert."""

    psi_threshold: float
    wasserstein_threshold: float
    ks_p_value_threshold: float
    min_snapshot_rows: int
    max_missing_rate_increase: float
    max_mae_increase_pct: float
    calibration_resamples: int
    calibration_quantile: float

    def __post_init__(self) -> None:
        if self.calibration_resamples < 0:
            raise ConfigError(
                f"drift.calibration_resamples must be zero or positive, "
                f"got {self.calibration_resamples}"
            )
        if not 0.0 < self.calibration_quantile < 1.0:
            raise ConfigError(
                f"drift.calibration_quantile must be in (0, 1), got {self.calibration_quantile}"
            )
        _require_positive(self.psi_threshold, "drift.psi_threshold")
        _require_positive(self.wasserstein_threshold, "drift.wasserstein_threshold")
        _require_positive(self.min_snapshot_rows, "drift.min_snapshot_rows")
        _require_positive(self.max_missing_rate_increase, "drift.max_missing_rate_increase")
        _require_positive(self.max_mae_increase_pct, "drift.max_mae_increase_pct")
        if not 0.0 < self.ks_p_value_threshold < 1.0:
            raise ConfigError(
                f"drift.ks_p_value_threshold must be in (0, 1), got {self.ks_p_value_threshold}"
            )


@dataclass(frozen=True)
class PromotionConfig:
    """Rules a challenger has to clear before it becomes the champion."""

    min_mae_improvement_pln: float
    alpha: float
    bootstrap_resamples: int
    min_holdout_rows: int
    max_artifact_mb: float
    mae_reproduction_tolerance_pln: float

    def __post_init__(self) -> None:
        _require_positive(self.min_mae_improvement_pln, "promotion.min_mae_improvement_pln")
        _require_positive(self.max_artifact_mb, "promotion.max_artifact_mb")
        _require_positive(self.bootstrap_resamples, "promotion.bootstrap_resamples")
        _require_positive(self.min_holdout_rows, "promotion.min_holdout_rows")
        _require_positive(
            self.mae_reproduction_tolerance_pln, "promotion.mae_reproduction_tolerance_pln"
        )
        if not 0.0 < self.alpha < 1.0:
            raise ConfigError(f"promotion.alpha must be in (0, 1), got {self.alpha}")


@dataclass(frozen=True)
class Config:
    """The whole configuration, plus the root every relative path was resolved against."""

    root: Path
    seed: int
    paths: PathsConfig
    splits: SplitConfig
    mlflow: MlflowConfig
    training: TrainingConfig
    replay: ReplayConfig
    drift: DriftConfig
    promotion: PromotionConfig


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ConfigError(f"{name} must be positive, got {value}")


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw.get(name)
    if not isinstance(section, dict):
        raise ConfigError(f"config section '{name}' is missing or not a mapping")
    return section


def _key(section: dict[str, Any], section_name: str, key: str) -> Any:
    if key not in section:
        raise ConfigError(f"config key '{section_name}.{key}' is missing")
    return section[key]


def _resolve(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{name} must be a non-empty path string, got {value!r}")
    path = Path(value)
    return path if path.is_absolute() else (root / path)


def default_config_path() -> Path:
    """Where to look for the config when the caller does not say.

    Environment variable first, then the working directory, then the source tree. The order
    exists because ``PACKAGE_ROOT`` is derived from this file's location, so in an installed
    package it points inside ``site-packages`` — the exact trap ADR 0001 documents for the
    modelling layer, which this package walked into the first time it ran from a container.
    Configuration belongs to the deployment, not to the code.
    """
    from_env = os.environ.get(CONFIG_PATH_VARIABLE)
    if from_env:
        return Path(from_env)
    from_working_directory = Path.cwd() / "configs" / "config.yaml"
    if from_working_directory.exists():
        return from_working_directory
    return DEFAULT_CONFIG_PATH


def load(path: Path | str | None = None) -> Config:
    """Read, validate and resolve the config file.

    Args:
        path: Config file to read; defaults to :func:`default_config_path`. Its parent's
            parent becomes the root for relative paths.
    """
    config_path = Path(path if path is not None else default_config_path()).resolve()
    if not config_path.exists():
        raise ConfigError(f"no config file at {config_path}")
    root = config_path.parent.parent

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} does not contain a YAML mapping")

    paths_section = _section(raw, "paths")
    splits_section = _section(raw, "splits")
    mlflow_section = _section(raw, "mlflow")
    training_section = _section(raw, "training")
    replay_section = _section(raw, "replay")
    drift_section = _section(raw, "drift")
    promotion_section = _section(raw, "promotion")

    if "seed" not in raw:
        raise ConfigError("config key 'seed' is missing")

    return Config(
        root=root,
        seed=int(raw["seed"]),
        paths=PathsConfig(
            raw_csv=_resolve(root, _key(paths_section, "paths", "raw_csv"), "paths.raw_csv"),
            processed_dir=_resolve(
                root, _key(paths_section, "paths", "processed_dir"), "paths.processed_dir"
            ),
            snapshots_dir=_resolve(
                root, _key(paths_section, "paths", "snapshots_dir"), "paths.snapshots_dir"
            ),
            models_dir=_resolve(
                root, _key(paths_section, "paths", "models_dir"), "paths.models_dir"
            ),
            reports_dir=_resolve(
                root, _key(paths_section, "paths", "reports_dir"), "paths.reports_dir"
            ),
        ),
        splits=SplitConfig(
            train_initial=float(_key(splits_section, "splits", "train_initial")),
            holdout_eval=float(_key(splits_section, "splits", "holdout_eval")),
            stream_pool=float(_key(splits_section, "splits", "stream_pool")),
        ),
        mlflow=MlflowConfig(
            experiment=str(_key(mlflow_section, "mlflow", "experiment")),
            registered_model=str(_key(mlflow_section, "mlflow", "registered_model")),
            backend_store=_resolve(
                root, _key(mlflow_section, "mlflow", "backend_store"), "mlflow.backend_store"
            ),
            artifact_root=_resolve(
                root, _key(mlflow_section, "mlflow", "artifact_root"), "mlflow.artifact_root"
            ),
            tracking_uri=_key(mlflow_section, "mlflow", "tracking_uri"),
        ),
        training=TrainingConfig(
            default_model=str(_key(training_section, "training", "default_model")),
            sample_rows=_key(training_section, "training", "sample_rows"),
        ),
        replay=ReplayConfig(
            snapshot_rows=int(_key(replay_section, "replay", "snapshot_rows")),
        ),
        drift=DriftConfig(
            psi_threshold=float(_key(drift_section, "drift", "psi_threshold")),
            wasserstein_threshold=float(_key(drift_section, "drift", "wasserstein_threshold")),
            ks_p_value_threshold=float(_key(drift_section, "drift", "ks_p_value_threshold")),
            min_snapshot_rows=int(_key(drift_section, "drift", "min_snapshot_rows")),
            max_missing_rate_increase=float(
                _key(drift_section, "drift", "max_missing_rate_increase")
            ),
            max_mae_increase_pct=float(_key(drift_section, "drift", "max_mae_increase_pct")),
            calibration_resamples=int(_key(drift_section, "drift", "calibration_resamples")),
            calibration_quantile=float(_key(drift_section, "drift", "calibration_quantile")),
        ),
        promotion=PromotionConfig(
            min_mae_improvement_pln=float(
                _key(promotion_section, "promotion", "min_mae_improvement_pln")
            ),
            alpha=float(_key(promotion_section, "promotion", "alpha")),
            bootstrap_resamples=int(_key(promotion_section, "promotion", "bootstrap_resamples")),
            min_holdout_rows=int(_key(promotion_section, "promotion", "min_holdout_rows")),
            max_artifact_mb=float(_key(promotion_section, "promotion", "max_artifact_mb")),
            mae_reproduction_tolerance_pln=float(
                _key(promotion_section, "promotion", "mae_reproduction_tolerance_pln")
            ),
        ),
    )
