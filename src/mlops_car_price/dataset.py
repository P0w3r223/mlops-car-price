"""Deterministic three-way split of the source dataset, with a content manifest.

The source is the same open Kaggle dataset the A3 model was built on, cleaned by the
*same* code (``car_price_ml.data``) so the MLOps layer never grows its own second
definition of "clean". The split has three purposes:

``train_initial``
    what the first champion is trained on.
``holdout_eval``
    frozen for the life of the project. Every champion/challenger comparison is decided
    on these exact rows, so a promotion decision measures the models, not the sampling.
``stream_pool``
    never trained on, never evaluated on — the reserve the replay generator turns into
    weekly "production" snapshots.

Each split is written with the SHA-256 of its bytes into ``manifest.json``; the hash of
that manifest is the dataset version stamped onto every training run. Two runs carrying
the same dataset hash saw byte-identical data.

    python -m mlops_car_price.dataset build
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from car_price_ml import config as a3_config
from car_price_ml import data as a3_data

from mlops_car_price.config import Config
from mlops_car_price.config import load as load_config

SPLIT_NAMES = ("train_initial", "holdout_eval", "stream_pool")
_HASH_CHUNK_BYTES = 1 << 20


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's contents, streamed so a large CSV is not held in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_indices(n_rows: int, proportions: dict[str, float], seed: int) -> dict[str, np.ndarray]:
    """Assign every row index to exactly one split, deterministically.

    The first two splits get their floored share and ``stream_pool`` absorbs the
    remainder, so the three parts are disjoint and together cover all ``n_rows``.
    """
    if n_rows < len(SPLIT_NAMES):
        raise ValueError(f"need at least {len(SPLIT_NAMES)} rows to split, got {n_rows}")

    shuffled = np.random.default_rng(seed).permutation(n_rows)
    n_train = int(n_rows * proportions["train_initial"])
    n_holdout = int(n_rows * proportions["holdout_eval"])
    if min(n_train, n_holdout, n_rows - n_train - n_holdout) == 0:
        raise ValueError(f"{n_rows} rows are too few for proportions {proportions}")

    return {
        "train_initial": shuffled[:n_train],
        "holdout_eval": shuffled[n_train : n_train + n_holdout],
        "stream_pool": shuffled[n_train + n_holdout :],
    }


def split_path(name: str, config: Config) -> Path:
    if name not in SPLIT_NAMES:
        raise KeyError(f"unknown split {name!r}; expected one of {SPLIT_NAMES}")
    return config.paths.processed_dir / f"{name}.parquet"


def build(config: Config) -> dict:
    """Clean the source CSV, write the three splits, and return the manifest.

    The A3 loader is always called with an explicit path: its module defaults resolve
    relative to the *installed* package, which lives in site-packages here.
    """
    if not config.paths.raw_csv.exists():
        raise FileNotFoundError(
            f"no source dataset at {config.paths.raw_csv} — see README (kaggle datasets download)"
        )

    raw = a3_data.load_raw(path=config.paths.raw_csv)
    clean = a3_data.clean(raw)

    indices = split_indices(len(clean), config.splits.as_dict(), config.seed)
    config.paths.processed_dir.mkdir(parents=True, exist_ok=True)

    splits: dict[str, dict] = {}
    for name in SPLIT_NAMES:
        path = split_path(name, config)
        frame = clean.iloc[np.sort(indices[name])].reset_index(drop=True)
        frame.to_parquet(path, index=False)
        splits[name] = {"rows": len(frame), "sha256": sha256_file(path)}

    manifest = {
        "car_price_ml_version": version("car-price-ml"),
        "reference_year": a3_config.REFERENCE_YEAR,
        "seed": config.seed,
        "proportions": config.splits.as_dict(),
        "source": {
            "name": config.paths.raw_csv.name,
            "raw_rows": len(raw),
            "clean_rows": len(clean),
            "sha256": sha256_file(config.paths.raw_csv),
        },
        "splits": splits,
    }
    # sort_keys keeps the manifest itself byte-stable, so its hash is a usable data version.
    config.paths.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def read_manifest(config: Config) -> dict:
    """Load the manifest, with a message that says how to create it if it is absent."""
    if not config.paths.manifest.exists():
        raise FileNotFoundError(
            f"no manifest at {config.paths.manifest} — "
            "run `python -m mlops_car_price.dataset build`"
        )
    return json.loads(config.paths.manifest.read_text(encoding="utf-8"))


def dataset_hash(config: Config) -> str:
    """SHA-256 of the manifest file: one short string identifying the exact data."""
    return sha256_file(config.paths.manifest)


def load_split(name: str, config: Config) -> pd.DataFrame:
    """Read one split; raises if the split was never built."""
    path = split_path(name, config)
    if not path.exists():
        raise FileNotFoundError(
            f"no {name} split at {path} — run `python -m mlops_car_price.dataset build`"
        )
    return pd.read_parquet(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["build"], help="build the splits and the manifest")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args(argv)

    config = load_config(args.config) if args.config else load_config()
    manifest = build(config)

    print(f"[dataset] source: {manifest['source']['name']}")
    print(f"[dataset] {manifest['source']['raw_rows']:,} raw rows -> "
          f"{manifest['source']['clean_rows']:,} after cleaning")
    for name in SPLIT_NAMES:
        print(f"[dataset]   {name:<14} {manifest['splits'][name]['rows']:>8,} rows")
    print(f"[dataset] manifest: {config.paths.manifest}")
    print(f"[dataset] dataset hash: {dataset_hash(config)[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
