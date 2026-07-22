# mlops-car-price

[![CI](https://github.com/P0w3r223/mlops-car-price/actions/workflows/ci.yml/badge.svg)](https://github.com/P0w3r223/mlops-car-price/actions/workflows/ci.yml)

**Keeping a model alive instead of just training one.** This repository is the MLOps layer
around the used-car price model from
[car-price-ml](https://github.com/P0w3r223/car-price-ml): versioned data, recorded
training runs, drift monitoring on simulated production traffic, and a promotion rule
that decides — on evidence — when a challenger may replace the champion.

> **Status: session 1 of 6.** Data versioning and MLflow tracking are in place. The model
> registry, drift monitoring, the retraining loop and the versioned API land in the
> sessions that follow; the roadmap is in the issues.

## The problem

A model that scores well offline is not a system. The questions this project answers are
the ones that come *after* the notebook:

- Which exact rows produced this number, and can I get them back?
- Has the incoming data moved away from what the model was trained on — and is that
  "movement" real or just noise from a large sample?
- A new model looks better by 378 PLN. Is that an improvement, or a coin flip?
- What is deployed right now, and what does it cost to keep replacing it?

## Architecture

```
configs/config.yaml        every threshold, proportion and path
src/mlops_car_price/
  config.py                YAML -> frozen dataclasses, validated on load
  dataset.py               three-way split + content manifest (the data version)
  training/train.py        the only path to a trained model; one run = one record
```

The modelling code is not reimplemented here. `car_price_ml` is installed as a dependency
pinned to tag `v0.1.0` and supplies cleaning, feature engineering, the models and the
metrics. This repo owns everything around the model — which is what "MLOps" means here.

### The data split

The source is the open Kaggle dataset `aleksandrglotov/car-prices-poland` (CC0). It is
cleaned by A3's own code and split once, deterministically:

| Split | Rows | Purpose |
|---|---:|---|
| `train_initial` | 70 715 | what the first champion is trained on |
| `holdout_eval` | 23 571 | **frozen** — every champion/challenger comparison uses these exact rows |
| `stream_pool` | 23 573 | never trained on; becomes weekly "production" snapshots |

Each split is hashed into `data/processed/manifest.json`, and the hash of that manifest is
stamped on every training run. Two runs carrying the same dataset hash saw byte-identical
data — which is the only condition under which comparing them means anything.

## Results so far

Both models trained on `train_initial`, scored on the frozen holdout, seed 42:

| Model | MAE (PLN) | RMSE (PLN) | MAPE | R² | Train time | Artifact |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM | 9 280 | 21 226 | 14.7% | 0.935 | 1.8 s | **3.3 MB** |
| RandomForest | 8 901 | 21 148 | 14.5% | 0.936 | 7.6 s | **338.6 MB** |

RandomForest wins by 378 PLN (4.1% of MAE) and costs **103× more storage**. A weekly
retraining loop keeping a registry of those artifacts would be measured in gigabytes.
Two questions follow, and both are deferred to the sessions that can answer them
properly rather than guessed at now:

- Is the 378 PLN gap statistically real on 23 571 paired rows? → the promotion gate.
- Is it worth 335 MB per model version? → the artifact-cost ADR.

That is why `training.default_model` is LightGBM: the offline winner is not automatically
the model a maintenance loop can carry.

## Run it

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"
kaggle datasets download -d aleksandrglotov/car-prices-poland -p data/raw --unzip
python -m mlops_car_price.dataset build && python -m mlops_car_price.training.train
```

Then `mlflow ui --backend-store-uri sqlite:///mlflow.db` to inspect the runs.

## Technical decisions

Full context in [`docs/decisions/`](docs/decisions/).

- **[ADR 0001](docs/decisions/0001-consume-car-price-ml-as-a-package.md) — the modelling
  code is a dependency, not a copy.** A3 is installed from a git tag, so a run is
  reproducible against a fixed modelling version and there is exactly one definition of
  "clean the data".
- **[ADR 0002](docs/decisions/0002-replay-instead-of-a-scraper.md) — production traffic is
  a replay, not a scraper.** A3 rejected scraping the Polish listing sites (database
  *sui generis* right, ToS), and that decision does not expire because a later project
  would find fresh data convenient.
- **SQLite, not a file store, for MLflow.** The model registry does not exist on a file
  backend — a constraint that only surfaces at `register_model` time.
- **Drift gates on effect size, not p-values.** At 100k rows a KS test rejects for shifts
  far too small to matter. Measured, not asserted — session 4.

## Limitations

- **The production stream is simulated.** The drift scenarios are named, parameterised and
  documented, but they are generated, not observed. Nothing here proves the model degrades
  on the real 2026 Polish market — only that the system detects and reacts to degradation
  when it happens.
- **The source dataset has no listing date.** "Weeks" are a replay construct; the split is
  random, not chronological, so this project cannot demonstrate genuine temporal drift.
- **Prices are historical** (dataset vintage ~2021), and `age` is derived from a fixed
  reference year inherited from A3.

## Licence

MIT. Source data: `aleksandrglotov/car-prices-poland` (CC0-1.0).
