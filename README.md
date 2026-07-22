# mlops-car-price

[![CI](https://github.com/P0w3r223/mlops-car-price/actions/workflows/ci.yml/badge.svg)](https://github.com/P0w3r223/mlops-car-price/actions/workflows/ci.yml)

**Keeping a model alive instead of just training one.** This repository is the MLOps layer
around the used-car price model from
[car-price-ml](https://github.com/P0w3r223/car-price-ml): versioned data, recorded
training runs, drift monitoring on simulated production traffic, and a promotion rule
that decides — on evidence — when a challenger may replace the champion.

> **Status: session 2 of 6.** Data versioning, MLflow tracking, the model registry and the
> promotion gate are in place. Drift monitoring, the retraining loop and the versioned API
> land in the sessions that follow; the roadmap is in the issues.

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
  tracking.py              MLflow wiring: tracking URI, experiment, client
  registry.py              versions and the champion/challenger aliases
  training/train.py        the only path to a trained model; one run = one record
  training/promote.py      the gate: score, judge, move the alias, record why
examples/artifact_cost.py  regenerates the operational cost table below
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

## What a model costs to operate

Accuracy is one column. Trained on `train_initial`, scored on the frozen holdout, seed 42
(regenerate with `python examples/artifact_cost.py`):

| Model | Holdout MAE | Train | Artifact | Load | Predict p50 | Predict p95 | Batch | Registry / year |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ridge | 15 422 PLN | 0.1 s | 0.0 MB | 0.00 s | 5.1 ms | 6.1 ms | 728k rows/s | 0.0 GB |
| LightGBM | 9 278 PLN | 1.7 s | 3.3 MB | 0.03 s | 12.9 ms | 13.7 ms | 119k rows/s | 0.2 GB |
| RandomForest | 8 908 PLN | 7.8 s | 338.5 MB | 0.44 s | 47.1 ms | 68.2 ms | 148k rows/s | 17.2 GB |

RandomForest is 370 PLN (4.0%) better and costs 103× the storage, 5× the p95 request
latency and 17.2 GB of registry a year at one retraining a week. So the deployment budget
is enforced by the gate, not by a paragraph — the real candidate was refused:

```
[promote] candidate v2 (RandomForest): MAE 8908.2 PLN
[promote] champion  v1 (LightGBM): MAE 9278.0 PLN
[promote] reason: artifact is 338.5 MB, over the 50.0 MB deployment budget
[promote] REJECTED - candidate v2, delta +369.8 PLN
```

## The bug this project found in its own foundation

Two runs of the same model, same seed, same rows, kept disagreeing:

```
RandomForest, random_state=42: 8841.2 PLN, then 8914.1 PLN
LightGBM,     random_state=42: 9331.0 / 9266.7 / 9278.2 PLN
```

The estimators were seeded; the **preprocessing was not**. A3's target encoder shuffled its
internal cross-fitting folds from an unseeded RNG, so identical data produced different
encodings. A ~70 PLN spread is nothing next to a 9 000 PLN MAE — and everything next to a
100 PLN promotion margin. The gate would have been reading noise part of the time.

Fixed upstream in [car-price-ml#3](https://github.com/P0w3r223/car-price-ml/pull/3)
(v0.1.1), not worked around here ([ADR 0004](docs/decisions/0004-reproducibility-fixed-upstream.md)).
Runs now reproduce to the decimal. The dataset hash changed with the pin, which is exactly
what a data version should do when the code that cleans the data changes.

## How a model reaches production

```bash
python -m mlops_car_price.training.train --model LightGBM --register  # -> version, alias challenger
python -m mlops_car_price.training.promote --version 1                # -> judged, alias moved
```

Registering is proposing, not deploying. The gate scores the candidate **and** the current
champion on the frozen holdout and refuses unless every rule passes: a holdout large enough
to judge on, an artifact that reproduces the MAE its own run recorded, the same dataset
version on both sides, the deployment budget, and an improvement past the configured margin.
Serving code asks for `models:/car-price@champion` and never names a version.

One rule is deliberately still missing: whether an improvement is bigger than noise. A margin
in złoty is not a significance test — the paired bootstrap over the per-row errors arrives in
session 5, which is why scoring already keeps them.

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
- **[ADR 0003](docs/decisions/0003-lightgbm-as-the-production-model.md) — LightGBM serves,
  and the budget is in the gate.** The offline winner is not automatically the model a
  maintenance loop can carry; 0.23 pp of MAPE bought two orders of magnitude on every
  operation the loop performs.
- **[ADR 0004](docs/decisions/0004-reproducibility-fixed-upstream.md) — the determinism bug
  was fixed in A3, not patched around here.** Duplicating the preprocessing would have
  defeated the point of consuming it as a package.
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
