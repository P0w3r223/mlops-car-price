# CLAUDE.md — mlops-car-price

Guidance for Claude Code (and any contributor) working in this repo.

## What this project is

Portfolio project **P1**. The MLOps layer around the used-car price model from project
A3: versioned data, recorded training runs, a model registry with champion/challenger
aliases, drift monitoring on simulated production traffic, and a promotion rule that
decides when a challenger may take over. The claim is not "I trained a model" but
"I maintain a model that ages on data — and I have a process for it".

## Architecture

```
configs/config.yaml          # every threshold, proportion and path — no literals in code
src/mlops_car_price/
  config.py                  # YAML -> frozen dataclasses, validated on load
  dataset.py                 # three-way split + content manifest (the data version)
  tracking.py                # MLflow wiring: tracking URI, experiment, client
  registry.py                # model versions + champion/challenger aliases
  replay.py                  # weekly "production" snapshots + named drift scenarios
  drift/metrics.py           # PSI, Wasserstein, KS, chi-square, missing rates
  drift/detector.py          # thresholds + calibration -> a verdict with reasons
  drift/evaluate.py          # false alarms and power for three competing detectors
  drift/report.py            # the verdict as markdown
  training/train.py          # the only way a model gets trained; one run = one record
  training/promote.py        # the gate: score, judge, move the alias, record why
  training/retrain.py        # the loop: drift -> challenger -> gate -> usually nothing
  api/main.py                # serves the champion alias; no model in the image
examples/                    # scripts that regenerate the README's tables
tests/
docs/decisions/              # ADRs
```

Data flows one way: `dataset` writes the splits and the manifest, `training` reads them
and records a run. Nothing writes back into the source data.

Vocabulary that must stay straight: a **run** is one training execution, a **model
version** is an artifact entered into the registry, an **alias** (`champion`,
`challenger`) is a movable label saying which version serves. Registering is proposing;
moving the `champion` alias is deploying.

The modelling code is **not** reimplemented here. `car_price_ml` (project A3, pinned to
tag `v0.1.0`) supplies cleaning, feature engineering, the model bake-off and the metrics.
This repo owns everything *around* the model.

## Commands

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"
kaggle datasets download -d aleksandrglotov/car-prices-poland -p data/raw --unzip
python -m mlops_car_price.dataset build            # splits + manifest
python -m mlops_car_price.training.train --model LightGBM --register
python -m mlops_car_price.training.promote --version 1 [--dry-run]
python -m mlops_car_price.replay --week 1 --scenario price_shock
python -m mlops_car_price.drift.detector --week 1 --scenario price_shock
python -m mlops_car_price.training.retrain --weeks 11 --scenario mileage_shift
uvicorn mlops_car_price.api.main:app                # serve locally
docker compose up -d mlflow && docker compose --profile bootstrap run --rm bootstrap
python examples/drift_scenarios.py                 # regenerate the scenario table
python examples/detector_evaluation.py             # false alarms + power (~25 min)
python examples/artifact_cost.py                   # regenerate the cost table
mlflow ui --backend-store-uri sqlite:///mlflow.db  # inspect runs and versions
pytest                                             # full suite
ruff check .                                       # lint
```

## Code rules

- **Typed, documented, no magic numbers.** Anything tunable lives in `config.yaml` and
  arrives through `config.load()`; a bad value fails at load time, naming the key.
- **Never rely on A3's default paths.** `car_price_ml.config.PROJECT_ROOT` resolves
  inside `site-packages` once the package is installed. Always pass the path explicitly
  (`data.load_raw(path=…)`, `model.save_model(…, models_dir=…)`). A regression test
  guards this, because the failure mode is silent — wrong data, not a crash.
- **Every training goes through `training/train.py`.** No notebooks, no ad-hoc scripts.
  A run that is not recorded cannot be compared, and comparison is the point.
- **Every run logs the dataset hash.** Two runs are comparable only if that hash matches.
- **Separate I/O from logic.** Split assignment, metrics and thresholds are pure
  functions over data; only the build and the run touch disk.

## Methodology rules (do not violate)

- **`holdout_eval` is frozen.** Champion and challenger are always scored on those exact
  rows. Never retrain on it, never resample it, never "refresh" it.
- **`stream_pool` is never trained on.** It exists to become production traffic; using it
  for training would make drift detection self-fulfilling.
- **Drift gates run on effect size, not p-values.** At these sample sizes a KS test rejects
  for shifts far too small to matter; the p-value is a diagnostic, PSI and Wasserstein decide.
- **A monitor is a classifier; measure both its error rates.** Any change to the detector
  or its thresholds means rerunning `examples/detector_evaluation.py` - a rule nobody
  measured is a rule nobody can defend. Fixed thresholds alarm on 100% of unshifted weeks
  below 1 000 rows; the calibrated detector on 0-7.5% at the same sizes with identical power.
- **A threshold has to beat the noise floor too.** PSI's null distribution depends on sample
  size and category count, so a column is flagged only when it clears both the configured
  threshold and what it scores against itself at that sample size (ADR 0006). Never silence
  a false alarm by raising the fixed threshold - measure the floor instead.
- **`stream_pool` is the only source of snapshots**, and a scenario is a parameterised
  transformation applied after the draw. Adding a scenario means adding a way for data to
  break, not a way to make a detector look good.
- **A challenger is promoted only on evidence**: the paired bootstrap CI of the MAE
  difference excludes zero, the point improvement clears the configured margin, and the
  holdout is large enough. The comparison is paired because both models score the same rows;
  measured on real data, the unpaired interval is 3.3x wider and nearly spans zero on a
  genuine improvement (ADR 0007). Never swap in `bootstrap_diff` here.
- **Serving code names an alias, never a version.** `models:/car-price@champion` is the only
  model reference the API may contain; a promotion plus a restart is the whole deployment.
- **Configuration belongs to the deployment, not the package.** `config.load()` resolves the
  env var, then the working directory, then the source tree - `PACKAGE_ROOT` points inside
  site-packages once installed, which is the same trap ADR 0001 documents for A3.
- **Refusing is the normal outcome.** A retraining loop whose challengers are always promoted
  is a deployment script, not a quality bar. Do not tune the gate until it says yes.
- **The deployment budget is a promotion rule, not advice.** A candidate over
  `promotion.max_artifact_mb` is refused however well it scores (ADR 0003). Raising the
  budget is a reviewed config change, never a workaround inside a run.
- **A candidate is re-scored before it is judged.** If a stored artifact no longer
  reproduces the MAE its own run recorded, nothing downstream is trustworthy — refuse.
- **Reproducibility is a claim to be tested.** Same seed and same data must give the same
  number; when that broke it was a real bug in the modelling layer, fixed there (ADR 0004).
- **The A3 methodology still applies** where this repo touches modelling: log-price target
  inverted before metrics, `age` instead of raw `year`, out-of-fold target encoding.

## Working rules

- Plan before code; discuss an architecture change before writing it.
- Small commits, Conventional Commits, one PR per session, with a "why".
- Non-trivial choices become an ADR in `docs/decisions/`: context, options, decision,
  consequences.
- Claude Code is also the translator: every concept here (run vs model version vs alias,
  PSI, KS, champion/challenger) has to be defensible out loud, without the code open.

## What not to do

- Do not reimplement anything `car_price_ml` already provides — extend it upstream instead.
- Do not commit data, models, `mlflow.db` or `mlruns/`.
- Do not log a RandomForest artifact to MLflow by default: it serialises to ~563 MB and a
  run history of them is measured in gigabytes. `--log-model` is opt-in on purpose.
- Do not change a public signature or a config key without asking.
- Do not weaken a drift or promotion threshold to make a run pass. A firing detector is a
  finding, not a flaky test.
