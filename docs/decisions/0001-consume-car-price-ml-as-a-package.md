# ADR 0001 — Consume car-price-ml as a pinned dependency, not a copy

Date: 2026-07-22
Status: accepted
Author: P0w3r223
Related to: [car-price-ml](https://github.com/P0w3r223/car-price-ml) tag `v0.1.0`

---

## Context

This project adds an MLOps layer — tracking, registry, drift, retraining — around a model
that already exists in project A3. A3 is a healthy repository: `src/car_price_ml/` holds
cleaning, feature engineering, the model bake-off and the PLN metrics, with tests and a
documented methodology. The MLOps layer needs all of it, and needs it to stay identical
over time: a drift report or a promotion decision is meaningless if the definition of
"clean the data" quietly changed underneath it.

## Options

1. **Copy the modelling modules into this repo.** No dependency plumbing, everything
   visible in one tree. But the same logic then lives in two places, and the copy starts
   drifting from the original the first time either side is touched. The methodology rules
   in A3's CLAUDE.md would have to be maintained twice.
2. **Extend A3 in place** with an `mlops/` package. Strongest "I maintain my projects"
   signal in the commit history, and no dependency at all. But it merges two concerns into
   one repository — the model and the system that operates it — and A3 is deliberately a
   teaching-sized full-ML-cycle project.
3. **Install A3 as a package, pinned to a git tag.** One definition of the model, and the
   pin makes a run reproducible against a fixed modelling version. Costs a dependency on a
   git URL and forces this repo to respect A3's public API.

## Decision

Option 3. `pyproject.toml` declares
`car-price-ml @ git+https://github.com/P0w3r223/car-price-ml@v0.1.0`, and A3 was tagged
`v0.1.0` for the purpose. Every training run records the resolved version of that package
alongside the dataset hash, so a run identifies both the data and the code that produced it.

## Consequences

- **A3's public API is now a contract.** Changing a signature there breaks this repo, which
  is the correct incentive: it turns "quick tweak" into "version bump".
- **Reuse required no changes to A3.** Every function that touches disk already accepts an
  explicit path (`data.load_raw(path=…)`, `model.save_model(…, models_dir=…)`), so the
  package works unchanged when installed.
- **One sharp edge, guarded by a test.** `car_price_ml.config.PROJECT_ROOT` is derived from
  the module file location, so after installation the *default* paths point into
  `site-packages`. Relying on them would read the wrong data silently rather than crash.
  This repo always passes paths explicitly, and a regression test points A3's default at a
  non-existent file to prove nothing falls back to it.
- **Improvements flow upstream, not sideways.** When this project needs something the
  modelling layer should own, it gets added to A3 and the pin moves — it is not
  reimplemented here.
