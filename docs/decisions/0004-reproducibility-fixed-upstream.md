# ADR 0004 — Reproducibility is fixed upstream, not worked around here

Date: 2026-07-22
Status: accepted
Author: P0w3r223
Related to: ADR 0001, [car-price-ml#3](https://github.com/P0w3r223/car-price-ml/pull/3)

---

## Context

While building the promotion gate, the same model trained twice on identical data with an
identical seed produced different holdout MAE:

```
RandomForest, random_state=42, same 70 715 rows: 8841.2 PLN, then 8914.1 PLN
LightGBM,     random_state=42, same 70 715 rows: 9331.0 / 9266.7 / 9278.2 PLN
```

The estimators were seeded; the **preprocessing** was not. A3's `build_preprocessor` handed
scikit-learn's `TargetEncoder` a plain integer for `cv`, which lets it shuffle its internal
cross-fitting folds from an unseeded RNG. Two fits on the same frame returned different
encodings, and every metric downstream inherited the wobble.

A spread of ~70 PLN is small next to a 9 000 PLN MAE. It is *not* small next to the things
this project decides with: the promotion margin is 100 PLN, and the RandomForest/LightGBM gap
is 370 PLN. A gate comparing candidates at that resolution would have been reading noise
part of the time.

## Options

1. **Work around it here** — average several training runs per candidate, or widen the
   promotion margin past the noise. Cheap locally, and it leaves a broken guarantee in the
   modelling package: A3 still claims a seed reproduces a result.
2. **Fork the preprocessing** into this repo with the seed fixed. Fast, and it starts the
   duplication that ADR 0001 exists to prevent.
3. **Fix it in A3 and move the pin.** The defect belongs to the modelling layer, so it gets
   fixed there, with a regression test asserting two fits agree exactly.

## Decision

Option 3. `car-price-ml` v0.1.1 passes an explicitly seeded `KFold` splitter to the target
encoder; `build_preprocessor` takes `random_state` and defaults it to the project seed. This
repo's dependency pin moved to `@v0.1.1`.

## Consequences

- **Reproducibility now holds end to end**: preprocessing output is identical across fits, and
  repeated runs return the same MAE to the decimal.
- **The dataset hash changed** (`3d351946b2b5` → `c5134841bd46`) because the manifest records
  the version of the cleaning code. That is the intended behaviour of a data version: a
  different pipeline is different data, and runs from before the fix are not silently
  comparable with runs after it.
- **A3's published metrics predate the fix** and will move by well under a percent when
  refreshed; tracked as an issue there rather than bundled into this work.
- **The measurement stays worth repeating.** "Same seed, same data, same result" is a claim
  that should be tested, not assumed — and it was false here in a repository that already
  had a commit titled *"reproducible training"*.
- Removing this noise does **not** remove the need for a statistical promotion gate. It
  removes one source of variation; sampling variation on the holdout remains, and that is
  what session 5's paired bootstrap is for.
