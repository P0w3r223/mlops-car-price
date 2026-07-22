# ADR 0003 — LightGBM serves, and the deployment budget is enforced by the gate

Date: 2026-07-22
Status: accepted
Author: P0w3r223 + Claude
Related to: ADR 0001, `examples/artifact_cost.py`, `reports/artifact_cost.md`

---

## Context

Project A3 ran a bake-off and RandomForest won: best MAE, and the model that was served.
That answer was correct for A3, where a model is trained once and deployed once.

This project retrains on a schedule, keeps every candidate as a registry version, loads the
champion on service start and ships it inside a container image. Those operations have a
price that offline accuracy does not show, so they were measured (`examples/artifact_cost.py`,
70 715 training rows, holdout of 23 571):

| Model | Holdout MAE | Train | Artifact | Load | Predict p50 | Predict p95 | Batch | Registry / year |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ridge | 15 422 PLN | 0.1 s | 0.0 MB | 0.00 s | 5.1 ms | 6.1 ms | 728k rows/s | 0.0 GB |
| LightGBM | 9 278 PLN | 1.7 s | 3.3 MB | 0.03 s | 12.9 ms | 13.7 ms | 119k rows/s | 0.2 GB |
| RandomForest | 8 908 PLN | 7.8 s | 338.5 MB | 0.44 s | 47.1 ms | 68.2 ms | 148k rows/s | 17.2 GB |

RandomForest is **370 PLN (4.0%) better** and costs **103× the storage**, **5× the p95
request latency** and **15× the cold start**. At one retraining a week it accumulates
**17.2 GB of registry a year** against LightGBM's 0.2 GB.

In relative terms the accuracy gap is 14.46% MAPE against 14.69% — 0.23 percentage points
on a valuation that is already a five-figure estimate with a wide interval.

## Options

1. **RandomForest stays champion.** Best offline number. Requires an artifact store measured
   in tens of gigabytes a year, a slower API and a service that takes half a second longer
   to become ready after every deploy.
2. **LightGBM becomes champion.** Gives up 0.23 pp of MAPE. Every operation the maintenance
   loop performs gets cheaper by one to two orders of magnitude.
3. **Shrink the forest** (fewer trees, larger leaves) to fit the budget. Plausible middle
   ground, but unmeasured — and it trades the same accuracy for size, just less legibly.
4. **Keep RandomForest but store only the newest version.** Cheap, and it destroys the
   history that makes rollback and "why is this serving?" answerable.

## Decision

Option 2. `training.default_model` is LightGBM, and the promotion gate refuses any candidate
whose artifact exceeds `promotion.max_artifact_mb` (50 MB) **regardless of its MAE**.

The budget lives in the gate rather than in this document on purpose. A rule written only in
prose is followed until the week someone is in a hurry; a rule in the gate produces a logged
refusal. The real RandomForest candidate was rejected with:

```
[promote] candidate v2 (RandomForest): MAE 8908.2 PLN
[promote] champion  v1 (LightGBM): MAE 9278.0 PLN
[promote] reason: artifact is 338.5 MB, over the 50.0 MB deployment budget
[promote] REJECTED - candidate v2, delta +369.8 PLN
```

## Consequences

- **We knowingly serve the second-best model offline.** The trade is stated in the README, not
  buried: 0.23 pp of MAPE bought a system that can be retrained weekly and rolled back.
- **Raising the budget is a deliberate act.** It is a config value under review, not a habit.
- **RandomForest keeps its role** as the accuracy reference in the cost table: it says how much
  the operational constraint costs, which is a number worth knowing.
- **Option 3 is unfinished business**, tracked as an issue: a size-capped forest might recover
  part of the gap inside the budget.
- **The 370 PLN gap is not yet known to be real.** It is a point estimate on one holdout; the
  paired bootstrap in session 5 decides whether a difference that size survives the noise —
  and the pipeline's own reproducibility noise was of the same order until it was fixed
  upstream (`car-price-ml` v0.1.1).
