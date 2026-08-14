# ADR 0007 — A challenger has to beat the champion by more than sampling noise

Date: 2026-07-22
Status: accepted
Author: P0w3r223
Related to: ADR 0003, [ab-lab ADR 0005](https://github.com/P0w3r223/ab-lab/blob/main/docs/decisions/0005-paired-bootstrap.md)

---

## Context

Until now the gate compared two point estimates against a margin in złoty: promote when the
candidate's MAE is at least 100 PLN lower. That answers "is this worth the swap?" and says
nothing at all about "could this be luck?".

The question is not academic here. Session 2 found that the pipeline's own reproducibility
noise was ~70 PLN before it was fixed upstream — the same order as the margin. And the
holdout is a sample: two models can differ by a few hundred złoty on 23 571 cars purely
because of which cars those are.

## Options

1. **Widen the margin until noise cannot cross it.** No new machinery, and no way to know
   what the right number is — it would be a guess about a distribution nobody measured.
2. **Paired t-test on the per-row error differences.** Cheap and standard, but absolute
   errors are strongly right-skewed and the difference inherits that; the t-interval is a
   large-sample approximation whose failure mode here is hard to state.
3. **Paired bootstrap of the difference in mean absolute error.** No distributional
   assumption beyond "the sample stands in for the population", and the pairing is explicit.
4. **Unpaired bootstrap** (`bootstrap_diff`, already available). Wrong tool: champion and
   challenger score the *same* cars, so the two error vectors are not independent samples.

## Decision

Option 3. A candidate must clear the configured margin **and** its improvement must survive
a paired bootstrap: the 95% interval for `champion_MAE − candidate_MAE` has to exclude zero.
Both conditions, because they answer different questions and either alone promotes the wrong
thing.

The method did not exist in the statistics package this portfolio already had, so it was
added there (`ab-lab` 0.2.0, `paired_bootstrap`) rather than written a second time here.

## Why paired, measured rather than argued

On the real comparison — LightGBM champion against a RandomForest challenger, both scored on
the same 23 571 holdout rows:

| | 95% interval for the improvement | width | p |
|---|---|---:|---:|
| paired bootstrap | (+264.4, +474.9) PLN | 210.5 | 0.0002 |
| unpaired bootstrap | (+28.3, +715.2) PLN | 686.8 | 0.0342 |

The per-row errors correlate at **0.906** — unsurprisingly, since the same unusual cars are
hard for both models. Ignoring that correlation makes the interval **3.3× wider** and pushes
its lower bound to +28 PLN, a hair from zero. On a slightly smaller holdout the unpaired
procedure would have refused a model that is genuinely better. That is the cost of using the
independent-samples tool on paired data, in the units this project cares about.

## Consequences

- **The 370 PLN question has an answer: it is real**, not sampling noise. The RandomForest
  challenger is genuinely more accurate — and is still refused, by the deployment budget in
  ADR 0003. The system now states an explicit trade ("a real 4% accuracy gain, declined
  because we cannot operate a 338 MB artifact") instead of an accidental one.
- **The cheap checks run first.** The bootstrap is 10 000 resamples over 23 571 rows; a
  candidate that already failed the margin never reaches it.
- **The interval is recorded with the decision** — `delta_ci_low`, `delta_ci_high` and
  `delta_p_value` land on the promotion run in MLflow, so a past promotion can be
  re-examined without recomputing anything.
- **A gate can now refuse for a new reason**, and the reason is worth reading: "improves by
  X PLN, but the 95% interval spans zero on N paired rows".
- **This does not make promotion safe, only honest.** The interval covers sampling variation
  on one frozen holdout. It says nothing about whether that holdout still resembles next
  month's traffic — which is what the drift monitoring is for.
