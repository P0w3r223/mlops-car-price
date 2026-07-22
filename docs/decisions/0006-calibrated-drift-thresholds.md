# ADR 0006 — Drift thresholds are calibrated against the null, not taken from a rule of thumb

Date: 2026-07-22
Status: accepted
Author: P0w3r223 + Claude
Related to: ADR 0005

---

## Context

The received wisdom for PSI is a fixed ladder: below 0.1 no drift, 0.1–0.2 moderate, above
0.2 act. The first monitoring run made the problem with that obvious — the detector fired on
the **control snapshot**, a week drawn from the reserve with nothing changed at all.

Measured on the real dataset, reference 70 715 rows against a 2 000-row snapshot, PSI scored
by each column against its own source with nothing drifted:

| Feature | Categories | PSI observed on an unshifted week | PSI from noise alone (99th pct) |
|---|---:|---:|---:|
| age | numeric | 0.002 | 0.009 |
| mileage | numeric | 0.002 | 0.008 |
| fuel | 6 | 0.000 | 0.006 |
| mark | ~30 | 0.013 | 0.017 |
| **model** | **~200** | **0.140** | **0.169** |

For `age`, the textbook 0.2 sits twenty-two times above the noise. For `model`, sampling noise
alone eats 0.169 of it, leaving a margin of 0.031 — and in a smaller sample the noise passes
0.2 outright: 200 categories drawn at 500 rows score **PSI 0.36 against their own source**.

PSI is an effect size, but its *null distribution* still depends on the sample size and on the
number of bins or categories. One fixed number cannot serve a 6-category column and a
200-category column at once.

## Options

1. **Keep fixed thresholds.** Simple, familiar, defensible in a slide. Alarms on the control
   case for high-cardinality columns, which is how monitors get muted.
2. **Hand-tune a threshold per column.** Works until the data changes shape, and encodes no
   reasoning that anyone can check later.
3. **Drop the noisy columns from monitoring.** Cheapest, and it blinds the monitor to the
   feature the model leans on most.
4. **Calibrate against the null.** Resample the reference at the snapshot's own size, score it
   against itself, and take a high quantile of that distribution as the floor a signal must
   clear.

## Decision

Option 4, combined with option 1 rather than replacing it. A column is flagged when its score
exceeds **both**:

- the configured threshold — "is this shift large enough to care about?", and
- its measured noise floor at this sample size — "is this more than the column does on its own?"

`drift.calibration_resamples` (40) and `drift.calibration_quantile` (0.99) control the
measurement; setting resamples to 0 restores the fixed-threshold behaviour, which is what
session 4 will measure this against.

## Consequences

- **The control case is quiet**, which is the property that makes an alarm worth reading.
- **The floor is data, not opinion.** It is recomputed per column and per snapshot size, so it
  keeps working when the traffic volume changes — a monitor tuned on 2 000-row weeks would
  otherwise become trigger-happy the moment weeks got smaller.
- **Cost is a few seconds per report** (40 resamples × 7 columns against a 70 715-row
  reference). Acceptable for a weekly job; the knob exists if it ever is not.
- **Small samples now say "not judged" rather than guessing.** Below
  `drift.min_snapshot_rows` no verdict is issued at all.
- **The fixed threshold still does real work.** Calibration alone would flag shifts that are
  statistically clear and operationally irrelevant — the same mistake as gating on a p-value,
  arrived at from the other side.
- **This came out of a failing test, not a review.** The synthetic fixture was the first place
  the fixed threshold visibly broke, and the honest fix was to change the detector rather than
  the assertion.
