# Why the textbook PSI threshold does not work

The first monitoring run flagged drift on the **control snapshot** — a week with nothing
changed. The culprit is the received wisdom that PSI above 0.2 means drift:

| Feature | Categories | PSI on an unshifted week | PSI from noise alone (99th pct) |
|---|---:|---:|---:|
| age | numeric | 0.002 | 0.009 |
| fuel | 6 | 0.000 | 0.006 |
| mark | ~30 | 0.013 | 0.017 |
| **model** | **~200** | **0.140** | **0.169** |

PSI is an effect size, but its null distribution still depends on sample size and category
count. For `age` the 0.2 threshold sits 22× above the noise; for `model` the noise alone eats
0.169 of it. Shrink the sample and it breaks outright: 200 categories drawn at 500 rows score
**PSI 0.36 against their own source**.

So a column is flagged only when it clears **both** the configured threshold ("is this worth
acting on?") and its own measured noise floor ("is this more than the column does by itself?"),
where the floor comes from resampling the reference at the snapshot's size
([ADR 0006](decisions/0006-calibrated-drift-thresholds.md)). p-values are reported for
every column and gate nothing — at these sample sizes they measure n, not drift.
