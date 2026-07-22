# Detector evaluation

Every figure is a detection rate over 200 independent snapshots, judged by three detectors on identical data.

## False alarms

Unshifted weeks at growing sample sizes. Nothing changed, so every figure here is an error. This is where a fixed threshold fails: PSI's null distribution grows as the sample shrinks, and the high-cardinality `model` column crosses 0.2 on noise alone.

| Rows in snapshot | fixed thresholds | calibrated (this project) | KS p-value |
|---|---:|---:|---:|
| 250 | 100.0% | 5.0% | 13.5% |
| 500 | 100.0% | 7.5% | 17.5% |
| 1,000 | 99.0% | 0.0% | 20.0% |
| 2,000 | 0.0% | 0.0% | 13.5% |
| 5,000 | 0.0% | 0.0% | 42.5% |

## Power

Detection rate against the size of the shift, on snapshots of 2,000 rows.

### `mileage_shift`

| Magnitude | fixed thresholds | calibrated (this project) | KS p-value |
|---|---:|---:|---:|
| 0.05 | 4.0% | 4.0% | 100.0% |
| 0.1 | 100.0% | 100.0% | 100.0% |
| 0.2 | 100.0% | 100.0% | 100.0% |
| 0.35 | 100.0% | 100.0% | 100.0% |
| 0.5 | 100.0% | 100.0% | 100.0% |

### `unseen_makes`

| Magnitude | fixed thresholds | calibrated (this project) | KS p-value |
|---|---:|---:|---:|
| 0.01 | 0.0% | 0.0% | 100.0% |
| 0.02 | 3.0% | 3.0% | 100.0% |
| 0.05 | 100.0% | 100.0% | 100.0% |
| 0.1 | 100.0% | 100.0% | 100.0% |
| 0.2 | 100.0% | 100.0% | 100.0% |

### `fuel_mix_shift`

| Magnitude | fixed thresholds | calibrated (this project) | KS p-value |
|---|---:|---:|---:|
| 0.02 | 100.0% | 100.0% | 100.0% |
| 0.05 | 100.0% | 100.0% | 100.0% |
| 0.1 | 100.0% | 100.0% | 100.0% |
| 0.2 | 100.0% | 100.0% | 100.0% |
| 0.4 | 100.0% | 100.0% | 100.0% |

## Sensitivity to sample size

The same negligible shift (`mileage_shift`, magnitude 0.05 standard deviations) at growing sample sizes. The shift never changes - only `n` does, and only one detector notices.

| Rows in snapshot | fixed thresholds | calibrated (this project) | KS p-value |
|---|---:|---:|---:|
| 250 | 100.0% | 11.0% | 16.5% |
| 500 | 100.0% | 17.0% | 100.0% |
| 1,000 | 99.5% | 13.0% | 100.0% |
| 2,000 | 1.5% | 1.5% | 100.0% |
| 5,000 | 0.0% | 0.0% | 100.0% |
