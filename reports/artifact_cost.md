| Model | Holdout MAE (PLN) | Train | Artifact | Load | Predict p50 | Predict p95 | Batch | Registry / year |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ridge | 15,422 | 0.1 s | 0.0 MB | 0.00 s | 5.1 ms | 6.1 ms | 727,554 rows/s | 0.0 GB |
| LightGBM | 9,278 | 1.7 s | 3.3 MB | 0.03 s | 12.9 ms | 13.7 ms | 118,993 rows/s | 0.2 GB |
| RandomForest | 8,908 | 7.8 s | 338.5 MB | 0.44 s | 47.1 ms | 68.2 ms | 148,049 rows/s | 17.2 GB |

Trained on 70,715 rows, scored on the frozen holdout of 23,571 rows. Registry column projects 52 retrainings a year, one version each.
