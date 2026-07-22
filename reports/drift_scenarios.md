| Scenario | What breaks | Feature drift | Prediction drift | MAE change | Alert |
|---|---|---|---|---:|---|
| `stable` | no change; the control case | - | no | -0.9% | no |
| `price_shock` | prices inflate, features unchanged | - | no | +135.2% | **yes** |
| `fuel_mix_shift` | more electric and hybrid cars | age, mileage, vol_engine, model, fuel | yes | +2.1% | **yes** |
| `mileage_shift` | higher mileage across the week | mileage | no | +24.5% | **yes** |
| `unseen_makes` | makes absent from the training data | mark | no | +7.5% | **yes** |
| `missing_engine_volume` | engine volume arrives empty | vol_engine | no | +26.1% | **yes** |

Snapshots of 2,000 rows drawn from `stream_pool`, compared against `train_initial`. Scenario magnitudes are the defaults in `replay.SCENARIOS`.
