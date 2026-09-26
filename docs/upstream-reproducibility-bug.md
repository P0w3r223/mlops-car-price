# A reproducibility bug in the upstream model

Two runs of the same model, same seed, same rows, kept disagreeing:

```
RandomForest, random_state=42: 8841.2 PLN, then 8914.1 PLN
LightGBM,     random_state=42: 9331.0 / 9266.7 / 9278.2 PLN
```

The estimators were seeded; the **preprocessing was not**. car-price-ml's target encoder shuffled its
internal cross-fitting folds from an unseeded RNG, so identical data produced different
encodings. A ~70 PLN spread is nothing next to a 9 000 PLN MAE — and everything next to a
100 PLN promotion margin. The gate would have been reading noise part of the time.

Fixed upstream in [car-price-ml#3](https://github.com/P0w3r223/car-price-ml/pull/3)
(v0.1.1), not worked around here ([ADR 0004](decisions/0004-reproducibility-fixed-upstream.md)).
Runs now reproduce to the decimal. The dataset hash changed with the pin, which is exactly
what a data version should do when the code that cleans the data changes.
