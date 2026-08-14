# ADR 0002 — Production traffic is a replay of held-out data, not a scraper

Date: 2026-07-22
Status: accepted
Author: P0w3r223
Related to: ADR 0001, `car-price-ml/docs/research/data-and-methodology.md`

---

## Context

Drift monitoring needs data arriving *after* training. The obvious source is fresh listings
from the Polish car marketplaces — and project A3 already examined that source and rejected
it: the listing databases are protected by the *sui generis* database right, the terms of
service forbid automated collection, and listings carry personal data. That decision is
documented in A3 and is not weakened by a later project finding fresh data convenient.

The dataset that A3 does use (`aleksandrglotov/car-prices-poland`, CC0) is static and, more
awkwardly, **carries no listing date** — its columns are make, model, generation, year,
mileage, engine volume, fuel, city, province and price. Even a chronological replay of the
real data is impossible: there is no time axis to replay along.

## Options

1. **Scrape a marketplace anyway.** Real drift, real timestamps — and a portfolio project
   that demonstrates ignoring a legal analysis its own author wrote three weeks earlier.
2. **Find a second open dataset and treat it as production traffic.** The shift between two
   sources is genuine covariate shift, but its size is whatever it happens to be: not
   controllable, so detector sensitivity cannot be measured, and column mismatches would
   dominate the work.
3. **Replay a held-out slice as weekly snapshots, with named drift scenarios.** The
   `stream_pool` split — never trained on, never evaluated on — is drawn from in weekly
   batches, optionally transformed by a parameterised scenario (price shock, fuel-mix
   shift, unseen makes, higher mileage, a broken schema).

## Decision

Option 3. `stream_pool` is the production stream, and drift is injected by named scenarios
whose magnitude is a parameter.

## Consequences

- **The shift size is known**, so the drift detector can be *evaluated* rather than merely
  run: false-positive rate on unshifted snapshots, detection rate as a function of shift
  magnitude. A detector nobody measured is an alarm nobody can trust.
- **Labels arrive with the traffic.** Because the snapshots are real rows with real prices,
  the system can confirm a drift alert against the model's actual MAE on that week — the
  link a production system normally waits weeks for.
- **One scenario is deliberately unfair to the detectors.** A pure price shock moves the
  target while leaving every feature untouched: feature-drift metrics cannot see it, and
  only the prediction-error signal catches it. That is the point of including it.
- **The honest limitation stands in the README.** This project demonstrates that the system
  reacts correctly to degradation; it does not prove the model degrades on the 2026 Polish
  market. Claiming otherwise would be the same shortcut this ADR refuses.
- **"Weeks" are a construct.** With no listing date, the replay order is random rather than
  chronological, so genuine seasonality or temporal drift is out of reach here.
