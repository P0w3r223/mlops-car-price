# ADR 0005 — Drift metrics are implemented here; Evidently referees them offline

Date: 2026-07-22
Status: accepted
Author: P0w3r223
Related to: ADR 0006, `tests/test_drift_oracle.py`

---

## Context

The drift monitor needs distribution distances (PSI, Wasserstein, KS, chi-square) and a way
to render a verdict. Evidently is the standard library for exactly this, and the original
project plan named it as the drift component.

Two things about the situation matter. First, the decision logic — thresholds, calibration,
which signal gates and which only reports — is where this project's value sits, and session 4
evaluates that logic tens of thousands of times, so it must be cheap to call and free of
rendering. Second, the numerics are not exotic: SciPy already ships KS, Wasserstein and
chi-square, and PSI is four lines of arithmetic.

Evidently was installed and tried rather than judged from its README:

- it pulls **41 transitive packages**, including a web framework (litestar), a plotting stack
  (plotly), an NLP toolkit (nltk) and a fake-data generator (faker);
- `import evidently` costs **6.1 s**;
- its default drift method for a numeric column at this sample size is
  *"Wasserstein distance (normed), threshold=0.1"* — the same metric and the same threshold
  this project had arrived at independently;
- its numbers and ours agree exactly: 0.0374 / 0.2361 / 0.7917 against 0.0374 / 0.2361 /
  0.7917 for three shift sizes, to five decimal places.

## Options

1. **Evidently as a runtime dependency.** The market-standard answer, and the reports are
   good-looking HTML. Costs 41 packages and a six-second import inside a monitoring step
   that otherwise takes milliseconds, and leaves the decision logic — the interesting part —
   inside someone else's abstraction.
2. **Own metrics, no Evidently at all.** Lean and fully controlled, but "my implementation is
   correct because I believe it is" is not an answer worth giving in an interview.
3. **Own metrics for the gate, Evidently as an offline oracle.** The library referees the
   implementation when the metrics change, and ships with nothing.

## Decision

Option 3. PSI and the per-column comparison live in `drift/metrics.py`; SciPy supplies the
KS, Wasserstein and chi-square primitives; the report renders as markdown. Evidently is an
optional `[oracle]` extra with a test (`tests/test_drift_oracle.py`) that skips when it is
absent, so CI never installs it.

## Consequences

- **The agreement is evidence, not faith.** The oracle test asserts our normalised Wasserstein
  matches Evidently's to 1e-4 across three shift sizes.
- **Independent arrival at the same default** (normed Wasserstein, threshold 0.1) is a useful
  sanity check on the design — and ADR 0006 then shows why even that default is not enough.
- **Markdown beats HTML here.** The report is a CI artifact and a file in a pull request,
  where two weeks diff against each other. An interactive dashboard is worse at that.
- **The evaluation harness in session 4 stays cheap**, because judging a snapshot is arithmetic
  over arrays with no rendering attached.
- **The cost of being wrong is bounded**: if the own implementation ever disagrees with the
  oracle, the extra is one `pip install` away and the test names the column that broke.
