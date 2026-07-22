"""Cross-check the drift metrics against Evidently, which is not a runtime dependency.

Evidently is a mature drift library, so it makes a good referee for an implementation that
has to be trusted. It is *not* installed by `[dev]` — pulling in a web framework, a plotting
stack and an NLP toolkit to render a report this project renders as markdown is a bad trade
(ADR 0005). Install it deliberately when the metrics change:

    pip install -e ".[oracle]" && pytest tests/test_drift_oracle.py

Note what the reference implementation agrees about: its default drift method for a numeric
column at this sample size is "Wasserstein distance (normed)" with a threshold of 0.1 —
the same metric and the same threshold this project arrived at independently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlops_car_price.drift import metrics

evidently = pytest.importorskip("evidently", reason="install the 'oracle' extra to run")

from evidently import DataDefinition, Dataset, Report  # noqa: E402
from evidently.presets import DataDriftPreset  # noqa: E402

SHIFTS = (0.0, 0.25, 0.8)


def evidently_normed_wasserstein(reference: pd.Series, current: pd.Series) -> float:
    definition = DataDefinition(numerical_columns=["x"])
    snapshot = Report([DataDriftPreset()]).run(
        Dataset.from_pandas(pd.DataFrame({"x": current}), data_definition=definition),
        Dataset.from_pandas(pd.DataFrame({"x": reference}), data_definition=definition),
    )
    values = [
        metric["value"]
        for metric in snapshot.dict()["metrics"]
        if metric["metric_name"].startswith("ValueDrift")
    ]
    return float(values[0])


@pytest.mark.parametrize("shift", SHIFTS)
def test_normalised_shift_agrees_with_evidently(shift: float):
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(0, 1, 5_000))
    current = pd.Series(rng.normal(shift, 1, 5_000))

    ours = metrics.numeric_drift("x", reference, current).normalised_shift

    assert ours == pytest.approx(evidently_normed_wasserstein(reference, current), abs=1e-4)
