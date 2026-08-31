"""End-to-end on synthetic data: tidy -> panel -> supervised -> one CV fold.

No network. Exercises the real functions so CI catches interface breaks.
"""

import numpy as np
import pandas as pd

from semicycle.config import load_config
from semicycle.features.build import build_features, build_target
from semicycle.nowcast.dataset import make_supervised
from semicycle.nowcast.evaluate import scoreboard, walk_forward
from semicycle.nowcast.models import ARBenchmark


def _synthetic_tidy(n_months: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2000-01-31", periods=n_months, freq="ME")
    rng = np.random.default_rng(0)
    cycle = np.sin(np.arange(n_months) / 9) * 0.15 + 1.0
    rows = []
    # a WSTS-like worldwide billings series with a cycle + drift
    level = np.cumprod(1 + 0.004 + 0.02 * (cycle - 1) + rng.normal(0, 0.01, n_months)) * 1e4
    for d, v in zip(idx, level, strict=True):
        rows.append({"date": d, "series": "wsts_worldwide", "value": v,
                     "published": d + pd.Timedelta(days=35)})
    # a leading indicator: same cycle shifted forward a bit + noise
    lead = level * (1 + 0.05 * np.roll(cycle - 1, -2)) * (1 + rng.normal(0, 0.02, n_months))
    for d, v in zip(idx, lead, strict=True):
        rows.append({"date": d, "series": "twrev_aggregate", "value": v,
                     "published": d + pd.Timedelta(days=11)})
    return pd.DataFrame(rows)


def test_offline_pipeline_runs_and_benchmark_scores():
    cfg = load_config()
    tidy = _synthetic_tidy()
    index = pd.date_range(tidy["date"].min(), tidy["date"].max(), freq="ME")

    feats = build_features(tidy, index, cfg)
    target = build_target(tidy, cfg).reindex(index)
    panel = feats.join(target)
    panel.index.name = "date"

    assert "target" in panel.columns
    assert panel["target"].notna().sum() > 100
    assert any(c.startswith("twrev_aggregate") for c in panel.columns)

    data = make_supervised(panel, horizon=3)
    assert len(data.y) > 100
    assert data.benchmark_cols  # target_pit + lags present

    results = walk_forward(
        data, {"ar_benchmark": ARBenchmark(data.benchmark_cols)},
        min_train_months=60, step_months=3, purge_months=3, embargo_months=1,
    )
    assert len(results) > 5
    board = scoreboard(results)
    assert "ar_benchmark" in board.index
    assert np.isfinite(board.loc["ar_benchmark", "MAE"])
