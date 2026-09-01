"""Module 4: the nowcast-based timing signal and the HTML dashboard."""

import numpy as np
import pandas as pd

from semicycle.config import load_config
from semicycle.report.dashboard import build_dashboard
from semicycle.strategy.signal import build_weights_nowcast, nowcast_signal


def _fake_oos(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2006-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"y_true": rng.normal(0.1, 0.1, n), "pred_lightgbm": rng.normal(0.1, 0.08, n)},
        index=idx,
    )


def test_nowcast_signal_is_causal():
    oos = _fake_oos()
    full = nowcast_signal(oos, zwindow=24)
    trunc = nowcast_signal(oos.iloc[:120], zwindow=24)
    pd.testing.assert_series_equal(full.iloc[:120], trunc, check_names=False)


def test_build_weights_nowcast_in_bounds():
    cfg = load_config()
    w = build_weights_nowcast(_fake_oos(), cfg.params.strategy)
    assert (w["weight"] >= cfg.params.strategy.min_weight - 1e-9).all()
    assert (w["weight"] <= cfg.params.strategy.max_weight + 1e-9).all()
    assert w.index.is_monotonic_increasing


def test_dashboard_builds_self_contained(tmp_path):
    cfg = load_config()
    # point reports_dir at an empty temp dir -> every panel is a graceful placeholder
    object.__setattr__(cfg, "params", cfg.params)
    cfg.params.paths["reports_dir"] = str(tmp_path)
    out = build_dashboard(cfg, brief="hello", chronology=pd.DataFrame())
    html = out.read_text(encoding="utf-8")
    assert out.name == "dashboard.html"
    assert "<title>Semiconductor Cycle Intelligence</title>" in html
    assert "hello" in html
    assert "http://" not in html and "https://" not in html.replace("initial-scale", "")
