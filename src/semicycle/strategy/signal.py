"""From the point-in-time cycle factor to a target weight on semiconductors.

The signal at month ``T`` uses only information available at ``T``:

  s_T = w_lvl · z(f_T) + w_mom · z(Δ₆ f_T)

where ``f`` is the pseudo-real-time DFM cycle factor (already in the panel) and
``z(·)`` is an expanding-window standardisation (no look-ahead). The signal maps
to a long-only weight on the semiconductor ETF:

  weight_T = clip( base + gain · s_T ,  min_weight ,  max_weight )

Everything else — cross-sectional tilts, leverage rules — is deliberately left
out: fewer knobs, a deflated Sharpe that means something.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.transforms import zscore_expanding


def cycle_signal(panel: pd.DataFrame, *, zwindow: int, level_weight: float,
                 momentum_weight: float) -> pd.Series:
    f = panel["cycle_factor"].astype(float)
    dchg = panel.get("cycle_factor__chg6")
    if dchg is None:
        dchg = f.diff(6)
    s = (
        level_weight * zscore_expanding(f, zwindow)
        + momentum_weight * zscore_expanding(dchg.astype(float), zwindow)
    )
    return s.rename("signal")


def timing_weight(signal: pd.Series, *, base_weight: float, gain: float,
                  min_weight: float, max_weight: float) -> pd.Series:
    w = base_weight + gain * signal
    return w.clip(min_weight, max_weight).rename("weight")


def _weights_from_signal(sig: pd.Series, s) -> pd.DataFrame:
    w = timing_weight(
        sig, base_weight=s.base_weight, gain=s.gain,
        min_weight=s.min_weight, max_weight=s.max_weight,
    )
    out = pd.DataFrame({"signal": sig, "weight": w})
    out = out.loc[out.index >= pd.Timestamp(s.start)]
    if getattr(s, "end", None):
        out = out.loc[out.index <= pd.Timestamp(s.end)]
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def build_weights(panel: pd.DataFrame, s) -> pd.DataFrame:
    """Target weight per month from the coincident cycle factor. `s` is `StrategyCfg`."""
    sig = cycle_signal(
        panel,
        zwindow=s.signal_zwindow,
        level_weight=s.level_weight,
        momentum_weight=s.momentum_weight,
    )
    return _weights_from_signal(sig, s)


def nowcast_signal(oos: pd.DataFrame, *, zwindow: int, model: str = "pred_lightgbm") -> pd.Series:
    """Signal from the walk-forward nowcast: the real-time forecast of billings
    growth `horizon` months ahead, standardised against its own past. Each row of
    `oos` is a genuine as-of-that-month out-of-sample prediction."""
    pred = oos[model].astype(float)
    return zscore_expanding(pred, zwindow).rename("signal")


def build_weights_nowcast(oos: pd.DataFrame, s) -> pd.DataFrame:
    """Target weight from the forward nowcast rather than the coincident factor."""
    return _weights_from_signal(nowcast_signal(oos, zwindow=s.signal_zwindow), s)
