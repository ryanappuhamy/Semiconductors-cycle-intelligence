"""Performance statistics, plus the two that quantify how much a backtest can be
trusted after a search: the **deflated Sharpe ratio** and the **probability of
backtest overfitting** (CSCV).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats as sps

_EULER = 0.5772156649015329
PPY = 12  # monthly data


def perf_stats(returns: pd.Series, *, ppy: int = PPY, rf: float = 0.0) -> pd.Series:
    r = returns.dropna()
    excess = r - rf / ppy
    n = len(r)
    ann_ret = (1 + r).prod() ** (ppy / n) - 1 if n else np.nan
    ann_vol = r.std(ddof=1) * math.sqrt(ppy)
    sharpe = (excess.mean() / r.std(ddof=1)) * math.sqrt(ppy) if r.std(ddof=1) else np.nan
    downside = r[r < 0].std(ddof=1) * math.sqrt(ppy)
    sortino = (excess.mean() * ppy) / downside if downside else np.nan
    curve = (1 + r).cumprod()
    dd = curve / curve.cummax() - 1
    max_dd = dd.min()
    return pd.Series(
        {
            "months": n,
            "ann_return": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "calmar": ann_ret / abs(max_dd) if max_dd else np.nan,
            "hit_rate": (r > 0).mean(),
            "skew": sps.skew(r),
            "kurtosis": sps.kurtosis(r, fisher=False),
        }
    )


def _sharpe_per_period(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return r.mean() / sd if sd else 0.0


def deflated_sharpe_ratio(
    returns: pd.Series, *, n_trials: int, trial_sharpes: np.ndarray | None = None,
    ppy: int = PPY,
) -> pd.Series:
    """Bailey & López de Prado (2014). Probability that the true Sharpe is > 0
    once you account for having selected the best of ``n_trials`` configurations.

    ``trial_sharpes`` (per-period SRs of every config tried) sets the variance of
    the selection process; without it we fall back to the estimator's own s.e.
    """
    r = returns.dropna().to_numpy()
    t = len(r)
    sr = _sharpe_per_period(r)
    g3 = float(sps.skew(r))
    g4 = float(sps.kurtosis(r, fisher=False))

    sr_var = (1 - g3 * sr + (g4 - 1) / 4 * sr**2) / (t - 1)
    sr_se = math.sqrt(max(sr_var, 1e-12))

    if trial_sharpes is not None and len(trial_sharpes) > 1:
        v_trials = float(np.var(trial_sharpes, ddof=1))
    else:
        v_trials = sr_se**2
    sigma = math.sqrt(max(v_trials, 1e-12))

    n = max(n_trials, 2)
    sr0 = sigma * (
        (1 - _EULER) * sps.norm.ppf(1 - 1 / n)
        + _EULER * sps.norm.ppf(1 - 1 / (n * math.e))
    )
    dsr = float(sps.norm.cdf((sr - sr0) / sr_se))
    return pd.Series(
        {
            "sharpe_ann": sr * math.sqrt(ppy),
            "sharpe_per_period": sr,
            "expected_max_sharpe_H0": sr0,
            "deflated_sharpe_ratio": dsr,
            "n_trials": n_trials,
        }
    )


def probability_of_backtest_overfitting(
    returns_matrix: pd.DataFrame, *, n_partitions: int = 10
) -> pd.Series:
    """CSCV (Bailey, Borwein, López de Prado, Zhu 2017).

    Split the sample into ``S`` contiguous blocks; for every way of choosing S/2
    blocks as in-sample, take the config with the best in-sample Sharpe and see
    where it ranks out-of-sample. PBO = share of splits where that config lands
    below the out-of-sample median (logit ≤ 0).
    """
    m = returns_matrix.dropna()
    s = n_partitions - (n_partitions % 2)
    rows = np.array_split(np.arange(len(m)), s)
    n_cfg = m.shape[1]

    logits = []
    for is_blocks in itertools.combinations(range(s), s // 2):
        is_idx = np.concatenate([rows[b] for b in is_blocks])
        oos_idx = np.concatenate([rows[b] for b in range(s) if b not in is_blocks])
        is_sr = m.iloc[is_idx].apply(lambda c: _sharpe_per_period(c.to_numpy()))
        oos_sr = m.iloc[oos_idx].apply(lambda c: _sharpe_per_period(c.to_numpy()))
        best = is_sr.idxmax()
        rank = oos_sr.rank().loc[best]           # 1 = worst, n_cfg = best
        omega = rank / (n_cfg + 1)
        logits.append(math.log(omega / (1 - omega)))

    logits = np.array(logits)
    return pd.Series(
        {
            "pbo": float((logits <= 0).mean()),
            "logit_median": float(np.median(logits)),
            "n_splits": len(logits),
            "n_configs": n_cfg,
        }
    )


def regime_attribution(
    strategy_ret: pd.Series, phases: pd.Series, *, ppy: int = PPY
) -> pd.DataFrame:
    """Strategy performance conditioned on the cycle phase at the start of each month."""
    df = pd.DataFrame({"ret": strategy_ret, "phase": phases.reindex(strategy_ret.index)}).dropna()
    out = df.groupby("phase")["ret"].agg(
        months="count",
        mean_monthly="mean",
        hit_rate=lambda x: (x > 0).mean(),
        vol_ann=lambda x: x.std(ddof=1) * math.sqrt(ppy),
    )
    out["ann_return"] = (1 + df.groupby("phase")["ret"].mean()) ** ppy - 1
    order = ["Early Cycle", "Mid Cycle", "Late Cycle", "Downturn"]
    return out.reindex([p for p in order if p in out.index])
