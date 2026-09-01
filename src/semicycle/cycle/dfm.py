"""Latent semiconductor cycle via a dynamic factor model.

The previous project combined three indicators with fixed hand-picked weights
(0.40 / 0.30 / 0.30). Here the common component is *estimated*: a single latent
factor with AR dynamics drives a set of coincident indicators, each with its own
loading and idiosyncratic noise. Fitted by EM (`statsmodels DynamicFactorMQ`),
which also handles the ragged edge — the indicators end in different months
because they are published with different lags.

    x_it = lambda_i * f_t + e_it          (i = indicator, t = month)
    f_t  = a_1 f_{t-1} + a_2 f_{t-2} + u_t

`f_t` is the semiconductor cycle index. Its sign is fixed so that a chosen
reference indicator (worldwide billings growth) loads positive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config
from ..features.transforms import mma, mom, yoy
from ..io.store import Store

_SOURCE_TABLE = {
    "wsts_": "wsts",
    "twrev_": "taiwan_revenue",
    "fred_": "fred",
    "px_": "prices",
}


def _apply_transform(s: pd.Series, spec: str) -> pd.Series:
    if spec == "yoy":
        return yoy(s)
    if spec == "yoy_mma3":          # 3-month average, then YoY — smoother cycle input
        return yoy(mma(s, 3))
    if spec == "level":
        return s
    if spec == "diff":
        return s.diff()
    if spec.startswith("mom"):
        return mom(s, int(spec[3:] or 1))
    if spec.startswith("mma"):
        return mma(s, int(spec[3:] or 3))
    raise ValueError(f"unknown transform: {spec}")


def build_cycle_inputs(cfg: Config) -> pd.DataFrame:
    """Wide monthly frame of the stationary-transformed DFM inputs, on the
    reference-month index (final values — this is the cycle *chronology*, not a
    real-time nowcast feature)."""
    store = Store(cfg.duckdb_path)
    tables = {t: store.read(t) for t in store.tables()}
    for df in tables.values():
        df["date"] = pd.to_datetime(df["date"])

    cols = {}
    for series, spec in cfg.params.cycle.inputs.items():
        table = next((v for k, v in _SOURCE_TABLE.items() if series.startswith(k)), None)
        if table is None or table not in tables:
            continue
        raw = tables[table]
        s = raw.loc[raw["series"] == series].set_index("date")["value"].sort_index()
        s = s.resample("ME").last()
        cols[series] = _apply_transform(s, spec)

    out = pd.DataFrame(cols)
    if cfg.params.cycle.start:
        out = out.loc[out.index >= pd.Timestamp(cfg.params.cycle.start)]
    return out.replace([np.inf, -np.inf], np.nan)


def _fit_dfm(x: pd.DataFrame, cfg: Config):
    from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

    model = DynamicFactorMQ(
        x.dropna(how="all"),
        factors=1,
        factor_orders=cfg.params.cycle.factor_orders,
        idiosyncratic_ar1=True,
        standardize=True,
    )
    return model.fit(maxiter=cfg.params.cycle.em_maxiter, disp=0)


def cycle_factor_pit(cfg: Config) -> pd.Series:
    """Pseudo-real-time cycle factor for use as a nowcast feature.

    The DFM inputs are built from `as_of_panel` (each series carried at its last
    *published* value), and we take the Kalman-**filtered** factor — the estimate
    of ``f_t`` using data through ``t`` only. Model parameters are still estimated
    on the full sample (standard "pseudo real-time" approximation); the factor
    path itself carries no future information.
    """
    from ..features.build import _load_tidy, _monthly_index
    from ..features.transforms import as_of_panel

    store = Store(cfg.duckdb_path)
    tidy = _load_tidy(store)
    pit = as_of_panel(tidy, _monthly_index(tidy))

    x = pd.DataFrame(
        {
            s: _apply_transform(pit[s].astype(float), spec)
            for s, spec in cfg.params.cycle.inputs.items()
            if s in pit
        }
    ).replace([np.inf, -np.inf], np.nan)
    if cfg.params.cycle.start:
        x = x.loc[x.index >= pd.Timestamp(cfg.params.cycle.start)]

    res = _fit_dfm(x, cfg)
    factor = res.factors.filtered.iloc[:, 0]

    ref = cfg.params.cycle.sign_reference
    if ref in x and x[ref].corr(factor) < 0:
        factor = -factor
    factor = (factor - factor.mean()) / factor.std()
    return factor.rename("cycle_factor")


def cycle_factor_recursive(cfg: Config, *, step: int = 1, min_months: int = 72) -> pd.Series:
    """Fully recursive real-time factor: at each month re-estimate the DFM on data
    up to that month and take the last smoothed value. Slower than
    :func:`cycle_factor_pit` (which fixes the parameters) — used to check that the
    pseudo-real-time approximation does not distort the factor path.
    """
    from ..features.build import _load_tidy, _monthly_index
    from ..features.transforms import as_of_panel

    store = Store(cfg.duckdb_path)
    tidy = _load_tidy(store)
    pit = as_of_panel(tidy, _monthly_index(tidy))
    x_full = pd.DataFrame(
        {
            s: _apply_transform(pit[s].astype(float), spec)
            for s, spec in cfg.params.cycle.inputs.items()
            if s in pit
        }
    ).replace([np.inf, -np.inf], np.nan)
    if cfg.params.cycle.start:
        x_full = x_full.loc[x_full.index >= pd.Timestamp(cfg.params.cycle.start)]
    x_full = x_full.dropna(how="all")

    ref = cfg.params.cycle.sign_reference
    vals: dict[pd.Timestamp, float] = {}
    months = x_full.index[min_months::step]
    for t in months:
        x = x_full.loc[:t]
        try:
            res = _fit_dfm(x, cfg)
            f = res.factors.smoothed.iloc[:, 0]
            if ref in x and x[ref].corr(f) < 0:
                f = -f
            vals[t] = float(f.iloc[-1])
        except Exception:  # noqa: BLE001, PERF203 - skip a fold that fails to converge
            continue

    out = pd.Series(vals).sort_index()
    return ((out - out.mean()) / out.std()).rename("cycle_factor_recursive")


@dataclass
class CycleFactor:
    factor: pd.Series           # the semiconductor cycle index (standardised)
    loadings: pd.Series         # one per input indicator
    correlations: pd.Series     # corr(indicator, factor) over the common sample
    inputs: pd.DataFrame
    results: object             # the fitted statsmodels results

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"loading": self.loadings, "corr_with_factor": self.correlations}
        ).sort_values("corr_with_factor", ascending=False)


def fit_cycle_factor(cfg: Config, inputs: pd.DataFrame | None = None) -> CycleFactor:
    """Full-sample cycle factor for the historical chronology: Kalman-**smoothed**
    factor from the DFM on reference-month-dated final data. (For the real-time
    nowcast feature use :func:`cycle_factor_pit` instead.)"""
    ccfg = cfg.params.cycle
    x = (build_cycle_inputs(cfg) if inputs is None else inputs).dropna(how="all")

    res = _fit_dfm(x, cfg)
    factor = res.factors.smoothed.iloc[:, 0].rename("cycle_factor")

    loadings = pd.Series(
        np.asarray(res.params[[p for p in res.param_names if "loading" in p]]),
        index=[c for c in x.columns],
        name="loading",
    ) if any("loading" in p for p in res.param_names) else pd.Series(dtype=float)

    corr = x.apply(lambda col: col.corr(factor))

    # sign: make the reference indicator load positive
    ref = ccfg.sign_reference
    if ref in corr and corr[ref] < 0:
        factor = -factor
        loadings = -loadings
        corr = -corr

    factor = (factor - factor.mean()) / factor.std()
    return CycleFactor(
        factor=factor, loadings=loadings, correlations=corr, inputs=x, results=res
    )
