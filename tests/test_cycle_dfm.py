"""Dynamic factor model: recovers a known common factor, transforms, sign fix."""

import numpy as np
import pandas as pd
import pytest

from semicycle.config import load_config
from semicycle.cycle.dfm import _apply_transform, fit_cycle_factor


def test_apply_transform():
    idx = pd.date_range("2010-01-31", periods=40, freq="ME")
    s = pd.Series(np.arange(40.0) + 100, index=idx)
    assert _apply_transform(s, "yoy").iloc[:12].isna().all()
    assert _apply_transform(s, "yoy").iloc[12] == pytest.approx(112 / 100 - 1)
    assert _apply_transform(s, "mom3").iloc[3] == pytest.approx(103 / 100 - 1)
    assert _apply_transform(s, "level").equals(s)


def test_dfm_recovers_common_factor():
    rng = np.random.default_rng(0)
    n = 240
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")

    # latent AR(1) cycle
    f = np.zeros(n)
    for t in range(1, n):
        f[t] = 0.85 * f[t - 1] + rng.normal(0, 1)
    f = (f - f.mean()) / f.std()

    loadings = [1.0, 0.8, 1.2, 0.6, 0.9]
    cols = {
        f"ind_{i}": f * lam + rng.normal(0, 0.5, n) for i, lam in enumerate(loadings)
    }
    inputs = pd.DataFrame(cols, index=idx)

    cfg = load_config()
    cfg.params.cycle.sign_reference = "ind_0"
    cfg.params.cycle.factor_orders = 1
    cfg.params.cycle.em_maxiter = 300

    cf = fit_cycle_factor(cfg, inputs=inputs)

    aligned = pd.Series(f, index=idx).reindex(cf.factor.index)
    corr = cf.factor.corr(aligned)
    assert corr > 0.9  # sign already fixed to be positive vs ind_0
    assert cf.correlations["ind_0"] > 0
    assert len(cf.factor) == n
