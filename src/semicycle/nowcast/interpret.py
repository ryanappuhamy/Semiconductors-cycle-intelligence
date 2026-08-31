"""Full-sample feature importance — for interpretation only, not evaluation.

Fits one LightGBM on the whole supervised frame and returns gain-based
importances. This is in-sample and says nothing about out-of-sample skill; it
just shows which inputs the model leans on (e.g. does Taiwan monthly revenue
lead worldwide billings, as the industry lore says?).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .dataset import Supervised


def feature_importance(data: Supervised, params: dict, top: int = 20) -> pd.DataFrame:
    import lightgbm as lgb

    cols = list(data.X.columns)
    model = lgb.LGBMRegressor(**params, verbosity=-1)
    model.fit(np.asarray(data.X, dtype=float), np.asarray(data.y, dtype=float))
    imp = (
        pd.DataFrame({"feature": cols, "gain": model.booster_.feature_importance("gain")})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    imp["gain_share"] = imp["gain"] / imp["gain"].sum()
    imp["family"] = imp["feature"].str.split("_").str[0]
    return imp.head(top)
