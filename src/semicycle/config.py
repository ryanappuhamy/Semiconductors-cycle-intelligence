"""Typed access to `config/*.yaml` plus project paths.

Everything downstream imports `load_config()` rather than reading YAML or
hard-coding paths, so a single edit to the config files reconfigures the whole
pipeline.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


def project_root() -> Path:
    """Repo root, resolved from this file (src/semicycle/config.py -> ../../..)."""
    env = os.environ.get("SEMICYCLE_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


# --- schema -----------------------------------------------------------------


class FredSeries(BaseModel):
    label: str
    release_lag_days: int


class WstsCfg(BaseModel):
    url: str
    release_lag_days: int
    regions: list[str]
    actuals_through: str | None = None
    mom_zscore_alarm: float = 4.0


class TaiwanCfg(BaseModel):
    api_url: str
    dataset: str
    start_date: str
    release_lag_days: int
    companies: dict[str, str]
    core_companies: list[str] = Field(default_factory=list)


class FredCfg(BaseModel):
    csv_url: str
    series: dict[str, FredSeries]


class PricesCfg(BaseModel):
    start_date: str
    release_lag_days: int
    benchmarks: dict[str, str]
    universe: list[str]


class Sources(BaseModel):
    wsts: WstsCfg
    taiwan_revenue: TaiwanCfg
    fred: FredCfg
    prices: PricesCfg


class TargetCfg(BaseModel):
    series: str
    smoothing_months: int
    transform: str
    horizons: list[int]


class FeaturesCfg(BaseModel):
    yoy: bool
    mom_3mma: bool
    zscore_expanding_min_periods: int
    momentum_windows_months: list[int]


class DatingCfg(BaseModel):
    window: int = 5
    min_phase: int = 5
    min_cycle: int = 18
    edge: int = 6
    level_smooth: int = 3
    min_run: int = 3


class CycleCfg(BaseModel):
    inputs: dict[str, str]
    factor_orders: int = 2
    em_maxiter: int = 200
    sign_reference: str
    start: str | None = None
    dating: DatingCfg = Field(default_factory=DatingCfg)


class CvCfg(BaseModel):
    scheme: str
    min_train_months: int
    step_months: int
    purge_months: int
    embargo_months: int
    oos_start: str | None = None


class ModelsCfg(BaseModel):
    ar_benchmark: dict = Field(default_factory=dict)
    elasticnet: dict = Field(default_factory=dict)
    lightgbm: dict = Field(default_factory=dict)


class StrategyCfg(BaseModel):
    rebalance: str
    cost_bps: float
    vol_target_annual: float


class Params(BaseModel):
    paths: dict[str, str]
    target: TargetCfg
    features: FeaturesCfg
    cycle: CycleCfg
    cv: CvCfg
    models: ModelsCfg
    strategy: StrategyCfg


class Config(BaseModel):
    sources: Sources
    params: Params
    root: Path

    # resolved absolute paths -------------------------------------------------
    @property
    def duckdb_path(self) -> Path:
        return self.root / self.params.paths["duckdb"]

    @property
    def panel_path(self) -> Path:
        return self.root / self.params.paths["panel"]

    @property
    def reports_dir(self) -> Path:
        return self.root / self.params.paths["reports_dir"]

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"


# --- loader ----------------------------------------------------------------


@lru_cache(maxsize=1)
def load_config() -> Config:
    root = project_root()
    with (root / "config" / "sources.yaml").open(encoding="utf-8") as fh:
        sources = yaml.safe_load(fh)
    with (root / "config" / "params.yaml").open(encoding="utf-8") as fh:
        params = yaml.safe_load(fh)
    return Config(sources=Sources(**sources), params=Params(**params), root=root)
