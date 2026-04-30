"""Load `config.yaml` into typed dataclasses.

All strategy parameters live in config.yaml; this module is the single point
where YAML becomes Python. No magic numbers anywhere else in the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass(frozen=True)
class DataConfig:
    ticker_gold: str
    ticker_silver: str
    start_date: str
    end_date: Optional[str]
    raw_cache_path: str
    processed_cache_path: str
    max_forward_fill_days: int


@dataclass(frozen=True)
class CointegrationConfig:
    rolling_window_days: int
    rolling_step_days: int
    significance_level: float
    rolling_cache_path: str


@dataclass(frozen=True)
class SpreadConfig:
    beta_window_days: int


@dataclass(frozen=True)
class OUConfig:
    min_half_life_days: int
    max_half_life_days: int


@dataclass(frozen=True)
class SignalsConfig:
    entry_z: float
    exit_z: float
    stop_z: float
    zscore_window_multiplier: int
    time_stop_multiplier: int


@dataclass(frozen=True)
class BacktestConfig:
    starting_capital: float
    train_years: int
    test_years: int
    step_years: int
    cost_bps_per_leg: float
    borrow_bps_annualized: float
    half_spread_bps: float


@dataclass(frozen=True)
class FiltersConfig:
    vol_lookback_days: int
    vol_percentile_cutoff: float
    coint_pvalue_cutoff: float


@dataclass(frozen=True)
class Config:
    data: DataConfig
    cointegration: CointegrationConfig
    spread: SpreadConfig
    ou: OUConfig
    signals: SignalsConfig
    backtest: BacktestConfig
    filters: FiltersConfig
    random_seed: int
    logging_level: str


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in raw:
        raise KeyError(f"config.yaml missing required section: '{key}'")
    return raw[key]


def load_config(path: str | Path = "config.yaml") -> Config:
    """Parse `config.yaml` into a frozen `Config` dataclass.

    Parameters
    ----------
    path : str or Path
        Path to the YAML config file. Resolved relative to CWD.

    Returns
    -------
    Config
        Frozen, fully typed configuration object.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    KeyError
        If a required section or field is missing from the YAML.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p.resolve()}")

    with p.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return Config(
        data=DataConfig(**_section(raw, "data")),
        cointegration=CointegrationConfig(**_section(raw, "cointegration")),
        spread=SpreadConfig(**_section(raw, "spread")),
        ou=OUConfig(**_section(raw, "ou")),
        signals=SignalsConfig(**_section(raw, "signals")),
        backtest=BacktestConfig(**_section(raw, "backtest")),
        filters=FiltersConfig(**_section(raw, "filters")),
        random_seed=int(raw.get("random_seed", 42)),
        logging_level=str(raw.get("logging", {}).get("level", "INFO")),
    )
