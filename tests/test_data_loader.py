"""Smoke tests for `src.data_loader`."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Config, DataConfig, load_config
from src.data_loader import (
    DataQualityReport,
    _validate_and_clean,
    load_pair_data,
)


EXPECTED_COLS = ["gold", "silver", "log_gold", "log_silver"]


def _synth_prices(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    gold = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    silver = 20 * np.exp(np.cumsum(rng.normal(0, 0.015, len(dates))))
    return pd.DataFrame({"gold": gold, "silver": silver}, index=dates)


def _override_cache_paths(cfg: Config, tmp_path: Path) -> Config:
    new_data = replace(
        cfg.data,
        raw_cache_path=str(tmp_path / "raw.parquet"),
        processed_cache_path=str(tmp_path / "processed.parquet"),
    )
    return replace(cfg, data=new_data)


@pytest.mark.network
def test_load_pair_data_shape(tmp_path: Path) -> None:
    """End-to-end smoke test: real yfinance pull returns a usable frame."""
    cfg = _override_cache_paths(load_config("config.yaml"), tmp_path)
    df = load_pair_data(cfg, refresh=True)
    assert list(df.columns) == EXPECTED_COLS
    assert len(df) > 1000
    assert df.index.is_monotonic_increasing


def test_no_nan_in_processed() -> None:
    dates = pd.bdate_range("2020-01-01", "2020-06-30")
    raw = _synth_prices(dates)
    df, _ = _validate_and_clean(raw, max_forward_fill_days=1)
    assert not df.isna().any().any()
    assert list(df.columns) == EXPECTED_COLS


def test_log_prices_consistent() -> None:
    dates = pd.bdate_range("2020-01-01", "2020-06-30")
    raw = _synth_prices(dates)
    df, _ = _validate_and_clean(raw, max_forward_fill_days=1)
    np.testing.assert_allclose(df["log_gold"].to_numpy(), np.log(df["gold"].to_numpy()))
    np.testing.assert_allclose(df["log_silver"].to_numpy(), np.log(df["silver"].to_numpy()))


def test_date_index_monotonic() -> None:
    dates = pd.bdate_range("2020-01-01", "2020-06-30")
    raw = _synth_prices(dates)
    df, _ = _validate_and_clean(raw, max_forward_fill_days=1)
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique


def test_cache_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Second load with `refresh=False` returns the same frame as the first."""
    cfg = _override_cache_paths(load_config("config.yaml"), tmp_path)
    dates = pd.bdate_range("2020-01-01", "2020-12-31")
    fake = _synth_prices(dates)

    import src.data_loader as dl

    def fake_download(*args, **kwargs):
        return fake.copy()

    monkeypatch.setattr(dl, "_download_from_yfinance", fake_download)

    first = load_pair_data(cfg, refresh=True)
    second = load_pair_data(cfg, refresh=False)
    pd.testing.assert_frame_equal(first, second)


def test_forward_fill_cap_respected() -> None:
    """1-day intra-index NaNs are filled; >1-day NaN runs are dropped, cap=1."""
    dates = pd.bdate_range("2020-01-01", "2020-01-31")
    raw = _synth_prices(dates)

    one_day_gap = dates[5]
    raw.loc[one_day_gap, ["gold", "silver"]] = np.nan

    long_gap_dates = dates[10:14]
    raw.loc[long_gap_dates, ["gold", "silver"]] = np.nan

    df, report = _validate_and_clean(raw, max_forward_fill_days=1)

    assert one_day_gap in df.index
    for d in long_gap_dates:
        assert d not in df.index
    assert report.n_forward_filled == 1
    intra = [g for g in report.gap_log if g[2] == 4]
    assert len(intra) == 1


def test_market_holidays_not_synthesized() -> None:
    """Holidays absent from yfinance index must NOT appear in processed output.

    Regression test for the bug where reindexing onto pd.bdate_range
    invented bars on NYSE holidays (e.g. MLK day, July 4th).
    """
    dates = pd.DatetimeIndex(
        ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"],
        name="date",
    )
    raw = _synth_prices(dates)
    df, report = _validate_and_clean(raw, max_forward_fill_days=1)
    assert pd.Timestamp("2020-01-04") not in df.index
    assert pd.Timestamp("2020-01-05") not in df.index
    assert len(df) == 4
    assert report.n_forward_filled == 0


def test_unusual_calendar_gap_logged_not_filled() -> None:
    """Multi-day market closures (Sandy, COVID halt) are logged but not imputed.

    Sandy: NYSE closed Mon 2012-10-29 + Tue 2012-10-30, so Fri -> Wed gap
    has 2 missing business days, which crosses our unusual-gap threshold.
    """
    dates = pd.DatetimeIndex(
        ["2012-10-26", "2012-10-31", "2012-11-01", "2012-11-02"],
        name="date",
    )
    raw = _synth_prices(dates)
    df, report = _validate_and_clean(raw, max_forward_fill_days=1)
    assert pd.Timestamp("2012-10-29") not in df.index
    assert pd.Timestamp("2012-10-30") not in df.index
    assert len(report.gap_log) == 1
    gap_start, gap_end, missing = report.gap_log[0]
    assert gap_start == pd.Timestamp("2012-10-26")
    assert gap_end == pd.Timestamp("2012-10-31")
    assert missing == 2


def test_gap_log_records_dropped_gaps() -> None:
    """Dropped intra-index NaN runs appear in gap_log; not double-logged as unusual gaps."""
    dates = pd.bdate_range("2020-01-01", "2020-02-29")
    raw = _synth_prices(dates)
    long_gap = dates[15:20]
    raw.loc[long_gap, ["gold", "silver"]] = np.nan

    _, report = _validate_and_clean(raw, max_forward_fill_days=1)

    intra = [g for g in report.gap_log if g[2] == len(long_gap)]
    assert len(intra) == 1
    gap_start, gap_end, gap_len = intra[0]
    assert gap_start == long_gap[0]
    assert gap_end == long_gap[-1]
    assert gap_len == len(long_gap)
    assert len(report.gap_log) == 1
