"""Pull, validate, and cache GLD/SLV daily price data."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataQualityReport:
    """Summary of the data validation pass."""

    n_rows: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    n_forward_filled: int
    n_dropped_gaps: int
    gap_log: list[tuple[pd.Timestamp, pd.Timestamp, int]]
    pulled_at: pd.Timestamp


def _download_from_yfinance(
    ticker_gold: str,
    ticker_silver: str,
    start: str,
    end: Optional[str],
) -> pd.DataFrame:
    """Pull adjusted close prices for both tickers from yfinance.

    Returns a DataFrame indexed by date with columns ['gold', 'silver'].
    Handles yfinance's multi-index column edge case (group_by='column').
    """
    import yfinance as yf

    logger.info("yfinance download: %s, %s [%s -> %s]", ticker_gold, ticker_silver, start, end or "today")
    raw = yf.download(
        tickers=[ticker_gold, ticker_silver],
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned empty frame; check tickers / network")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
        elif "Close" in raw.columns.get_level_values(1):
            close = raw.xs("Close", axis=1, level=1)
        else:
            raise RuntimeError(f"yfinance frame missing 'Close' level; got {raw.columns}")
    else:
        close = raw[["Close"]]

    close = close[[ticker_gold, ticker_silver]].copy()
    close.columns = ["gold", "silver"]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = "date"
    return close


def _validate_and_clean(
    raw: pd.DataFrame,
    max_forward_fill_days: int,
    unusual_missing_bdays: int = 2,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Validate, gap-handle, and add log-price columns.

    The yfinance index *is* the canonical trading calendar (NYSE-aligned
    after auto_adjust). We do NOT reindex onto a synthetic business-day
    calendar — that would invent observations on market holidays.

    Strategy:
      - Sort. Run NaN-run detection on the actual trading-day index.
        Runs <= cap forward-filled, longer runs dropped.
      - Detect "unusual gaps" — consecutive trading days more than
        `unusual_gap_calendar_days` calendar days apart. These represent
        market closures (Sandy, COVID-era halts) and are logged as
        anomalies but never imputed.
    """
    if raw.empty:
        raise RuntimeError("raw frame is empty")
    df = raw.sort_index().copy()

    nan_mask = df[["gold", "silver"]].isna().any(axis=1)
    n_forward_filled = 0
    drop_positions: list[int] = []
    intra_gap_log: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []

    in_gap = False
    gap_start_idx = 0
    for i, is_nan in enumerate(nan_mask.to_numpy()):
        if is_nan and not in_gap:
            in_gap = True
            gap_start_idx = i
        elif not is_nan and in_gap:
            in_gap = False
            run_len = i - gap_start_idx
            if run_len <= max_forward_fill_days:
                n_forward_filled += run_len
            else:
                run = df.index[gap_start_idx:i]
                intra_gap_log.append((run[0], run[-1], run_len))
                drop_positions.extend(range(gap_start_idx, i))
    if in_gap:
        run_len = len(nan_mask) - gap_start_idx
        run = df.index[gap_start_idx:]
        intra_gap_log.append((run[0], run[-1], run_len))
        drop_positions.extend(range(gap_start_idx, len(nan_mask)))

    df = df.ffill(limit=max_forward_fill_days)
    if drop_positions:
        df = df.drop(df.index[drop_positions])
    df = df.dropna(subset=["gold", "silver"])

    intra_dropped_dates: set[pd.Timestamp] = set()
    for gs, ge, _ in intra_gap_log:
        intra_dropped_dates.update(pd.date_range(gs, ge).tolist())

    unusual_gaps: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    idx = df.index
    for i in range(1, len(idx)):
        prev, curr = idx[i - 1], idx[i]
        missing_bdays = len(pd.bdate_range(prev, curr)) - 2
        if missing_bdays >= unusual_missing_bdays:
            spans_intra = any(
                prev < d < curr for d in intra_dropped_dates
            )
            if not spans_intra:
                unusual_gaps.append((prev, curr, missing_bdays))

    gap_log = intra_gap_log + unusual_gaps

    df["log_gold"] = np.log(df["gold"])
    df["log_silver"] = np.log(df["silver"])
    if df.isna().any().any():
        raise RuntimeError("processed frame still contains NaNs after cleaning")

    for gap_start, gap_end, gap_len in intra_gap_log:
        logger.warning("dropped intra-index NaN run: %s -> %s (%d trading days)", gap_start.date(), gap_end.date(), gap_len)
    for gap_start, gap_end, missing in unusual_gaps:
        logger.warning("unusual market gap: %s -> %s (%d missing business days, no imputation)", gap_start.date(), gap_end.date(), missing)
    if n_forward_filled:
        logger.info("forward-filled %d intra-index NaN(s) within cap=%d", n_forward_filled, max_forward_fill_days)

    report = DataQualityReport(
        n_rows=len(df),
        start_date=df.index.min(),
        end_date=df.index.max(),
        n_forward_filled=n_forward_filled,
        n_dropped_gaps=len(gap_log),
        gap_log=gap_log,
        pulled_at=pd.Timestamp.now("UTC"),
    )
    return df, report


def _report_to_json(report: DataQualityReport) -> bytes:
    payload = {
        "n_rows": report.n_rows,
        "start_date": report.start_date.isoformat(),
        "end_date": report.end_date.isoformat(),
        "n_forward_filled": report.n_forward_filled,
        "n_dropped_gaps": report.n_dropped_gaps,
        "gap_log": [(s.isoformat(), e.isoformat(), n) for s, e, n in report.gap_log],
        "pulled_at": report.pulled_at.isoformat(),
    }
    return json.dumps(payload).encode("utf-8")


def _save_processed(df: pd.DataFrame, report: DataQualityReport, path: Path) -> None:
    """Write processed frame to parquet, embedding `pulled_at` in metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df)
    md = dict(table.schema.metadata or {})
    md[b"data_quality_report"] = _report_to_json(report)
    md[b"pulled_at"] = report.pulled_at.isoformat().encode("utf-8")
    table = table.replace_schema_metadata(md)
    pq.write_table(table, path)


def _save_raw(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _load_processed(path: Path) -> pd.DataFrame:
    """Load processed parquet from disk."""
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    if df.index.freq is None and df.index.is_monotonic_increasing:
        try:
            df.index.freq = pd.infer_freq(df.index)
        except (ValueError, TypeError):
            pass
    return df


def load_pair_data(config: Config, refresh: bool = False) -> pd.DataFrame:
    """Load processed GLD/SLV price data, pulling from yfinance if needed.

    Parameters
    ----------
    config : Config
        Strategy configuration (provides tickers, dates, cache paths, FF cap).
    refresh : bool, default False
        If True, ignore on-disk cache and re-pull from yfinance.

    Returns
    -------
    pd.DataFrame
        Indexed by date (business-day frequency), columns:
        ['gold', 'silver', 'log_gold', 'log_silver'].
    """
    processed_path = Path(config.data.processed_cache_path)
    raw_path = Path(config.data.raw_cache_path)

    if processed_path.is_file() and not refresh:
        logger.info("loading processed cache: %s", processed_path)
        return _load_processed(processed_path)

    raw = _download_from_yfinance(
        ticker_gold=config.data.ticker_gold,
        ticker_silver=config.data.ticker_silver,
        start=config.data.start_date,
        end=config.data.end_date,
    )
    _save_raw(raw, raw_path)

    df, report = _validate_and_clean(raw, config.data.max_forward_fill_days)
    _save_processed(df, report, processed_path)

    logger.info(
        "data ready: %d rows, %s -> %s, ff=%d, dropped_gaps=%d",
        report.n_rows,
        report.start_date.date(),
        report.end_date.date(),
        report.n_forward_filled,
        report.n_dropped_gaps,
    )
    return df
