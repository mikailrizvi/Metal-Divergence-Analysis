"""Cointegration testing for the gold-silver pair."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass(frozen=True)
class ADFResult:
    """Augmented Dickey-Fuller result. Null hypothesis: unit root present."""

    statistic: float
    pvalue: float
    used_lag: int
    n_obs: int
    crit_1pct: float
    crit_5pct: float
    crit_10pct: float


@dataclass(frozen=True)
class EngleGrangerResult:
    """Engle-Granger two-step cointegration result."""

    beta: float
    alpha: float
    residual_adf: ADFResult
    pvalue: float
    n_obs: int


@dataclass(frozen=True)
class JohansenResult:
    """Johansen trace-test result for a 2-variable VECM."""

    trace_stats: tuple[float, float]
    crit_values_95: tuple[float, float]
    cointegrating_vector: tuple[float, float]
    n_cointegrating_relations_at_95: int


def adf_test(series: pd.Series, regression: str = "c") -> ADFResult:
    """Run ADF on a series and pack the result.

    Parameters
    ----------
    series : pd.Series
        Time series. Drops NaNs internally.
    regression : {"c", "ct", "n"}
        ADF deterministic-trend specification.

    Returns
    -------
    ADFResult
    """
    arr = series.dropna().to_numpy()
    if len(arr) < 20:
        raise ValueError(f"ADF requires at least ~20 obs, got {len(arr)}")
    stat, pvalue, used_lag, n_obs, crit, _icbest = adfuller(
        arr, regression=regression, autolag="AIC"
    )
    return ADFResult(
        statistic=float(stat),
        pvalue=float(pvalue),
        used_lag=int(used_lag),
        n_obs=int(n_obs),
        crit_1pct=float(crit["1%"]),
        crit_5pct=float(crit["5%"]),
        crit_10pct=float(crit["10%"]),
    )


def engle_granger_test(
    log_gold: pd.Series,
    log_silver: pd.Series,
) -> EngleGrangerResult:
    """Engle-Granger two-step cointegration test.

    Step 1: OLS log(GLD) = alpha + beta * log(SLV) + e.
    Step 2: ADF on residuals e (no constant, since OLS already removed mean).

    statsmodels' `coint(y0, y1)` runs both steps internally and returns
    a MacKinnon p-value that adjusts critical values for the residual
    regression. We use that p-value directly. We additionally fit OLS
    once to recover (alpha, beta) and the residual ADF detail.
    """
    s_gold, s_silver = log_gold.align(log_silver, join="inner")
    s_gold = s_gold.dropna()
    s_silver = s_silver.loc[s_gold.index].dropna()
    common = s_gold.index.intersection(s_silver.index)
    y = s_gold.loc[common].to_numpy()
    x = s_silver.loc[common].to_numpy()
    if len(common) < 50:
        raise ValueError(f"Engle-Granger needs ~50+ aligned obs, got {len(common)}")

    X = sm.add_constant(x)
    ols = sm.OLS(y, X).fit()
    alpha = float(ols.params[0])
    beta = float(ols.params[1])

    resid = ols.resid
    adf_stat, _adf_p_internal, used_lag, n_obs_adf, crit, _ = adfuller(
        resid, regression="n", autolag="AIC"
    )

    coint_t, coint_pvalue, coint_crit = coint(
        y, x, trend="c", autolag="AIC"
    )

    residual_adf = ADFResult(
        statistic=float(adf_stat),
        pvalue=float(coint_pvalue),
        used_lag=int(used_lag),
        n_obs=int(n_obs_adf),
        crit_1pct=float(coint_crit[0]),
        crit_5pct=float(coint_crit[1]),
        crit_10pct=float(coint_crit[2]),
    )
    return EngleGrangerResult(
        beta=beta,
        alpha=alpha,
        residual_adf=residual_adf,
        pvalue=float(coint_pvalue),
        n_obs=len(common),
    )


def johansen_test(log_gold: pd.Series, log_silver: pd.Series) -> JohansenResult:
    """Johansen trace test on (log GLD, log SLV).

    `coint_johansen(y, det_order, k_ar_diff)`:
      - det_order = 0 -> constant term in cointegration relation
      - k_ar_diff = 1 -> 1 lag in the differenced VAR
    Returns object with `.lr1` (trace stats), `.cvt` (90/95/99 crits),
    `.evec` (cointegrating vectors as columns).
    """
    df = pd.concat([log_gold, log_silver], axis=1, join="inner").dropna()
    if len(df) < 50:
        raise ValueError(f"Johansen needs ~50+ aligned obs, got {len(df)}")
    df.columns = ["log_gold", "log_silver"]
    res = coint_johansen(df.values, det_order=0, k_ar_diff=1)

    trace = (float(res.lr1[0]), float(res.lr1[1]))
    cv95 = (float(res.cvt[0, 1]), float(res.cvt[1, 1]))

    n_relations = 0
    if trace[0] > cv95[0]:
        n_relations = 1
        if trace[1] > cv95[1]:
            n_relations = 2

    vec = res.evec[:, 0]
    norm = vec / vec[0]
    coint_vec = (float(norm[0]), float(norm[1]))

    return JohansenResult(
        trace_stats=trace,
        crit_values_95=cv95,
        cointegrating_vector=coint_vec,
        n_cointegrating_relations_at_95=n_relations,
    )


def rolling_cointegration(
    log_gold: pd.Series,
    log_silver: pd.Series,
    window: int,
    step: int,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Rolling Engle-Granger cointegration p-value on sliding windows.

    Returns
    -------
    pd.DataFrame
        Indexed by the END date of each window, columns:
        ['pvalue', 'beta', 'alpha', 'n_obs'].
    """
    if cache_path is not None:
        cp = _resolve(cache_path)
        if cp.is_file():
            logger.info("loading rolling cointegration cache: %s", cp)
            return pd.read_parquet(cp)

    df = pd.concat([log_gold, log_silver], axis=1, join="inner").dropna()
    df.columns = ["log_gold", "log_silver"]
    n = len(df)
    if window > n:
        raise ValueError(f"window ({window}) > available obs ({n})")

    rows: list[tuple[pd.Timestamp, float, float, float, int]] = []
    for end in range(window - 1, n, step):
        start = end - window + 1
        sub = df.iloc[start : end + 1]
        try:
            r = engle_granger_test(sub["log_gold"], sub["log_silver"])
            rows.append((sub.index[-1], r.pvalue, r.beta, r.alpha, r.n_obs))
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.warning("rolling EG failed at %s: %s", sub.index[-1].date(), exc)

    out = pd.DataFrame(
        rows, columns=["end_date", "pvalue", "beta", "alpha", "n_obs"]
    ).set_index("end_date")
    logger.info("rolling cointegration: %d windows, window=%d, step=%d", len(out), window, step)

    if cache_path is not None:
        cp = _resolve(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cp)

    return out
