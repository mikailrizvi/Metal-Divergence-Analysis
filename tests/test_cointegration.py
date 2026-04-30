"""Unit tests for `src.cointegration`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cointegration import (
    adf_test,
    engle_granger_test,
    johansen_test,
    rolling_cointegration,
)


def _stationary_ar1(n: int, rho: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, 1, n)
    x = np.empty(n)
    x[0] = 0.0
    for i in range(1, n):
        x[i] = rho * x[i - 1] + eps[i]
    return pd.Series(x, index=pd.bdate_range("2010-01-01", periods=n))


def _random_walk(n: int, seed: int, sigma: float = 1.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        np.cumsum(rng.normal(0, sigma, n)),
        index=pd.bdate_range("2010-01-01", periods=n),
    )


def _cointegrated_pair(n: int, beta_true: float, seed: int) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    x = pd.Series(
        np.cumsum(rng.normal(0, 1, n)),
        index=pd.bdate_range("2010-01-01", periods=n),
    )
    noise = rng.normal(0, 0.5, n)
    y = beta_true * x + pd.Series(noise, index=x.index)
    return y, x


def test_adf_rejects_stationary() -> None:
    s = _stationary_ar1(500, rho=0.5, seed=0)
    r = adf_test(s)
    assert r.pvalue < 0.05


def test_adf_does_not_reject_random_walk() -> None:
    s = _random_walk(500, seed=1)
    r = adf_test(s)
    assert r.pvalue > 0.10


def test_engle_granger_detects_synthetic_pair() -> None:
    y, x = _cointegrated_pair(n=500, beta_true=2.0, seed=2)
    r = engle_granger_test(y, x)
    assert r.pvalue < 0.05
    assert abs(r.beta - 2.0) < 0.1


def test_engle_granger_rejects_independent_random_walks() -> None:
    pvals = []
    for seed in range(10):
        y = _random_walk(400, seed=10 + seed)
        x = _random_walk(400, seed=20 + seed)
        pvals.append(engle_granger_test(y, x).pvalue)
    assert np.mean([p > 0.05 for p in pvals]) >= 0.7


def test_johansen_finds_one_cointegrating_relation() -> None:
    y, x = _cointegrated_pair(n=500, beta_true=2.0, seed=3)
    r = johansen_test(y, x)
    assert r.n_cointegrating_relations_at_95 >= 1
    assert r.cointegrating_vector[0] == pytest.approx(1.0)
    assert abs(r.cointegrating_vector[1] + 2.0) < 0.5


def test_rolling_cointegration_shape() -> None:
    y, x = _cointegrated_pair(n=600, beta_true=2.0, seed=4)
    out = rolling_cointegration(y, x, window=200, step=50)
    expected_n = (600 - 200) // 50 + 1
    assert len(out) == expected_n
    assert list(out.columns) == ["pvalue", "beta", "alpha", "n_obs"]
    assert out.index.is_monotonic_increasing


def test_rolling_cointegration_cache_roundtrip(tmp_path: Path) -> None:
    y, x = _cointegrated_pair(n=400, beta_true=2.0, seed=5)
    cache = tmp_path / "rolling.parquet"
    first = rolling_cointegration(y, x, window=200, step=50, cache_path=cache)
    second = rolling_cointegration(y, x, window=200, step=50, cache_path=cache)
    pd.testing.assert_frame_equal(first, second)
    assert cache.is_file()
