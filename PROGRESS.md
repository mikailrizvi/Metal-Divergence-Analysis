# Progress Log

Running log of completed phases. Updated after each phase ships.

## Phase 0 — Scaffold
- Repo initialized, remote set to `Metal-Divergence-Analysis`
- Directory structure, .gitignore, config.yaml, CLAUDE.md, README.md, PLAN.md, requirements.txt in place
- Single notebook `notebooks/analysis.ipynb` will accumulate all phase work
- Fresh `.venv` created

## Phase 1 — Data Acquisition and Validation
- `src/config.py` — loads config.yaml into frozen dataclasses
- `src/data_loader.py` — yfinance pull + cache, `DataQualityReport` dataclass, `load_pair_data(cfg, refresh)`
- 8 offline tests + 1 network-marked smoke test, all pass
- pytest.ini deselects network tests by default (`pytest -m network` to opt in)
- Real pull: **4,861 trading days**, 2007-01-03 → 2026-04-29, 0 NaNs, 0 forward-fills
- One unusual market gap correctly logged (no imputation): Hurricane Sandy, 2012-10-26 → 2012-10-31 (2 missing trading days, NYSE closed Mon+Tue)
- GLD/SLV daily log-return correlation: **0.79** (high — consistent with a shared macro driver, motivates cointegration test)
- Plots saved: `results/plots/01_prices_dual_axis.png`, `results/plots/01_log_returns.png`

### Bug caught and fixed during Phase 1
- Initial `_validate_and_clean` reindexed onto `pd.bdate_range`, which **invented price observations on NYSE holidays** (~178 forward-filled days). Caught at the checkpoint by auditing the FF count. Replaced with use-yfinance-index-as-canonical + missing-business-day gap detection. Regression test added (`test_market_holidays_not_synthesized`).

## Phase 2 — Cointegration Testing
- `src/cointegration.py` — `adf_test`, `engle_granger_test`, `johansen_test`, `rolling_cointegration` with parquet-cached output. Frozen `ADFResult`, `EngleGrangerResult`, `JohansenResult` dataclasses.
- 7 unit tests on synthetic data (cointegrated pair detected, independent random walks not detected, Johansen recovers the true vector). All passing.
- statsmodels APIs verified: `adfuller(autolag='AIC')`, `coint(trend='c')` for MacKinnon p-value, `coint_johansen(det_order=0, k_ar_diff=1)`.

### Phase 2 finding (significant)
**Full-sample cointegration is REJECTED** on 2007-01-03 → 2026-04-29:
- log(GLD) ADF p = 0.945, log(SLV) ADF p = 0.877 → both I(1) as required
- Engle-Granger: β = 0.851, α = 2.391, residual MacKinnon **p = 0.34** → fail to reject unit root
- Johansen trace: r=0 stat 7.77 vs 95% crit 15.49 → **0 cointegrating relations**
- Johansen implied β = 1.016, EG β = 0.851 — disagreement is itself diagnostic of an unstable relationship

**Rolling EG (504-day window, 21-day step, 208 windows):**
- 5.3% of windows reach p < 0.05; 9.1% reach p < 0.10
- Rolling β wanders from 0.03 to 1.12 (mean 0.50)
- p-value spikes cluster on predicted regime breaks: 2011 silver squeeze, 2019-20 COVID, and a fresh spike during the 2024-26 gold rally

**Implication for downstream phases:**
- Static β is unsafe; Phase 3 must use rolling β (already planned)
- The rolling-p-value filter (originally Phase 8) is **load-bearing**, not a refinement — only trade when rolling EG p < 0.10
- This is the project's defensible thesis: GLD/SLV is *conditionally* cointegrated, the strategy detects when it is and filters out when it isn't

Plot: `results/plots/02_rolling_cointegration.png`
