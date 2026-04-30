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
