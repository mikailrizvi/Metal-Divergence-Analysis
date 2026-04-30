# Gold-Silver Statistical Arbitrage: Cointegration-Based Pairs Trading

## Project Overview

A rigorous, defensible pairs trading research project on the gold-silver spread. The goal is not to claim that pairs trading "works" but to test it scientifically: establish the statistical relationship, build a properly validated backtest, stress-test across regimes, and document honestly what works and what breaks.

This project is designed to be defensible in a quant interview. Every methodological choice has a justification. Every reported number traces back to reproducible code.

## Thesis

Gold and silver share macroeconomic drivers — real rates, inflation expectations, dollar strength, safe-haven demand — and historically exhibit a stable long-run relationship. Short-term deviations from that relationship may revert, creating opportunities for a market-neutral pairs trade.

The honest version of this thesis: the relationship is regime-dependent. It holds during normal markets and breaks during dislocations (2011 silver squeeze, COVID liquidity events). A useful strategy must either model that regime structure or filter out periods where the relationship breaks down.

---

## How to Use This Document with Claude Code

This is your master spec. Save it as `PLAN.md` in your project root. Also create a `CLAUDE.md` (template at the bottom of this file) for persistent coding conventions.

### Initial Setup Before Prompting

1. Create empty project directory and initialize git: `git init`
2. Save this file as `PLAN.md`
3. Save the `CLAUDE.md` template (bottom of this doc) in the project root
4. Open Claude Code in the directory

### The Opening Prompt

Paste this as your first message after Claude Code has read `PLAN.md` and `CLAUDE.md`:

> We're building the project in PLAN.md. Rules of engagement:
> 1. Work one phase at a time. Stop after each phase and wait for approval before continuing.
> 2. For each phase, write file structure and function signatures as stubs first. Show me. After I approve, fill in the implementations.
> 3. Write pytest unit tests for critical functions (cointegration, OU fitting, signal generation, backtester).
> 4. Commit to git after each phase with a descriptive message.
> 5. At every checkpoint listed in the plan, show me the result before continuing.
> 6. If you're uncertain about a library API or a methodology choice, say so rather than guessing. For statsmodels functions specifically, verify the actual API by checking the docstring before using it.
> 7. Flag any time a result looks too good. Out-of-sample Sharpe > 2.5 is suspicious and likely indicates a bug.
> 8. Set numpy and any other random seeds for reproducibility.
> 9. Cache the yfinance data pull to disk on first download; load from disk thereafter.
>
> Confirm you understand and ask any setup questions before starting Phase 1.

### Per-Phase Prompting Pattern

For each phase, prompt with this structure:

> Implement Phase [N] from PLAN.md. Scope is exactly: [list the deliverables for that phase]. Stop after completing this phase. Do not start Phase [N+1].
>
> Before writing implementation code:
> 1. Ask me any clarifying questions about ambiguities.
> 2. Write the function stubs with type hints and docstrings.
> 3. Show me the stubs and acceptance criteria. Wait for approval.
>
> After implementation:
> 1. Run the tests.
> 2. Show me the checkpoint output specified in the plan.
> 3. Wait for me to confirm before committing.

### When Something Breaks

Do not say "fix it." Force diagnostic thinking:

> The [specific result] looks wrong because [reason]. Trace through the [relevant pipeline] and identify where the problem could be. Show me the analysis with hypotheses ranked by likelihood before changing any code.

This prevents Claude Code from patching symptoms.

### Token Budget Strategy

Even on Max with Opus 4.7, this is a multi-hour project. Context degrades when the conversation gets long. Strategy:

- Start a fresh Claude Code session for each phase if the prior conversation is over 30-40 messages.
- `PLAN.md` and `CLAUDE.md` carry the state between sessions.
- After each phase, ask Claude Code to update `PROGRESS.md` with what's been done. That becomes your handoff doc.

---

## Tech Stack

- Python 3.11+
- pandas, numpy for data manipulation
- statsmodels for econometrics (ADF, Engle-Granger, Johansen, OU fitting)
- yfinance for price data
- matplotlib and seaborn for visualization
- scipy for optimization
- pytest for testing
- Jupyter for exploratory notebooks
- Pure Python backtester (no third-party backtesting library, written from scratch)

## Repository Structure

```
gold-silver-pairs/
├── README.md
├── PLAN.md                  (this file)
├── CLAUDE.md                (coding conventions for Claude Code)
├── PROGRESS.md              (running log, updated after each phase)
├── requirements.txt
├── config.yaml              (all strategy parameters live here, no magic numbers)
├── data/
│   ├── raw/                 (cached yfinance pulls)
│   └── processed/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_cointegration_analysis.ipynb
│   ├── 03_ou_modeling.ipynb
│   ├── 04_backtest.ipynb
│   └── 05_stress_testing.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py            (loads config.yaml into a Config dataclass)
│   ├── data_loader.py
│   ├── cointegration.py
│   ├── ou_process.py
│   ├── signals.py
│   ├── backtester.py
│   ├── performance.py
│   └── plotting.py
├── tests/
│   ├── test_cointegration.py
│   ├── test_ou_process.py
│   ├── test_signals.py
│   └── test_backtester.py
├── results/
│   ├── plots/
│   └── tables/
└── writeup.md
```

---

## Universal Engineering Standards

These apply to every phase. They live in `CLAUDE.md` too but are restated here for emphasis.

- **Type hints on every function signature.** No exceptions.
- **Docstrings on every public function** in numpy or Google style. Include parameter types, return types, and a one-line summary.
- **Use `@dataclass`** for structured data: `Trade`, `Position`, `BacktestResult`, `Config`. Never pass loose dicts around.
- **Logging, not print.** Use Python's `logging` module. INFO level for backtest progress, DEBUG for per-trade details.
- **Set random seeds** at the top of every script that has any randomness.
- **No magic numbers.** All thresholds, window sizes, costs go in `config.yaml`.
- **No bare except.** Catch specific exceptions.
- **Pin dependencies in `requirements.txt`.** Use `pip freeze` after install.
- **No silent data interpolation.** Forward-fill at most 1 day of missing data; drop longer gaps; log every gap found.
- **Avoid lookahead like a fire.** Signals at time t use only data up to and including t. Trade execution at t+1's open. Write a unit test that enforces this.
- **Cache expensive operations.** yfinance pulls, cointegration tests on rolling windows. Save to disk, reload on subsequent runs.

---

## Phase 1: Data Acquisition and Validation

### Goals

Pull clean, validated daily price data for gold and silver covering at least 15 years to span multiple market regimes.

### Tasks

1. Pull daily adjusted close prices for GLD (gold ETF) and SLV (silver ETF) from yfinance
2. Date range: 2007-01-01 to present (covers 2008 crisis, 2011 silver squeeze, 2020 COVID, 2022 inflation)
3. Cache raw pull to `data/raw/`. Subsequent runs load from disk unless a `--refresh` flag is set.
4. Validate: check for missing dates, look for outliers, verify split-adjusted prices align with public records
5. Forward-fill at most 1 day of missing data; drop longer gaps; log everything
6. Compute log prices for cointegration analysis
7. Save processed data to `data/processed/`

### Implementation Notes

Use `yfinance.download` with `auto_adjust=True`. Handle multi-index column edge case. Document any data quality issues found in code comments.

### Checkpoint (Show Before Continuing)

- Print summary: date range, number of observations, missing days handled, any anomalies
- Plot GLD and SLV prices on dual-axis chart
- Plot daily log returns and their distributions
- Tests pass

### Deliverables

- `src/data_loader.py` with `load_pair_data(refresh: bool = False) -> pd.DataFrame`
- `tests/test_data_loader.py` (smoke tests for shape, date alignment, no NaN beyond logged gaps)
- `notebooks/01_data_exploration.ipynb`

---

## Phase 2: Cointegration Testing

### Theory Reminder

Two time series are cointegrated if each is individually non-stationary (random walk), but a linear combination of them is stationary (mean-reverting). Cointegration is what makes pairs trading work in principle. Without it, the spread doesn't reliably mean-revert and the strategy is gambling.

### Tasks

1. ADF test on log(GLD) and log(SLV) individually. Expected: fail to reject unit root.
2. Engle-Granger two-step procedure:
   - Regress log(GLD) on log(SLV) and a constant. Save residuals.
   - ADF test on residuals. Reject null = cointegrated.
3. Johansen test as a secondary check.
4. Rolling cointegration: 2-year windows, sliding monthly. Plot p-value over time.
5. Cache rolling results to disk (it's slow to recompute).

### Implementation Notes

Use `statsmodels.tsa.stattools.adfuller`, `statsmodels.tsa.stattools.coint`, and `statsmodels.tsa.vector_ar.vecm.coint_johansen`. The Johansen test return object is non-obvious; check the docstring before using.

### Checkpoint

- Full-sample p-values for ADF (each leg), Engle-Granger, Johansen
- Rolling p-value plot over time. Should show clear regime variation. If it's flat, something is wrong.
- Hypothesis: 2011 and possibly COVID windows show p-values rising above 0.10
- Tests pass

### Deliverables

- `src/cointegration.py` with `engle_granger_test()`, `johansen_test()`, `rolling_cointegration()`
- `tests/test_cointegration.py`
- `notebooks/02_cointegration_analysis.ipynb`

---

## Phase 3: Spread Construction

### Theory Reminder

Don't use raw price differences (different scales). Use the cointegrating regression residual:

```
spread_t = log(GLD_t) - β * log(SLV_t) - α
```

where β and α come from the Engle-Granger regression. Recompute β on a rolling basis to handle drift in the relationship.

### Tasks

1. Implement spread construction with both static and rolling β (60-day rolling default, configurable)
2. Plot the spread, mark the long-term mean
3. Compute spread statistics: mean, std, skew, autocorrelation (lag-1, lag-5)
4. Use rolling β for the actual strategy

### Checkpoint

- Spread plot looks bounded and mean-reverting (not trending)
- Lag-1 autocorrelation is high (>0.9 typical for daily data) but not 1.0
- Tests pass

### Deliverables

- Spread construction in `src/cointegration.py`
- Spread plots and stats in notebook 02

---

## Phase 4: Ornstein-Uhlenbeck Modeling

### Theory Reminder

The Ornstein-Uhlenbeck process is the canonical continuous-time mean-reverting process:

```
dX_t = θ(μ - X_t)dt + σ dW_t
```

- θ: speed of mean reversion (higher = faster reversion)
- μ: long-term mean
- σ: volatility
- Half-life of mean reversion: ln(2) / θ

Half-life tells us the average time for the spread to revert halfway to its mean. It's the key input for setting holding periods and stop-times.

### Tasks

1. Fit OU parameters via OLS on the discrete-time AR(1) form: regress (X_t - X_{t-1}) on X_{t-1}. Slope coefficient = -θ * Δt where Δt = 1 day.
2. Compute half-life = ln(2) / θ.
3. Use half-life to set:
   - Rolling z-score window length (4x half-life, rounded to integer days)
   - Time-stop for trades (3x half-life days)

### Implementation Notes

Verify half-life is in 5-50 day range. If negative or >100 days, the spread isn't actually mean-reverting in this period — that's a finding to document, not a bug to hide.

### Checkpoint

- Half-life value (single number for full sample, plus rolling estimates)
- Sanity check: 5-50 days
- If outside that range, halt and document
- Tests pass

### Deliverables

- `src/ou_process.py` with `fit_ou()`, `half_life()`
- `tests/test_ou_process.py` (test on synthetic OU data with known parameters)
- `notebooks/03_ou_modeling.ipynb`

---

## Phase 5: Signal Generation

### Tasks

1. Compute rolling z-score of the spread. Window = 4x half-life.
2. Define signal logic:
   - Entry: |z| > 2 (enter on next bar's open)
   - Exit: |z| < 0.5 OR z crosses zero
   - Stop-loss: |z| > 4
   - Time-stop: exit after 3x half-life days regardless
3. Generate position series:
   - z > 2: short spread (short GLD, long β units of SLV)
   - z < -2: long spread (long GLD, short β units of SLV)
   - Otherwise: hold existing position or stay flat

### Checkpoint

- Signals plotted overlaid on z-score chart
- Number of trades per year is reasonable (10-50 typical)
- Tests pass, including the lookahead test (signal at t depends only on data up to t-1 or earlier)

### Deliverables

- `src/signals.py` with `generate_signals()`
- `tests/test_signals.py` including a lookahead bias test

---

## Phase 6: Backtesting Engine

### Design Principles

- Walk-forward, not in-sample
- Fit β and OU parameters on training window only
- Apply signals out-of-sample
- Realistic frictions
- Vectorized pandas implementation is fine for daily bars

### Walk-Forward Structure

- Training window: 3 years (rolling)
- Test window: 1 year (out-of-sample)
- Step: 1 year forward
- Concatenate OOS periods for final performance

### Frictions

- Transaction costs: 5 bps per leg, 20 bps round-trip total per pair trade
- Bid-ask: execute at mid + half-spread on entry, mid - half-spread on exit
- Borrow costs: 50 bps annualized for short side
- Position sizing: equal dollar exposure per leg, scaled by β
- Capital: $100,000 starting

### Tasks

1. Implement vectorized pandas backtester
2. Track positions, P&L, equity curve, trade log (use `Trade` dataclass)
3. Apply all frictions correctly
4. Walk-forward refactor: separate training fit from out-of-sample evaluation
5. Critical: never use future information

### Checkpoint

- v1 in-sample equity curve looks plausible
- Walk-forward OOS equity curve drops vs in-sample (it must — if it doesn't, lookahead is leaking)
- OOS Sharpe < 2.5. If higher, hunt for bugs before continuing.
- Trade log inspectable: random 10 trades make sense
- Tests pass

### Deliverables

- `src/backtester.py` with `Backtester` class and `BacktestResult` dataclass
- `tests/test_backtester.py` with synthetic-data tests and lookahead test
- `notebooks/04_backtest.ipynb`

---

## Phase 7: Performance Analytics

### Metrics

- Total return (gross and net of costs)
- Annualized return
- Annualized volatility
- Sharpe ratio (annualized, using daily returns)
- Sortino ratio
- Max drawdown and duration
- Calmar ratio
- Hit rate (% of profitable trades)
- Average win / average loss
- Profit factor (gross profits / gross losses)
- Number of trades, average holding period
- Correlation to S&P 500 (target: low)
- Beta to S&P 500 (target: ~0)
- Annual returns table

### Plots

- Equity curve (strategy vs SPY benchmark)
- Drawdown curve
- Rolling 1-year Sharpe
- Trade-level scatter (z-score at entry vs P&L)
- Distribution of returns
- Annual returns bar chart

### Checkpoint

- All metrics in a single summary table
- All plots saved to `results/plots/`
- Sanity check: Sharpe to S&P correlation should be low (<0.3 ideally)

### Deliverables

- `src/performance.py` with all metrics
- `src/plotting.py` for all charts
- All output saved to `results/plots/` and `results/tables/`

---

## Phase 8: Stress Testing and Regime Analysis

### The Killer Section

This is where the project moves from "good freshman project" to "this person thinks like a quant."

### Tasks

1. Slice performance by year. Identify which years drive returns and which drag.
2. Specific stress windows:
   - 2008 crisis (Sept 2008 - March 2009)
   - 2011 silver squeeze (April - May 2011)
   - 2020 COVID (Feb - April 2020)
   - 2022 inflation shock (full year)
3. For each: return, max DD, hit rate, cointegration p-value during the window
4. Propose and test mitigations:
   - Vol filter: stop trading when 20-day realized vol of either leg exceeds 95th percentile
   - Cointegration filter: stop trading when rolling Engle-Granger p-value exceeds 0.10
   - Combined filter
5. Compare filtered vs unfiltered. Document honestly which filters help.

### Checkpoint

- Year-by-year performance table
- Stress window analysis table
- Filtered vs unfiltered equity curve comparison
- Honest conclusion: which filter helps, which doesn't, why

### Deliverables

- `notebooks/05_stress_testing.ipynb`
- Updated equity curves with filters
- Honest writeup section

---

## Phase 9: Writeup and Documentation

### README.md

- Project summary (2-3 paragraphs)
- Key results table (Sharpe, max DD, returns)
- Equity curve image
- How to reproduce
- File structure
- Honest limitations section

### writeup.md

A 5-10 page research-note style writeup:

1. Abstract
2. Background and motivation
3. Data
4. Cointegration analysis (full sample + rolling p-value plot)
5. OU modeling and half-life
6. Strategy construction
7. Backtest methodology (emphasize walk-forward, costs, no lookahead)
8. Results (tables, charts)
9. Stress testing and regime analysis
10. Limitations and future work
11. References

### Reference Papers (cite these, know the core ideas)

- Gatev, Goetzmann, Rouwenhorst (2006), "Pairs Trading: Performance of a Relative Value Arbitrage Rule"
- Avellaneda and Lee (2010), "Statistical Arbitrage in the U.S. Equities Market"
- Vidyamurthy (2004), "Pairs Trading: Quantitative Methods and Analysis" (book)

### Honest Caveats for the Limitations Section

- Daily data only — intraday execution dynamics not modeled
- ETFs not futures — production version would use rolled GC/SI futures
- Borrow costs estimated, not actual broker quotes
- Not adjusted for capacity (strategy may not scale to large AUM)
- The pair was chosen because it's known to work historically; out-of-sample for an unknown future pair is harder

---

## Defensibility Checklist

Before declaring the project done, verify you can answer all of the following without notes:

- Why cointegration and not just correlation?
- What's the difference between Engle-Granger and Johansen?
- Why is half-life important and how is it computed from OU parameters?
- Why walk-forward and not in-sample?
- Where does lookahead bias hide and how did you avoid it?
- What were the most realistic transaction cost assumptions and why?
- During which historical period did the strategy perform worst and why?
- If the cointegration p-value rises above some threshold, should you stop trading? Why?
- What's the strategy's correlation to SPX and why does that matter?
- What's the next thing you'd add if you had another month?

---

## Resume Bullets (Draft, Fill in Real Numbers After Build)

DO NOT fill these in with plausible-sounding numbers. Run the backtest, get the actual numbers, then fill in.

- Implemented a market-neutral gold-silver pairs trading strategy using Engle-Granger cointegration testing (p<0.01) and modeled the spread as an Ornstein-Uhlenbeck process with estimated half-life of [X] days to inform holding-period rules.
- Built a walk-forward Python backtester (pandas, numpy, statsmodels) covering 17+ years of daily data with 20bps round-trip transaction costs and borrow fees, achieving out-of-sample Sharpe of [X] and max drawdown of [Y]%.
- Stress-tested through 2008, 2011 silver squeeze, and March 2020 dislocations; identified regime-dependent breakdown of cointegration and implemented a rolling-p-value filter that improved out-of-sample Sharpe from [X] to [Y].

---

## Sanity Checkpoints (Stop and Verify)

These are the moments where bugs hide. Stop at each, look at output, then continue.

- **After Phase 1:** Price plots look right. No silent NaN handling.
- **After Phase 2:** Rolling cointegration p-value tells a story (not flat). Bumps near 2011 and 2020.
- **After Phase 4:** Half-life in 5-50 day range. If not, halt and investigate.
- **After Phase 6 v1:** Sharpe > 3 means bug, not alpha. Audit before continuing.
- **After Phase 6 walk-forward:** OOS Sharpe drops from in-sample. If it doesn't drop at all, lookahead remains.
- **After Phase 8:** Filters that help should help in interpretable ways (vol filter helps in 2008/2020, cointegration filter helps in 2011). If filters help everywhere or nowhere, something's off.

---

## Today's Execution Plan

If running this with Claude Code in one session:

1. Setup (repo init, requirements, data loader, validation, Phase 1) — 45 min
2. Cointegration analysis (Phase 2) — 60 min
3. Spread construction + OU fitting (Phases 3-4) — 60 min
4. Signals + v1 backtester (Phases 5-6 first half) — 75 min
5. Walk-forward + frictions (Phase 6 second half) — 45 min
6. Performance analytics (Phase 7) — 45 min
7. Stress testing (Phase 8) — 45 min
8. Writeup and polish (Phase 9) — 60 min

Total: ~7-8 hours focused work. Realistic version: split across two days with the writeup on day 2.

---

## CLAUDE.md Template

Save this content separately as `CLAUDE.md` in the project root. Claude Code reads it on every invocation.

```markdown
# Coding Conventions for This Project

## Style
- Python 3.11+
- Type hints on all function signatures
- Numpy-style docstrings on all public functions
- Use @dataclass for structured data
- snake_case for functions and variables, PascalCase for classes
- Max line length 100

## Imports
- Standard library first, then third-party, then local
- One import per line for local imports
- No wildcard imports

## Logging
- Use the `logging` module, not print statements
- Set up a module-level logger: `logger = logging.getLogger(__name__)`
- INFO level for high-level progress, DEBUG for details, WARNING for anomalies

## Data Handling
- Forward-fill at most 1 day of missing data
- Drop longer gaps and log them
- Set numpy random seed at top of any script with randomness: `np.random.seed(42)`
- Cache yfinance pulls to disk; reload from disk on subsequent runs

## Configuration
- All strategy parameters live in config.yaml
- Loaded into a Config dataclass via src/config.py
- No magic numbers in source files

## Testing
- pytest, tests in tests/ directory
- Test critical functions: cointegration, OU fit, signals, backtester
- Critical: include a lookahead-bias test for the backtester

## Backtesting Discipline
- Signal at time t uses only data up to and including t
- Trade execution at t+1 open
- Walk-forward, not in-sample
- All frictions modeled (transaction costs, bid-ask, borrow)

## Git
- Commit after each phase
- Descriptive commit messages: "Phase 2: cointegration testing complete"
- Never commit data/raw or large output files (add to .gitignore)

## When Uncertain
- For library APIs (especially statsmodels), verify by checking docstrings before using
- Flag any out-of-sample Sharpe > 2.5 as suspicious
- If a result looks too good, halt and audit before proceeding
- Never patch symptoms — diagnose root cause first
```
