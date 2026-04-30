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
