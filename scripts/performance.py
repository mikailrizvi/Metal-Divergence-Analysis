import numpy as np
import pandas as pd

def calculate_sharpe_ratio(returns, rf_rate=0.04):
    """Calculates annualized Sharpe Ratio."""
    if returns.std() == 0:
        return 0
    ann_ret = (1 + returns.mean())**252 - 1
    ann_vol = returns.std() * np.sqrt(252)
    return (ann_ret - rf_rate) / ann_vol

def calculate_max_drawdown(cum_returns):
    """Calculates the maximum peak-to-trough decline."""
    peaks = cum_returns.expanding().max()
    drawdown = (cum_returns / peaks) - 1
    return drawdown.min()

def get_strategy_stats(df, returns_col, label="Strategy"):
    """Returns a dictionary of key performance metrics."""
    returns = df[returns_col].fillna(0)
    cum_returns = (1 + returns).cumprod()
    
    stats = {
        "Label": label,
        "Total Return": f"{(cum_returns.iloc[-1] - 1):.2%}",
        "Ann. Volatility": f"{(returns.std() * np.sqrt(252)):.2%}",
        "Sharpe Ratio": round(calculate_sharpe_ratio(returns), 2),
        "Max Drawdown": f"{calculate_max_drawdown(cum_returns):.2%}"
    }
    return stats