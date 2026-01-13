import numpy as np
import pandas as pd

def calculate_sharpe_ratio(returns, risk_free_rate=0):
    clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    if clean_returns.std() == 0 or len(clean_returns) < 2:
        return 0.0
    
    # Calculate daily sharpe
    mean_daily = clean_returns.mean()
    std_daily = clean_returns.std()
    
    daily_sharpe = (mean_daily - (risk_free_rate/252)) / std_daily
    
    # Annualize it
    ann_sharpe = daily_sharpe * np.sqrt(252)
    
    return ann_sharpe

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