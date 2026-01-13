import pandas as pd
import numpy as np

def run_backtest_final(data, window, z_thresh, cost_bps=5):
    """
    data: DataFrame with a 'GSR' column (Gold/Silver Ratio)
    window: Rolling window for mean/std
    z_thresh: Z-score threshold for entry
    cost_bps: Transaction costs in Basis Points (5 bps = 0.05%)
    """
    df = data.copy()
    
    # 1. Standard Signal Logic
    df['Rolling_Mean'] = df['GSR'].rolling(window=window).mean()
    df['Rolling_Std'] = df['GSR'].rolling(window=window).std()
    df['Z_Score'] = (df['GSR'] - df['Rolling_Mean']) / df['Rolling_Std']
    
    # 2. Position Logic (1 = Long, -1 = Short, 0 = Cash)
    df['Position'] = 0
    df.loc[df['Z_Score'] > z_thresh, 'Position'] = -1  # Ratio too high: Sell Gold/Buy Silver
    df.loc[df['Z_Score'] < -z_thresh, 'Position'] = 1  # Ratio too low: Buy Gold/Sell Silver
    
    # 3. Returns Calculation (RAW DECIMALS)
    # Important: .pct_change() gives 0.01 for 1%, which is what we want.
    df['GSR_Ret'] = df['GSR'].pct_change()
    
    # Shift position by 1 day to avoid "Look-Ahead Bias" 
    # (You trade today based on yesterday's close)
    df['Gross_Ret'] = df['Position'].shift(1) * df['GSR_Ret']
    
    # 4. Transaction Costs
    # We pay the cost when posiiton changes
    df['Trades'] = df['Position'].diff().fillna(0).abs()
    
    cost_pct = cost_bps / 10000
    df['Transaction_Costs'] = df['Trades'] * cost_pct
    
    # 5. Final Net Return
    df['Net_Ret'] = df['Gross_Ret'] - df['Transaction_Costs']
    
    return df['Net_Ret'].fillna(0)