import yfinance as yf
import pandas as pd
import os

def download_metals_data():
    tickers = ["SI=F", "GC=F", "HG=F"]
    
    print("Fetching data from Yahoo Finance...")
    # We set auto_adjust=True explicitly. 
    # This makes 'Close' actually the 'Adjusted Close'.
    data = yf.download(tickers, period="5y", interval="1d", auto_adjust=True)
    
    # In the new version, 'data' is a MultiIndex. 
    # We want the 'Close' column for all tickers.
    df = data['Close'].dropna()
    
    # Rename columns to be human-readable
    df.columns = ['Copper', 'Gold', 'Silver']
     

    if not os.path.exists('data'):
        os.makedirs('data')
    
    df.to_csv("data/metals_data.csv")
    print("Success! Data saved to data/metals_data.csv")
    print(data.shape)

if __name__ == "__main__":
    download_metals_data()