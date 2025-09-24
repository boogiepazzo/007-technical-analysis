# AGNC Technical Analysis - Data Download and Preparation
# This module handles data download and basic preparation

import config
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def download_and_prepare_data():
    """Download AGNC data and prepare it for analysis"""
    global df, r, px, df_raw
    
    print("Downloading AGNC data...")
    start = (datetime.today() - timedelta(days=int(config.YEARS_BACK*365.25))).strftime("%Y-%m-%d")
    df_raw = config.yf.download(config.TICKER, start=start, progress=False, auto_adjust=True)
    df = df_raw.copy()

    # Ensure proper datetime index for time series analysis
    df.index = pd.to_datetime(df.index)

    px = df["Close"].copy()  # use close price for returns (auto_adjust=True handles adjustments)
    r = np.log(px).diff().dropna() * 100.0  # % log-returns

    print(f"Data downloaded successfully!")
    print(f"- Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"- Total trading days: {len(df)}")
    print(f"- Current price: ${float(df['Close'].iloc[-1]):.2f}")
    print(f"- Price range: ${float(df['Low'].min()):.2f} - ${float(df['High'].max()):.2f}")

    # Display basic statistics
    print(f"\nBasic Statistics:")
    print(f"- Mean daily return: {float(r.mean()):.3f}%")
    print(f"- Volatility (std): {float(r.std()):.3f}%")
    print(f"- Min daily return: {float(r.min()):.3f}%")
    print(f"- Max daily return: {float(r.max()):.3f}%")

    return df, r, px, df_raw

if __name__ == "__main__":
    download_and_prepare_data()
