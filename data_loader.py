import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(ticker, start_date, end_date):
    # Ensure directory exists
    os.makedirs("data/raw", exist_ok=True)

    # Download data
    df = yf.download(ticker, start=start_date, end=end_date)

    # 🔥 FIX: Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Reset index so Date becomes a column
    df.reset_index(inplace=True)

    # Save clean CSV
    df.to_csv("data/raw/stock_data.csv", index=False)

    print("✅ Stock data downloaded and saved successfully")
    return df


if __name__ == "__main__":
    fetch_stock_data("TCS.NS", "2015-01-01", "2024-01-01")