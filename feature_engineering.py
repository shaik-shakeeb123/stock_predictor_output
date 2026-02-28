import pandas as pd
import numpy as np

def create_features(input_path, output_path):
    df = pd.read_csv(input_path)

    # Date handling
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Moving Averages
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()

    # Daily return (can cause inf)
    df['Daily_Return'] = df['Close'].pct_change()

    # Target = next day close
    df['Target'] = df['Close'].shift(-1)

    # 🔥 CRITICAL FIX
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop all NaN rows
    df.dropna(inplace=True)

    # Save clean featured data
    df.to_csv(output_path)

    print("✅ Feature engineering completed successfully")
    return df


if __name__ == "__main__":
    create_features(
        "data/processed/cleaned_data.csv",
        "data/processed/featured_data.csv"
    )