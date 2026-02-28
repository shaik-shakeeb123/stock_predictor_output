import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def preprocess_data(input_path, output_path):
    # Ensure output directory exists
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Load raw data
    df = pd.read_csv(input_path)

    # Convert Date column
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # ✅ KEEP ONLY NUMERIC STOCK COLUMNS
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Remove missing values
    df.dropna(inplace=True)

    # Scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Save scaler
    joblib.dump(scaler, "models/scaler.pkl")

    # Create scaled DataFrame
    df_scaled = pd.DataFrame(
        scaled_data,
        columns=df.columns,
        index=df.index
    )

    # Save cleaned data
    df_scaled.to_csv(output_path)

    print("✅ Preprocessing completed successfully")
    return df_scaled


if __name__ == "__main__":
    preprocess_data(
        "data/raw/stock_data.csv",
        "data/processed/cleaned_data.csv"
    )