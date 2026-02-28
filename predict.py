import pandas as pd
import joblib

def predict_next_day(data_path):
    df = pd.read_csv(data_path)

    latest = df.drop(['Date', 'Target'], axis=1).iloc[-1:]

    model = joblib.load("models/random_forest.pkl")
    prediction = model.predict(latest)

    print("Predicted Next-Day Price (Normalized):", prediction[0])

if __name__ == "__main__":
    predict_next_day("data/processed/featured_data.csv")