import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_models(data_path):
    df = pd.read_csv(data_path)

    X = df.drop(['Date', 'Target'], axis=1)
    y = df['Target']

    split = int(len(df) * 0.8)
    X_test, y_test = X[split:], y[split:]

    lr = joblib.load("models/linear_regression.pkl")
    rf = joblib.load("models/random_forest.pkl")

    for name, model in [("Linear Regression", lr), ("Random Forest", rf)]:
        pred = model.predict(X_test)
        print(f"\n{name}")
        print("MAE :", mean_absolute_error(y_test, pred))
        print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
        print("R²  :", r2_score(y_test, pred))

if __name__ == "__main__":
    evaluate_models("data/processed/featured_data.csv")