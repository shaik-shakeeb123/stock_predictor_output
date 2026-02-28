import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def train_models(data_path):
    df = pd.read_csv(data_path)

    X = df.drop(['Date', 'Target'], axis=1)
    y = df['Target']

    split = int(len(df) * 0.8)
    X_train, y_train = X[:split], y[:split]

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    joblib.dump(lr, "models/linear_regression.pkl")

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, "models/random_forest.pkl")

if __name__ == "__main__":
    train_models("data/processed/featured_data.csv")