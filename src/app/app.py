from flask import Flask, render_template, request
from datetime import timedelta
import yfinance as yf
import pandas as pd
import joblib
import os
import requests
import numpy as np
from pandas.tseries.offsets import BDay

FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "MA_5",
    "MA_10",
    "Daily_Return"
]

app = Flask(__name__)

def multi_day_forecast(model, scaler, df, days=5):
    forecasts = []
    temp_df = df.copy()

    for _ in range(days):
        X = temp_df[FEATURE_COLUMNS].iloc[-1:]
        pred_norm = model.predict(X)[0]

        dummy = [0, 0, 0, pred_norm, 0]
        actual_price = scaler.inverse_transform([dummy])[0][3]
        forecasts.append(round(actual_price, 2))

        new_row = temp_df.iloc[-1].copy()
        new_row["Close"] = pred_norm

        temp_df = pd.concat(
            [temp_df, pd.DataFrame([new_row])],
            ignore_index=True
        )

    return forecasts


def get_symbol_from_company(company_name):
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {"q": company_name, "quotesCount": 1, "newsCount": 0}
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        if response.status_code != 200 or not response.text.strip():
            return None
        data = response.json()
        if "quotes" in data and len(data["quotes"]) > 0:
            return data["quotes"][0].get("symbol")
    except Exception:
        return None

    return None


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "linear_regression.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    symbol = None
    prices = []
    dates = []
    forecast_5 = []
    forecast_10 = []
    future_dates_5 = []
    future_dates_10 = []
    signal = "HOLD"
    signal_reason = "No strong signal"
    pred_low=[]
    pred_high=[]
    prob_up=[]
    prob_down=[]
    next_day_date = None
    error = None

    if request.method == "POST":
        company_name = request.form.get("symbol", "").strip()

        if not company_name:
            error = "Please enter a company name."
        else:
            symbol = get_symbol_from_company(company_name)

            if symbol is None:
                error = "Company not found. Please try another name."
            else:
                data = yf.download(symbol, period="3mo", interval="1d")

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                if data.empty:
                    error = "No market data available."
                else:
                    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
                    data.dropna(inplace=True)

                    prices = data['Close'].tail(20).tolist()
                    dates = data.index.strftime('%Y-%m-%d')[-20:].tolist()

                    if dates:
                        last_date = pd.to_datetime(dates[-1])
                        future_dates_5 = [(last_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 6)]
                        future_dates_10 = [(last_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 11)]


                    

                    scaled = scaler.transform(data)
                    df = pd.DataFrame(scaled, columns=data.columns)

                    df['MA_5'] = df['Close'].rolling(5).mean()
                    df['MA_10'] = df['Close'].rolling(10).mean()
                    df['Daily_Return'] = df['Close'].pct_change()
                    df['RSI'] = calculate_rsi(df['Close'])
                    df.dropna(inplace=True)

                    if len(df) >= 2:
                        last = df.iloc[-1]
                        prev = df.iloc[-2]

                        if last['RSI'] < 30:
                            signal = "BUY"
                            signal_reason = "RSI indicates oversold condition"
                        elif last['RSI'] > 70:
                            signal = "SELL"
                            signal_reason = "RSI indicates overbought condition"
                        elif prev['MA_5'] < prev['MA_10'] and last['MA_5'] > last['MA_10']:
                            signal = "BUY"
                            signal_reason = "Short-term MA crossed above long-term MA"
                        elif prev['MA_5'] > prev['MA_10'] and last['MA_5'] < last['MA_10']:
                            signal = "SELL"
                            signal_reason = "Short-term MA crossed below long-term MA"

                    model_features = [
                        'Open', 'High', 'Low', 'Close',
                        'Volume', 'MA_5', 'MA_10', 'Daily_Return'
                    ]

                    latest_features = df[model_features].iloc[-1:].values
                    pred_norm = model.predict(latest_features)[0]

                    dummy = [0, 0, 0, pred_norm, 0]
                    actual_price = scaler.inverse_transform([dummy])[0][3]
                    prediction = round(actual_price, 2)
                    # ===================== STEP 1 ADDITIONS =====================

                    # Last actual closing price
                    last_close = data['Close'].iloc[-1]

                    # 🔹 Prediction range (±2%)
                    range_percent = 0.02
                    pred_low = round(prediction * (1 - range_percent), 2)
                    pred_high = round(prediction * (1 + range_percent), 2)

                    # 🔹 Probability logic
                    if prediction > last_close:
                        prob_up = 65
                        prob_down = 35
                    else:
                        prob_up = 35
                        prob_down = 65

                    # 🔹 Next trading day (skip weekends)
                    next_trading_day = (pd.to_datetime(dates[-1]) + BDay(1)).strftime("%d %b %Y")

                    forecast_5 = multi_day_forecast(model, scaler, df, 5)
                    forecast_10 = multi_day_forecast(model, scaler, df, 10)
    
    if dates:
        last_date = pd.to_datetime(dates[-1])

        next_day = last_date + timedelta(days=1)

        # 🚫 Skip weekends (Saturday=5, Sunday=6)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)

        next_day_date = next_day.strftime("%d %b %Y")              

    return render_template(
    "index.html",
    prediction=prediction,
    next_day_date=next_day_date,

    # ✅ NEW STEP-1 VARIABLES
    pred_low=pred_low,
    pred_high=pred_high,
    prob_up=prob_up,
    prob_down=prob_down,

    symbol=symbol,
    prices=prices,
    dates=dates,
    forecast_5=forecast_5,
    forecast_10=forecast_10,
    future_dates_5=future_dates_5,
    future_dates_10=future_dates_10,
    signal=signal,
    signal_reason=signal_reason,
    error=error
)


@app.route("/search")
def search_company():
    query = request.args.get("q", "").strip()
    if not query or len(query) < 2:
        return {"results": []}

    url = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "quotesCount": 5, "newsCount": 0}
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        data = response.json()
        results = []

        for item in data.get("quotes", []):
            name = item.get("shortname") or item.get("longname")
            symbol = item.get("symbol")
            if name and symbol:
                results.append({"name": name, "symbol": symbol})

        return {"results": results}
    except Exception:
        return {"results": []}


if __name__ == "__main__":
    app.run(debug=True)