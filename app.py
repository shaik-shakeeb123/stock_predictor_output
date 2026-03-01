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
    "Daily_Return",
    "MA_5",
    "MA_10",
    "Volume"
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

def multi_day_forecast(model, df_scaled, last_close, days=5):
    forecasts = []
    temp_df = df_scaled.copy()
    current_price = last_close

    for _ in range(days):
        X = temp_df[FEATURE_COLUMNS].iloc[-1:].values  # <-- IMPORTANT
        predicted_return = model.predict(X)[0]
        predicted_return = np.clip(predicted_return, -0.10, 0.10)

        next_price = current_price * (1 + predicted_return)
        forecasts.append(round(next_price, 2))

        new_row = temp_df.iloc[-1].copy()
        new_row["Daily_Return"] = predicted_return

        temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)
        current_price = next_price

    return forecasts
def get_currency(symbol):
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return "₹", "INR"
    return "$", "USD"
def get_symbol_from_company(company_name):
    # ✅ If user already typed correct Yahoo symbol
    company_name = company_name.strip().upper()
    if company_name.endswith(".NS") or company_name.endswith(".BO"):
        return company_name

    url = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {"q": company_name, "quotesCount": 5, "newsCount": 0}
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        data = response.json()
        quotes = data.get("quotes", [])

        # 1️⃣ Prefer NSE
        for q in quotes:
            sym = q.get("symbol", "")
            if sym.endswith(".NS"):
                return sym

        # 2️⃣ Then BSE
        for q in quotes:
            sym = q.get("symbol", "")
            if sym.endswith(".BO"):
                return sym

        # 3️⃣ Fallback (US / global)
        if quotes:
            return quotes[0].get("symbol")

    except Exception:
        return None

    return None




# ✅ CORRECT BASE DIRECTORY
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# 🔹 Global variables
model = None
scaler = None
def load_models():
    global model, scaler
    try:
        if model is None:
            model = joblib.load(MODEL_PATH)
        if scaler is None:
            scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}")
# model = joblib.load(MODEL_PATH)
# scaler = joblib.load(SCALER_PATH)


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
    load_models() 
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
    currency_code = None
    currency_symbol = None
    next_day_date = None
    error = None

    if request.method == "POST":
        company_name = request.form.get("symbol", "").strip()

        if not company_name:
            error = "Please enter a company name."
        else:
            symbol = get_symbol_from_company(company_name)
        # 🔹 Detect currency based on symbol
            if symbol is None:
                error = "Company not found. Please try another name."
            else:
                if symbol.endswith(".NS") or symbol.endswith(".BO"):
                    currency_symbol = "₹"
                    currency_code = "INR"
                else:
                    currency_symbol = "$"
                    currency_code = "USD"    

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


                    

                   # 1️⃣ Start with raw data
                    df = data.copy()

                    # 2️⃣ Create technical indicators FIRST
                    df['MA_5'] = df['Close'].rolling(5).mean()
                    df['MA_10'] = df['Close'].rolling(10).mean()
                    df['Daily_Return'] = df['Close'].pct_change()
                    df['RSI'] = calculate_rsi(df['Close'])

                    # 3️⃣ Drop rows with NaN values
                    df.dropna(inplace=True)
                    df_signal = df.copy()   # ✅ ADD THIS HERE

                    # 4️⃣ Scale ONLY trained features
                    df[FEATURE_COLUMNS] = scaler.transform(df[FEATURE_COLUMNS])

                    if len(df) >= 2:
                        last = df_signal.iloc[-1]
                        prev = df_signal.iloc[-2]

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

                    # 🔹 Take last row features (already scaled)
                    latest_features = df[FEATURE_COLUMNS].iloc[-1:]

                    # 🔹 Predict next-day return
                    # df is ALREADY scaled
                    latest_features = df[FEATURE_COLUMNS].iloc[-1:]
                    predicted_return = model.predict(latest_features.values)[0]

                    # 🔒 HARD SAFETY CLIP (±10%)
                    predicted_return = np.clip(predicted_return, -0.10, 0.10)

                    # 🔹 Convert return → price
                    last_close = data["Close"].iloc[-1]
                    prediction = round(last_close * (1 + predicted_return), 2)
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

                    forecast_5 = multi_day_forecast(model, df, last_close, 5)
                    forecast_10 = multi_day_forecast(model, df, last_close, 10)
    
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
    currency_symbol=currency_symbol,
    currency_code=currency_code,

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)