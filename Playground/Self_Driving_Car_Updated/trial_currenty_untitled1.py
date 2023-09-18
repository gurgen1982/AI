# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 22:37:39 2023

@author: user
"""

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Data Collection
def fetch_crypto_data(symbol, interval, start_date, end_date):
    base_url = "https://api.binance.com/api/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(pd.Timestamp(start_date).timestamp() * 1000),
        "endTime": int(pd.Timestamp(end_date).timestamp() * 1000),
        "limit": 1000
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=["timestamp", "Open", "High", "Low", "Close", "Volume", "close_time",
                                     "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
                                     "taker_buy_quote_asset_volume", "ignore"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df["Close"] = df["Close"].astype(float)
    return df

# Replace 'BTCUSDT' with the cryptocurrency pair of your choice (e.g., 'ETHUSDT', 'BTCBUSD', etc.)
symbol = "BTCUSDT"
interval = "1d"
start_date = "2020-01-01"
end_date = "2021-01-01"
crypto_data = fetch_crypto_data(symbol, interval, start_date, end_date)

# Step 2: Feature Engineering
def create_features(data):
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["SMA_200"] = data["Close"].rolling(window=200).mean()
    data["Daily_Return"] = data["Close"].pct_change()
    data["Price_Up"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)
    return data.dropna()

crypto_data = create_features(crypto_data)

# Step 3: Data Preprocessing
X = crypto_data[["SMA_50", "SMA_200", "Daily_Return"]]
y = crypto_data["Price_Up"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Step 4: Model Training (Using XGBoost)
model = xgb.XGBClassifier(random_state=40)
model.fit(X_train, y_train)

# Step 5: Prediction
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
