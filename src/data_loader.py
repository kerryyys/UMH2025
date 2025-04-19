import numpy as np
import pandas as pd
import joblib

# Simulate some crypto price data
def generate_dummy_data():
    dates = pd.date_range(start="2024-01-01", end="2024-06-01")
    data = {
        "Date": dates,
        "Bitcoin": np.cumsum(np.random.randn(len(dates)) * 50 + 30000),
        "Ethereum": np.cumsum(np.random.randn(len(dates)) * 5 + 2000),
    }
    return pd.DataFrame(data)

# Will replace above function with the API call to fetch real data

# Load the model
def load_model(model_path):
    return joblib.load(model_path)

# import requests
# import pandas as pd
# from requests.auth import HTTPBasicAuth

# API_KEY = "p85PodLHYEP2zIfquwiYUgRejWRl40tnKsVFza6peLfju4eg"
# API_SECRET = "your_api_secret"

# def fetch_real_time_data(symbol="BTCUSDT", interval="1m", limit=100):
#     url = "https://api.cybotrade.rs/market_data/candles"
#     params = {
#         "symbol": symbol,
#         "interval": interval,
#         "limit": limit
#     }
#     response = requests.get(url, params=params, auth=HTTPBasicAuth(API_KEY, API_SECRET))

#     if response.status_code != 200:
#         raise Exception(f"API Error {response.status_code}: {response.text}")
    
#     data = response.json()

#     df = pd.DataFrame(data['candles'], columns=["timestamp", "open", "high", "low", "close", "volume"])
#     df["Date"] = pd.to_datetime(df["timestamp"], unit="ms")
#     df.set_index("Date", inplace=True)
#     df = df.astype(float)

#     return df
