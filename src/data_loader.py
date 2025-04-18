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