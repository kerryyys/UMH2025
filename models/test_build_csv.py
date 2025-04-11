import pandas as pd
import numpy as np
import time

def generate_mock_data(years=5):
    total_periods = years * 365 * 48  # 48 intervals per day
    dates = pd.date_range(start="2021-01-01", periods=total_periods, freq="30min")
    
    np.random.seed(int(time.time()))  # 👈 More randomized each time
    time_index = np.linspace(0, 8*np.pi, total_periods)
    
    # Optional: random volatility and base price
    volatility = np.random.uniform(0.0003, 0.0007)
    trend_factor = np.random.uniform(29000, 31000)
    
    price_trend = np.cumsum(
        np.random.normal(0, volatility, total_periods) +
        0.0001 * np.sin(time_index/100) +
        0.0003 * np.random.randn(total_periods)
    ) * 30000 + trend_factor
    
    data = {
        "Timestamp": dates,
        "Whale_Inflow": np.clip(np.sin(time_index)) * 2 + np.random.normal(0, 0.3, total_periods), 
        "Exchange_Inflow": np.abs(3 + np.sin(time_index/24) * 1.5 + np.random.normal(0, 0.5, total_periods)),
        "Exchange_Outflow": np.abs(4 + np.cos(time_index/24) * 2 + np.random.normal(0, 0.6, total_periods)),
        "Active_Addresses": np.random.randint(800, 2500, total_periods),
        "Tx_Count": np.random.randint(5000, 15000, total_periods),
        "Price": price_trend
    }
    
    return pd.DataFrame(data)

# Generate and save data
df = generate_mock_data(3)
df.to_csv("./data/mock_onchain_data.csv", index=False)
