import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import joblib
import os

# Set max CPU usage for parallelism, if needed
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# === Load and Prepare Data ===
features = [
    "active_addresses", "exchange_inflow", "exchange_outflow",
    "exchange_whale_ratio", "transaction_count", "reserve_usd",
    "SSR_v", "funding_rate", "open_interest"
]

df = pd.read_csv("./data/processed_data/final_merged.csv", parse_dates=["start_time"])
df = df.sort_values("start_time").dropna(subset=features + ["price"]).reset_index(drop=True)

# === Feature Scaling ===
scaler = StandardScaler()
X = scaler.fit_transform(df[features])

# === Train Gaussian HMM ===
train_size = int(len(X) * 0.8)
X_train = X[:train_size]

model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000, tol=1e-4, random_state=42)
model.fit(X_train)

# === Save Model & Scaler ===
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/HMM_Model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model and scaler saved to ./models/")