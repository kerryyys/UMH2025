import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import ta
import joblib
import os

# Load and sort
df = pd.read_csv("./data/processed_data/final_merged.csv", parse_dates=["start_time"])
df = df.sort_values("start_time").reset_index(drop=True)

# Prevent division by zero
df["price"].replace(0, np.nan, inplace=True)

# Basic features
df["return"] = np.log(df["price"] / df["price"].shift(1)).clip(-0.5, 0.5)
df["volatility"] = df["return"].rolling(10).std()
df["momentum"] = df["price"] - df["price"].shift(10)

# Synthetic high/low
df["high"] = df["price"] * 1.005
df["low"] = df["price"] * 0.995

# Technical Indicators
df["rsi"] = ta.momentum.RSIIndicator(df["price"]).rsi()
df["macd_diff"] = ta.trend.MACD(df["price"]).macd_diff()
df["ema"] = ta.trend.EMAIndicator(df["price"], window=10).ema_indicator()
df["cci"] = ta.trend.CCIIndicator(df["high"], df["low"], df["price"]).cci()
df["stoch_k"] = ta.momentum.StochasticOscillator(df["high"], df["low"], df["price"]).stoch()
df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["price"]).average_true_range()

# Lag features
lags = ["return", "volatility", "momentum", "rsi", "macd_diff", "ema"]
for col in lags:
    df[f"{col}_lag1"] = df[col].shift(1)

# Drop NaNs
df = df.dropna()

# Features for GMM
features = [
    "return", "volatility", "momentum", "rsi", "macd_diff", "ema",
    "cci", "stoch_k", "atr"
] + [f"{col}_lag1" for col in lags]

scaler = StandardScaler()
X = scaler.fit_transform(df[features])

# Gaussian Mixture Model
gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=42)
df["Regime"] = gmm.fit_predict(X)

# Reassign regimes by average return
regime_avg = df.groupby("Regime")["return"].mean().sort_values(ascending=False)
regime_map = {old: new for new, old in enumerate(regime_avg.index)}
df["Market_Regime"] = df["Regime"].map(regime_map)
reverse_map = {v: k for k, v in regime_map.items()}

# Map numeric regime values to corresponding labels
label_map = {0: "Bull", 1: "Neutral", 2: "Bear"}
df["Market_Regime"] = df["Market_Regime"].map(label_map)

# Add probabilities
probs = gmm.predict_proba(X)
df["Bull_Prob"] = probs[:, reverse_map[0]]
df["Bear_Prob"] = probs[:, reverse_map[3]]

# Action logic
def get_action(row):
    if row["Bull_Prob"] > 0.65:
        return "🟢 BUY"
    elif row["Bear_Prob"] > 0.65:
        return "🔴 SELL"
    else:
        return "🟡 HOLD"

df["Action"] = df.apply(get_action, axis=1)
df["Signal"] = df["Action"].map({"🟢 BUY": 1, "🔴 SELL": -1, "🟡 HOLD": 0})

# Volatility filter: avoid trading in high vol
vol_thresh = df["volatility"].quantile(0.95)
df.loc[df["volatility"] > vol_thresh, "Signal"] = 0

# Smooth signals with rolling median to reduce noise
df["Smoothed_Signal"] = df["Signal"].rolling(3, center=True).median().fillna(0)

# Apply strategy returns
df["Strategy_Return"] = df["Smoothed_Signal"].shift(1) * df["return"]

# Simulated stop-loss: cut loss > 5%
df.loc[df["Strategy_Return"] < -0.05, "Strategy_Return"] = -0.05

# Cumulative returns
df["Cumulative_Strategy"] = df["Strategy_Return"].cumsum().apply(np.exp)
df["Cumulative_Market"] = df["return"].cumsum().apply(np.exp)

# === Confidence Estimation Function ===
def predict_next_regime_confidence():
    latest_features = scaler.transform(df[features].iloc[[-1]])
    predicted_probs = gmm.predict_proba(latest_features)[0]
    predicted_regime = np.argmax(predicted_probs)
    confidence = predicted_probs[predicted_regime]
    
    label = label_map.get(regime_map.get(predicted_regime, -1), "Unknown")
    return label, confidence

# Evaluation
sharpe = df["Strategy_Return"].mean() / df["Strategy_Return"].std() * np.sqrt(252)
max_dd = (df["Cumulative_Strategy"] / df["Cumulative_Strategy"].cummax() - 1).min()
trade_freq = df["Smoothed_Signal"].ne(0).mean() * 100

# Output
print("📊 Strategy Evaluation:")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd * 100:.2f}%")
print(f"Trade Frequency: {trade_freq:.2f}%\n")

# Real-time
latest = df.iloc[-1]
print("📍 Real-Time Regime Forecast")
print(f"Current Market Regime: {latest['Market_Regime']}")
print(f"Recommended Action: {latest['Action']}\n")

# Predict next regime and confidence
next_regime, confidence = predict_next_regime_confidence()
print(f"🧠 Predicted Next Regime: {next_regime} (Confidence: {confidence:.2%})")

# Recent actions
print("Recent Signals:")
print(df[["start_time", "price", "Market_Regime", "Action"]].tail())

# Plot performance
fig = df[["Cumulative_Market", "Cumulative_Strategy"]].plot(
    title="📈 Strategy vs Market Performance", figsize=(10, 5), grid=True
)

# Save plot
plot_dir = './results'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
fig.figure.savefig(f"{plot_dir}/gmm_performance.png")

# Save the trained model
model_dir = './models/pkl'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
joblib.dump(gmm, f"{model_dir}/gmm_model.pkl")

print("📈 Plot and model saved successfully!")
