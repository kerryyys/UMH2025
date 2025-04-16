import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# === Load Model & Scaler ===
model = joblib.load("models/HMM_Model.pkl")
scaler = joblib.load("models/scaler.pkl")

# === Load Data ===
features = [
    "active_addresses", "exchange_inflow", "exchange_outflow",
    "exchange_whale_ratio", "transaction_count", "reserve_usd",
    "SSR_v", "funding_rate", "open_interest"
]

df = pd.read_csv("./data/processed_data/final_merged.csv", parse_dates=["start_time"])
df = df.sort_values("start_time").dropna(subset=features + ["price"]).reset_index(drop=True)
X = scaler.transform(df[features])

# === Train-Test Split ===
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
df_train, df_test = df.iloc[:train_size].copy(), df.iloc[train_size:].copy()

# === Regime Assignment ===
df_train["Regime"] = model.predict(X_train)
df_train["Return"] = df_train["price"].pct_change().fillna(0)

regime_order = df_train.groupby("Regime")["Return"].mean().sort_values().index
regime_map = {int(regime_order[0]): "Bear", int(regime_order[1]): "Neutral", int(regime_order[2]): "Bull"}

# Predict test set regimes using sliding window
test_predictions = []
window_size = 30
for i in range(len(X_test)):
    sequence = np.vstack((X_train, X_test[:i+1]))[-window_size:]
    state = model.predict(sequence)[-1]
    test_predictions.append(state)

df_test["Regime"] = test_predictions
df_test["Market_Regime"] = df_test["Regime"].map(regime_map)

# === Combine & Signal Mapping ===
df = pd.concat([df_train, df_test]).reset_index(drop=True)
df["Market_Regime"] = df["Market_Regime"].fillna("Unknown")

signal_map = {"Bull": 1.0, "Neutral": 0.0, "Bear": -1.0, "Unknown": 0.0}
action_map = {1.0: "🟢 BUY", 0.0: "🟡 HOLD", -1.0: "🔴 SELL"}
df["Signal"] = df["Market_Regime"].map(signal_map)
df["Action"] = df["Signal"].map(action_map)

# === Strategy Backtesting ===
def calculate_strategy_returns(df):
    df = df.copy()
    df["Position"] = df["Signal"].shift()
    df["Return"] = df["price"].pct_change().fillna(0)
    df["Strategy_Return"] = df["Position"] * df["Return"]
    df["Trade_Size"] = df["Signal"].diff().abs()
    df["Fees"] = 0.0006 * df["Trade_Size"]
    df["Slippage"] = 0.0002 * df["Trade_Size"]
    df["Net_Return"] = df["Strategy_Return"] - df["Fees"] - df["Slippage"]
    df["Cumulative_Return"] = (1 + df["Net_Return"]).cumprod()
    df["Trades"] = (df["Trade_Size"] > 0).astype(int)
    return df

df = calculate_strategy_returns(df)

# === Save Results ===
os.makedirs("results", exist_ok=True)
df[["start_time", "price", "Market_Regime", "Action", "Net_Return"]].to_csv(
    "./data/crypto_strategy_output.csv", index=False
)

# === Plot Results ===
def plot_results(df):
    color_map = {"Bull": "green", "Bear": "red", "Neutral": "yellow", "Unknown": "gray"}
    colors = df["Market_Regime"].map(color_map)

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 1, 1)
    plt.scatter(df["start_time"], df["price"], c=colors.values, s=10)
    plt.title("Price with Market Regimes")
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[r], markersize=10)
               for r in color_map]
    plt.legend(handles, color_map.keys())

    plt.subplot(2, 1, 2)
    plt.plot(df["start_time"], df["Cumulative_Return"])
    plt.title("Strategy Cumulative Returns")

    plt.tight_layout()
    plt.savefig("./results/performance_visualization.png")
    plt.show()

plot_results(df)
print("✅ Backtest complete and results saved.")
