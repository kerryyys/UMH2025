import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

# === Setup ===
os.makedirs("./models", exist_ok=True)
os.makedirs("./results", exist_ok=True)

# === Load Data ===
df = pd.read_csv("./data/processed_data/final_merged.csv", parse_dates=["start_time"])
features = [
    "active_addresses",
    "exchange_inflow",
    "exchange_outflow",
    "exchange_whale_ratio",
    "transaction_count",
    "reserve_usd",
    "SSR_v",
    "funding_rate",
    "open_interest"
]

df = df.sort_values("start_time").dropna(subset=features + ["price"]).reset_index(drop=True)

# === Feature Scaling ===
scaler = StandardScaler()
X = scaler.fit_transform(df[features])

# === Train-Test Split ===
train_size = int(0.8 * len(df))
X_train, X_test = X[:train_size], X[train_size:]
df_train, df_test = df.iloc[:train_size].copy(), df.iloc[train_size:].copy()

# === Train KMeans ===
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_train["Regime"] = kmeans.fit_predict(X_train)
df_train["Return"] = df_train["price"].pct_change().fillna(0)

# Save model
dump(kmeans, "./models/pkl/KMeans_Model.pkl")

# === Regime Mapping Based on Performance ===
regime_order = df_train.groupby("Regime")["Return"].mean().sort_values().index
regime_map = {
    int(regime_order[0]): "Bear",
    int(regime_order[1]): "Neutral",
    int(regime_order[2]): "Bull"
}

# === Predict Test Regimes ===
df_test["Regime"] = kmeans.predict(X_test)
df_test["Return"] = df_test["price"].pct_change().fillna(0)
df_test["Market_Regime"] = df_test["Regime"].map(regime_map)

# === Combine Sets ===
df_all = pd.concat([df_train, df_test]).reset_index(drop=True)
df_all["Market_Regime"] = df_all["Regime"].map(regime_map)

# === Generate Signals & Actions ===
signal_map = {"Bull": 1.0, "Neutral": 0.0, "Bear": -1.0}
action_map = {1.0: "🟢 BUY", 0.0: "🟡 HOLD", -1.0: "🔴 SELL"}

df_all["Signal"] = df_all["Market_Regime"].map(signal_map)
df_all["Action"] = df_all["Signal"].map(action_map)

# === Backtest Strategy ===
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

df_all = calculate_strategy_returns(df_all)

# === Evaluation ===
def evaluate_strategy(df):
    returns = df["Net_Return"].dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(24 * 365)
    drawdown = (df["Cumulative_Return"] - df["Cumulative_Return"].cummax()) / df["Cumulative_Return"].cummax()
    max_dd = drawdown.min()
    trade_freq = df["Trades"].mean()

    print("\n📊 Strategy Evaluation:")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Trade Frequency: {trade_freq:.2%}")

evaluate_strategy(df_all)

# === Plot Results ===
def plot_results(df):
    color_map = {"Bull": "green", "Bear": "red", "Neutral": "yellow"}
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 1, 1)
    plt.scatter(df["start_time"], df["price"], c=df["Market_Regime"].map(color_map), s=10)
    plt.title("Market Regime Detection via K-Means")
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', label=label,
                   markerfacecolor=color, markersize=10)
        for label, color in color_map.items()
    ])

    plt.subplot(2, 1, 2)
    plt.plot(df["start_time"], df["Cumulative_Return"])
    plt.title("Strategy Cumulative Returns")
    plt.tight_layout()
    plt.savefig("./results/kmeans_clustering_performance.png")
    plt.show()

plot_results(df_all)

# === Real-time Regime Forecast ===
def predict_latest_regime():
    latest_features = X[-1].reshape(1, -1)
    cluster = kmeans.predict(latest_features)[0]
    return regime_map.get(cluster, "Unknown")

latest_regime = predict_latest_regime()

print("\n📍 Real-Time Regime Forecast")
print(f"Current Market Regime: {latest_regime}")
print(f"Recommended Action: {df_all['Action'].iloc[-1]}")
print("\nRecent Signals:")
print(df_all[["start_time", "price", "Market_Regime", "Action"]].tail(5).to_string(index=False))

# === Save Output ===
df_all[["start_time", "price", "Market_Regime", "Action", "Net_Return"]].to_csv(
    "./data/crypto_kmeans_clustering_output.csv", index=False)
