import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import joblib

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# === Feature Set Used ===
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

# === Load and Prepare Data ===
df = pd.read_csv("./data/processed_data/final_merged.csv", parse_dates=["start_time"])
df = df.sort_values("start_time").reset_index(drop=True)

df = df.dropna(subset=features + ["price"])
scaler = StandardScaler()
X = scaler.fit_transform(df[features])

# === Train-Test Split ===
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
df_train = df.iloc[:train_size].copy()
df_test = df.iloc[train_size:].copy()

# === Train Gaussian HMM ===
model = GaussianHMM(
    n_components=3,
    covariance_type="diag",
    n_iter=1000,
    tol=1e-4,
    random_state=42
)
model.fit(X_train)

# Save the model
joblib.dump(model, './models/HMM_Model.pkl')

# === Regime Assignment ===
df_train["Regime"] = model.predict(X_train)
df_train["Return"] = df_train["price"].pct_change().fillna(0)

# Map regime to performance
regime_order = df_train.groupby("Regime")["Return"].mean().sort_values().index
regime_map = {
    int(regime_order[0]): "Bear",
    int(regime_order[1]): "Neutral",
    int(regime_order[2]): "Bull"
}

# === Predict Test Set Regimes ===
test_predictions = []
window_size = 30

for i in range(len(X_test)):
    sequence = np.vstack((X_train, X_test[:i+1]))[-window_size:]
    current_state = model.predict(sequence)[-1]
    test_predictions.append(current_state)

df_test["Regime"] = test_predictions
df_test["Market_Regime"] = df_test["Regime"].map(regime_map)

# === Combine Train & Test Sets ===
df = pd.concat([df_train, df_test]).reset_index(drop=True)
df["Market_Regime"] = df["Market_Regime"].fillna("Unknown")

# === Trading Signal Mapping ===
signal_map = {"Bull": 1.0, "Neutral": 0.0, "Bear": -1.0, "Unknown": 0.0}
action_map = {1.0: "🟢 BUY", 0.0: "🟡 HOLD", -1.0: "🔴 SELL"}

df["Signal"] = df["Market_Regime"].map(signal_map)
df["Action"] = df["Signal"].map(action_map)

# === Backtesting Function ===
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

# === Strategy Evaluation ===
def evaluate_strategy(df):
    returns = df["Net_Return"].dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(24 * 365)

    cum = df["Cumulative_Return"]
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd = drawdown.min()

    trade_freq = df["Trades"].mean()

    print(f"\n📈 Strategy Evaluation")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Trade Frequency: {trade_freq:.2%}")

evaluate_strategy(df)

# === Real-time Prediction Helper ===
def predict_next_state():
    current_features = X[-30:]  # last 30 rows
    current_state = model.predict(current_features)[-1]
    next_state = np.argmax(model.transmat_[current_state])
    return regime_map.get(next_state, "Unknown"), model.transmat_[current_state].max()

# === Final Recommendation ===
next_regime, confidence = predict_next_state()
print("\n=== Trading Recommendations ===")
print(f"Current Market Regime: {df['Market_Regime'].iloc[-1]}")
print(f"Recommended Action: {df['Action'].iloc[-1]}")
print(f"\nNext Period Prediction: {next_regime} (Confidence: {confidence:.1%})")
print("\nLatest Signals:")
print(df[["start_time", "price", "Market_Regime", "Action"]].tail(5).to_string(index=False))

# === Visualization ===
def plot_results(df):
    os.makedirs("./results", exist_ok=True)
    plt.figure(figsize=(14, 10))

    color_map = {"Bull": "green", "Bear": "red", "Neutral": "yellow", "Unknown": "gray"}
    colors = df["Market_Regime"].map(color_map)

    plt.subplot(2, 1, 1)
    plt.scatter(df["start_time"], df["price"], c=colors.values, s=10)
    plt.title("Price with Market Regimes")
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=color_map[r], markersize=10)
               for r in color_map]
    plt.legend(handles, color_map.keys())

    plt.subplot(2, 1, 2)
    plt.plot(df["start_time"], df["Cumulative_Return"])
    plt.title("Strategy Cumulative Returns")
    plt.tight_layout()
    plt.savefig("./results/performance_visualization.png")
    plt.show()

plot_results(df)

# === Save Output ===
df[["start_time", "price", "Market_Regime", "Action", "Net_Return"]].to_csv(
    "./data/crypto_strategy_output.csv", 
    index=False
)