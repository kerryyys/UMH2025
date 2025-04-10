import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data (mock data in this case, replace with real data)
df = pd.read_csv("./data/mock_onchain_data.csv", parse_dates=["Timestamp"])
df = df.sort_values("Timestamp")

# Normalize on-chain features (features like Whale_Inflow, Exchange_Outflow, etc.)
features = ["Whale_Inflow", "Exchange_Inflow", "Exchange_Outflow", "Active_Addresses", "Tx_Count"]
scaler = StandardScaler()
X = scaler.fit_transform(df[features])

# Train the Hidden Markov Model (HMM) with 3 hidden states (Bull, Bear, Neutral)
model = GaussianHMM(n_components=3, covariance_type="full", n_iter=3000)
model.fit(X)

# Predict the hidden states (market regimes)
hidden_states = model.predict(X)
df["Regime"] = hidden_states

# Map HMM states to market regimes
# This map will associate the most profitable states with Bull, Neutral, and Bear
# Map HMM states to market regimes
df["Return"] = df["Price"].pct_change().fillna(0)
regime_returns = df.groupby("Regime")["Return"].mean().sort_values()

# Change the mapping to accommodate only 2 states
regime_map = {regime_returns.index[0]: "Bear", 
              regime_returns.index[1]: "Bull"}  # Only 2 states now
df["Market_Regime"] = df["Regime"].map(regime_map)

# Map the market regime to trading signals
# Strategy: Bull = Long, Bear = Short
signal_map = {"Bull": 1, "Bear": -1}  # No Neutral state in this case
df["Signal"] = df["Market_Regime"].map(signal_map)

# Backtest strategy based on trading signals (considering trading fee)
df["Strategy_Return"] = df["Signal"].shift(1) * df["Return"]
df["Net_Strategy_Return"] = df["Strategy_Return"] - 0.0006 * df["Signal"].diff().abs().fillna(0)
df["Cumulative_Return"] = (1 + df["Net_Strategy_Return"]).cumprod()

# Calculate the performance metrics: Sharpe Ratio, Max Drawdown, and Trade Frequency
sharpe_ratio = df["Net_Strategy_Return"].mean() / df["Net_Strategy_Return"].std() * np.sqrt(252)
rolling_max = df["Cumulative_Return"].cummax()
mdd = (df["Cumulative_Return"] / rolling_max - 1).min()
trade_freq = df["Signal"].diff().abs().fillna(0).mean() * 100

# Output performance metrics
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {mdd:.2%}")
print(f"Trade Frequency: {trade_freq:.2f}% per row")

# Save predictions and trade signals to a CSV file for further analysis
df[["Timestamp", "Price", "Market_Regime", "Signal", "Net_Strategy_Return"]].to_csv("crypto_strategy_output.csv", index=False)

# Function to generate trade signals based on the detected regime (Bull, Bear, Neutral)
def generate_trade_signal(regime):
    if regime == 0:  # Bull regime: Buy
        return "buy"
    elif regime == 1:  # Bear regime: Sell or stay out
        return "sell"
    else:  # Neutral regime: Hold or scalp
        return "hold"

# After predicting the regime, generate the trade signal
regime = model.predict(X)  # You can use X_test if testing on unseen data
trade_signal = [generate_trade_signal(r) for r in regime]

# Optional: Plot cumulative returns of the strategy
plt.figure(figsize=(12, 6))
df.plot(x="Timestamp", y=["Cumulative_Return"], title="Backtested Cumulative Return", legend=False)
plt.grid(True)
plt.show()
