import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

def main():
    # === Data Loading & Cleaning =============================================
    df = pd.read_csv("./data/merged_dataset.csv")
    
    # Handle price data
    price_col = [c for c in df.columns if 'price' in c.lower()][0]
    df = df.rename(columns={price_col: 'price'}).dropna(subset=['price'])
    
    # Convert timestamps with validation
    df['timestamp'] = pd.to_datetime(df['start_time'], unit='ms', errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp')
    
    # Filter flatline data
    price_pct_change = df['price'].pct_change().abs()
    df = df[price_pct_change > 1e-6].copy()

    # === Conservative Feature Engineering ====================================
    df = df.assign(
        log_ret=np.log(df['price']/df['price'].shift()),
        vol=df['price'].pct_change().rolling(24).std(),
        net_flow=(df['exchange_outflow_x'] - df['exchange_inflow_x']).rolling(12).mean(),
        oi_change=df['open_interest'].pct_change(6)
    ).dropna()
    
    features = ['log_ret', 'vol', 'net_flow', 'oi_change']
    df = df[features + ['price', 'timestamp']].dropna()

    # === Stable HMM Configuration ============================================
    scaler = RobustScaler()
    X = scaler.fit_transform(df[features])
    
    model = GaussianHMM(
        n_components=2,  # Simplified to 2 regimes
        covariance_type="diag",
        n_iter=1000,
        tol=1e-3,
        random_state=42
    )
    model.fit(X[:int(len(X)*0.7)])  # Walk-forward training

    # === Robust Regime Detection =============================================
    df['regime'] = model.predict(X)

    # Map numerical regime to labels
    regime_label_map = {0: 'Risk_Off', 1: 'Risk_On'}
    df['regime_label'] = df['regime'].map(regime_label_map)

    # Convert regime labels to numerical codes
    regime_code_map = {'Risk_Off': 0, 'Risk_On': 1}
    df['regime_code'] = df['regime_label'].map(regime_code_map)

    # === Conservative Signal Generation ======================================
    # 1. Regime confirmation with persistence filter
    df['regime_persist_code'] = df['regime_code'].rolling(6).apply(
        lambda x: x[-1] if np.all(x == x[0]) else np.nan,
        raw=True
    ).ffill()

    # Convert back to labels after numerical operations
    df['regime_persist'] = df['regime_persist_code'].map({0: 'Risk_Off', 1: 'Risk_On'})

    # 2. Volatility-adjusted position sizing
    df['position'] = np.where(
        df['regime_persist'] == 'Risk_On',
        0.5 + 0.3*(1 - df['vol'].rank(pct=True)),
        -0.5
    )

    # 3. Cost-aware trading
    df['trades'] = df['position'].diff().abs()
    df['strategy_ret'] = df['position'].shift() * df['log_ret'] - 0.0006*df['trades']
    df['cum_ret'] = np.exp(df['strategy_ret'].cumsum())

    # === Performance Analysis ================================================
    def calculate_metrics(returns):
        sharpe = returns.mean() / returns.std() * np.sqrt(365*24)
        cumulative = np.exp(returns.cumsum())
        drawdown = (cumulative / cumulative.cummax() - 1).min()
        return sharpe, drawdown
    
    sharpe, max_dd = calculate_metrics(df['strategy_ret'].dropna())
    trade_freq = df['trades'].mean()

    print(f"\n=== Final Metrics ===")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Trade Frequency: {trade_freq:.2%}")

    # === Visualization ======================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Price and regime plot
    ax1.plot(df['timestamp'], df['price'], label='Price')
    ax1.scatter(df['timestamp'], df['price'], 
               c=np.where(df['regime_persist']=='Risk_On','g','r'), 
               s=10, alpha=0.3)
    ax1.set_title("Price with Regime Detection")
    
    # Equity curve
    ax2.plot(df['timestamp'], df['cum_ret'], label='Strategy')
    ax2.plot(df['timestamp'], np.exp(df['log_ret'].cumsum()), label='Buy & Hold')
    ax2.set_title("Cumulative Returns")
    ax2.legend()

    plt.show()

if __name__ == "__main__":
    main()