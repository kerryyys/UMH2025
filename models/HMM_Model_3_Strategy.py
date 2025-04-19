import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class RegimeBasedStrategy:
    def __init__(self, regime_data, initial_capital=100000):
        self.data = regime_data.copy()
        self.initial_capital = initial_capital
        self.positions = pd.Series(index=self.data.index, dtype=float)
        self.portfolio_value = pd.Series(index=self.data.index, dtype=float)
        self.trades = []

    def generate_signals(self):
        """Generate trading signals based on regime predictions"""
        self.data['Signal'] = 0

        # Define position sizes based on regime confidence
        regime_position_size = {
            'Strong Bull': 0.8,    # Full long position
            'Weak Bull': 0.5,      # Half long position
            'Neutral': 0.0,        # No position
            'Weak Bear': -0.3,     # Half short position
            'Strong Bear': -0.5    # Full short position
        }

        # Generate initial positions based on regimes
        self.data['Position'] = self.data['Market_Regime'].map(regime_position_size)

        # Generate signals on regime changes
        self.data['Signal'] = self.data['Position'].diff()

        return self.data['Signal']

    def backtest(self):
      """Run backtest of the strategy"""
      self.generate_signals()

      self.data['Returns'] = self.data['price'].pct_change()
      self.data['Strategy_Returns'] = self.data['Position'].shift(1) * self.data['Returns']

      # Apply 6% trading fee on trades
      fee_rate = 0.06
      self.data['Fee'] = 0.0
      self.data.loc[self.data['Signal'] != 0, 'Fee'] = fee_rate
      self.data['Strategy_Returns'] -= self.data['Fee'] * abs(self.data['Signal'])

      self.data['Cumulative_Returns'] = (1 + self.data['Strategy_Returns']).cumprod()
      self.data['Portfolio_Value'] = self.initial_capital * self.data['Cumulative_Returns']

      # Track trades
      trades = self.data[self.data['Signal'] != 0].copy()
      for idx, row in trades.iterrows():
          trade = {
              'Date': idx,
              'Type': 'BUY' if row['Signal'] > 0 else 'SELL',
              'Size': abs(row['Signal']),
              'Price': row['price'],
              'Regime': row['Market_Regime']
          }
          self.trades.append(trade)

      return self.data

    def calculate_performance_metrics(self):
      """Calculate key performance metrics"""
      trading_days_per_year = 252

      # Basic return metrics
      total_return = self.data['Portfolio_Value'].iloc[-1] / self.initial_capital - 1
      daily_returns = self.data['Strategy_Returns']

      # Calculate metrics
      metrics = {
          'Total Return': total_return,
          'Annual Return': (1 + total_return) ** (trading_days_per_year / len(self.data)) - 1,
          'Volatility': daily_returns.std() * np.sqrt(trading_days_per_year),
          'Sharpe Ratio': (daily_returns.mean() / daily_returns.std()) * np.sqrt(trading_days_per_year),
          'Max Drawdown': (self.data['Portfolio_Value'] / self.data['Portfolio_Value'].cummax() - 1).min(),
          'Win Rate': (daily_returns > 0).mean(),
          'Trading Frequency': len(self.trades) / len(self.data),  # trades per day
          'Profit Factor': abs(daily_returns[daily_returns > 0].sum() / daily_returns[daily_returns < 0].sum())
      }

      # Calculate regime-specific metrics
      regime_metrics = {}
      for regime in self.data['Market_Regime'].unique():
          regime_data = self.data[self.data['Market_Regime'] == regime]
          regime_returns = regime_data['Strategy_Returns']

          if len(regime_returns) > 0:
              regime_metrics[regime] = {
                  'Return': regime_returns.mean() * trading_days_per_year,
                  'Volatility': regime_returns.std() * np.sqrt(trading_days_per_year),
                  'Sharpe': (regime_returns.mean() / regime_returns.std()) * np.sqrt(trading_days_per_year) if regime_returns.std() != 0 else 0,
                  'Duration': len(regime_data)
              }

      return metrics, regime_metrics

    def plot_results(self):
        """Plot strategy performance and regime transitions"""
        plt.figure(figsize=(15, 10))

        # Plot portfolio value
        plt.subplot(2, 1, 1)
        self.data['Portfolio_Value'].plot(label='Portfolio Value')
        plt.title('Portfolio Value Over Time')
        plt.grid(True)
        plt.legend()

        # Plot regime distribution and returns
        plt.subplot(2, 1, 2)
        for regime in self.data['Market_Regime'].unique():
            mask = self.data['Market_Regime'] == regime
            plt.scatter(self.data[mask].index,
                      self.data[mask]['Strategy_Returns'],
                      label=regime, alpha=0.5)

        plt.title('Strategy Returns by Regime')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('./results/strategy_performance.png')
        plt.close()

        # Plot regime transition heatmap
        transitions = pd.crosstab(self.data['Market_Regime'],
                                self.data['Market_Regime'].shift(-1))

        plt.figure(figsize=(10, 8))
        sns.heatmap(transitions, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Regime Transition Heatmap')
        plt.tight_layout()
        plt.savefig('./results/regime_transitions_heatmap.png')
        plt.close()

# Load the labeled data
labeled_data = pd.read_csv("./results/hmm_labeled_data.csv", parse_dates=['start_time'])
labeled_data.set_index('start_time', inplace=True)

# Initialize and run the strategy
strategy = RegimeBasedStrategy(labeled_data)
backtest_results = strategy.backtest()
performance_metrics, regime_metrics = strategy.calculate_performance_metrics()

# Print performance metrics
print("\n=== Overall Strategy Performance ===")
for metric, value in performance_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\n=== Performance by Regime ===")
regime_df = pd.DataFrame(regime_metrics).T
print(regime_df)

# Plot results
strategy.plot_results()

# Save detailed results
results_summary = pd.DataFrame({
    'Metric': list(performance_metrics.keys()),
    'Value': list(performance_metrics.values())
})
results_summary.to_csv('./results/strategy_performance_metrics.csv', index=False)

# Save regime-specific metrics
regime_df.to_csv('./results/regime_specific_metrics.csv')

# Generate trade log
trade_log = pd.DataFrame(strategy.trades)
trade_log.to_csv('./results/trade_log.csv', index=False)

# Calculate additional risk metrics
def calculate_risk_metrics(returns):
    """Calculate additional risk metrics for the strategy"""
    daily_returns = returns['Strategy_Returns']

    var_95 = np.percentile(daily_returns, 5)
    cvar_95 = daily_returns[daily_returns <= var_95].mean()

    rolling_vol = daily_returns.rolling(window=21).std() * np.sqrt(252)

    risk_metrics = {
        'Value at Risk (95%)': var_95,
        'Conditional VaR (95%)': cvar_95,
        'Maximum Daily Loss': daily_returns.min(),
        'Maximum Daily Gain': daily_returns.max(),
        'Average Rolling Volatility': rolling_vol.mean(),
    }

    return risk_metrics

risk_metrics = calculate_risk_metrics(backtest_results)
print("\n=== Risk Metrics ===")
for metric, value in risk_metrics.items():
    print(f"{metric}: {value:.4f}")

# Plot rolling performance metrics
plt.figure(figsize=(15, 10))

# Rolling Sharpe Ratio
rolling_returns = backtest_results['Strategy_Returns'].rolling(window=63)
rolling_sharpe = (rolling_returns.mean() / rolling_returns.std()) * np.sqrt(252)

plt.subplot(2, 1, 1)
rolling_sharpe.plot()
plt.title('63-Day Rolling Sharpe Ratio')
plt.grid(True)

# Rolling Drawdown
portfolio_value = backtest_results['Portfolio_Value']
rolling_max = portfolio_value.rolling(window=252, min_periods=1).max()
drawdown = (portfolio_value - rolling_max) / rolling_max

plt.subplot(2, 1, 2)
drawdown.plot()
plt.title('Rolling Drawdown')
plt.grid(True)

plt.tight_layout()
plt.savefig('./results/rolling_metrics.png')
plt.close()