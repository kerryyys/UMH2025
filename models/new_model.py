import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
import os

# Load your data
df = pd.read_csv("./data/processed_data/final_merged.csv")

# Convert timestamp to datetime
df['start_time'] = pd.to_datetime(df['start_time'])

# Feature Engineering: Creating transformed features
df['log_ret'] = np.log(df['price'] / df['price'].shift())
df['vol'] = df['price'].pct_change().rolling(24).std()
df['net_flow'] = (df['exchange_outflow'] - df['exchange_inflow']).rolling(12).mean()
df['oi_change'] = df['open_interest'].pct_change(6)

# Drop NaNs after feature transformations
df.dropna(inplace=True)

# Define the feature set (including transformed features)
features = [
    'active_addresses', 'exchange_inflow', 'exchange_outflow', 'price', 'transaction_count', 
    'exchange_whale_ratio', 'reserve_usd', 'SSR_v', 'open_interest', 'high_open_interest', 
    'low_open_interest', 'open_value_of_open_interest', 'funding_rate', 'high_funding_rate', 
    'low_funding_rate', 'open_funding_rate', 'long_account', 'long_short_ratio', 'short_account', 
    'log_ret', 'vol', 'net_flow', 'oi_change'
]

# Select only numeric columns
numeric_cols = df[features].select_dtypes(include=[np.number])

# Scale the features using RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(numeric_cols)

# Split the data into training and testing (80/20 split)
X_train, X_test = train_test_split(X_scaled, test_size=0.2, shuffle=False)

# Fit the HMM model
n_components = 5  # Number of hidden states in the HMM (can be adjusted)
hmm = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=1000)
hmm.fit(X_train)

# Predict hidden states on the test data
hidden_states = hmm.predict(X_test)

# Visualize the predicted hidden states over time
plt.figure(figsize=(10, 6))
plt.plot(hidden_states, label='Predicted Hidden States')
plt.title('Hidden States Prediction Over Time')
plt.xlabel('Time')
plt.ylabel('Hidden State')
plt.legend()
plt.show()

# Save the trained model and predicted states for later use
joblib.dump(hmm, "hmm_model.pkl")
np.savetxt("predicted_hidden_states.csv", hidden_states, delimiter=",")

# Now, let's calculate the performance metrics
# Assuming you already have the 'price' and 'timestamp' in the df

# Calculate daily returns
df['daily_returns'] = df['price'].pct_change()

# Sharpe Ratio (assuming risk-free rate is 0 for simplicity)
mean_return = df['daily_returns'].mean()
std_return = df['daily_returns'].std()
sharpe_ratio = mean_return / std_return

# Calculate Trade Frequency (number of state transitions)
state_transitions = np.diff(hidden_states)
trade_frequency = np.count_nonzero(state_transitions)  # Count of state transitions

# Convert trade frequency to percentage
trade_frequency_percentage = (trade_frequency / len(hidden_states)) * 100

# Calculate Maximum Drawdown
cumulative_returns = (1 + df['daily_returns']).cumprod()
peak_value = cumulative_returns.cummax()
drawdown = (cumulative_returns - peak_value) / peak_value
max_drawdown = drawdown.min()

# Output the performance metrics
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Trade Frequency Percentage: {trade_frequency_percentage}")
print(f"Maximum Drawdown: {max_drawdown:.4f}")

# Plot the drawdown for visualization
plt.figure(figsize=(10, 6))
plt.plot(drawdown)
plt.title("Maximum Drawdown")
plt.xlabel("Time")
plt.ylabel("Drawdown")
plt.show()

# Save the model and states to files for future use
joblib.dump(hmm, "./models/hmm_model.pkl")
df_states = pd.DataFrame({
    'timestamp': df['start_time'][len(df)-len(hidden_states):],
    'predicted_hidden_states': hidden_states
})


# Define the directory where you want to save the predictions
predictions_dir = "./predictions"

# Check if the directory exists, and create it if not
if not os.path.exists(predictions_dir):
    os.makedirs(predictions_dir)

# Save the predicted hidden states to CSV
df_states.to_csv(os.path.join(predictions_dir, "predicted_hidden_states.csv"), index=False)
