#Sample integrate NLP with HMM

# 1. Load and Prepare the Data
import pandas as pd
import numpy as np

# Load the daily sentiment data
df = pd.read_csv('./data/NLP/processed/reddit_daily_sentiment.csv', parse_dates=['date'])

# Sort by date to ensure chronological order
df.sort_values('date', inplace=True)

# Extract the sentiment scores and reshape for HMM
sentiment_scores = df['avg_sentiment'].values.reshape(-1, 1)

# 2. Train the HMM
from hmmlearn import hmm

# Define the HMM with 3 hidden states
model = hmm.GaussianHMM(n_components=3, covariance_type='diag', n_iter=1000, random_state=42)

# Fit the model to the sentiment scores
model.fit(sentiment_scores)

# 3. Predict Hidden States
hidden_states = model.predict(sentiment_scores)

# Add the hidden states to the DataFrame
df['hidden_state'] = hidden_states

# 4. Visualize the Results
import matplotlib.pyplot as plt

# Define colors for each state
state_colors = ['red', 'green', 'blue']

# Plot the sentiment scores with colors based on hidden states
plt.figure(figsize=(15, 6))
for state in range(model.n_components):
    idx = df['hidden_state'] == state
    plt.plot(df['date'][idx], df['avg_sentiment'][idx], '.', label=f'State {state}', color=state_colors[state])

plt.xlabel('Date')
plt.ylabel('Average Sentiment Score')
plt.title('Daily Average Sentiment with HMM-Inferred States')
plt.legend()
plt.show()

# 📊 Interpreting the Results
# Hidden States: The HMM identifies patterns in the sentiment scores and assigns each day to a hidden state. These states can correspond to underlying sentiment regimes, such as bullish, bearish, or neutral market sentiments.

# State Transitions: The model also learns the probabilities of transitioning from one state to another, which can be insightful for understanding sentiment dynamics over time.

# 🔄 Next Steps
# Model Evaluation: Assess the model's performance by examining the log-likelihood or using cross-validation techniques.

# Number of States: Experiment with different numbers of hidden states to find the model that best captures the sentiment dynamics.

# Feature Expansion: Incorporate additional features, such as trading volume or price volatility, to enrich the model.

# Forecasting: Use the trained HMM to predict future sentiment states and inform trading strategies.