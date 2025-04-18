import pandas as pd

# Load historical data
df_hist = pd.read_csv('./data/NLP/processed/reddit_historical_posts_with_sentiment.csv')

# Ensure 'date' is in datetime format
df_hist['date'] = pd.to_datetime(df_hist['date'])

# Load new data with sentiment
df_new = pd.read_csv('./data/NLP/processed/reddit_posts_2023_2025_with_sentiment.csv')

# Ensure 'date' is in datetime format
df_new['date'] = pd.to_datetime(df_new['date'])

# Concatenate the datasets
df_combined = pd.concat([df_hist, df_new], ignore_index=True)

# Save the combined dataset
df_combined.to_csv('./data/NLP/processed/reddit_combined_historical_posts_with_sentiment.csv', index=False)
