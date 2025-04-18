import pandas as pd

# Load the historical Reddit posts with sentiment
df_hist = pd.read_csv("./data/NLP/processed/reddit_combined_historical_posts_with_sentiment.csv", parse_dates=["date"])
df_hist["date"] = pd.to_datetime(df_hist["date"]).dt.date

# Try to load today's posts (if exist)
try:
    df_today = pd.read_csv("./data/NLP/processed/reddit_dailypost_with_sentiment.csv", parse_dates=["date"])
    df_today["date"] = pd.to_datetime(df_today["date"]).dt.date
    df = pd.concat([df_hist, df_today], ignore_index=True)
    print("🆕 Merged today's posts with historical data.")
except FileNotFoundError:
    df = df_hist
    print("📁 No daily file found. Using only historical data.")

# Group by date and calculate average sentiment
print("📈 Grouping by date...")
daily_sentiment = df.groupby("date")["sentiment_score"].mean().reset_index()
daily_sentiment.columns = ["date", "avg_sentiment"]

# Preview result
print("📅 Daily averages:")
print(daily_sentiment.tail())

# Save to file
output_path = "./data/NLP/processed/reddit_daily_sentiment.csv"
daily_sentiment.to_csv(output_path, index=False)
print(f"✅ Saved to {output_path}")
