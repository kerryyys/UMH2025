import praw
import pandas as pd
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import os

# Setup
nltk.download("vader_lexicon")
sid = SentimentIntensityAnalyzer()

# Reddit credentials
reddit = praw.Reddit(
    client_id="XBYUpCu5e9O_Mci3EAYEog",
    client_secret="o8-LPf7uKGD_0xiQy3fkBaJrbxe3Vg",
    user_agent="limabijiNLPModel/0.1 by limabiji",
)

subreddits = ["CryptoCurrency", "Bitcoin", "ethereum", "CryptoMarkets", "altcoin"]

# File paths (SEPARATED)
post_sentiment_path = "./data/NLP/processed/reddit_post_sentiments.csv"
daily_sentiment_path = "./data/NLP/processed/reddit_daily_sentiment.csv"
os.makedirs(os.path.dirname(post_sentiment_path), exist_ok=True)

# Load existing data (if exists)
if os.path.exists(post_sentiment_path):
    reddit_df = pd.read_csv(post_sentiment_path)
    reddit_df["date"] = pd.to_datetime(reddit_df["date"]).dt.strftime("%Y-%m-%d")
else:
    reddit_df = pd.DataFrame()

if os.path.exists(daily_sentiment_path):
    sentiment_df = pd.read_csv(daily_sentiment_path)
else:
    sentiment_df = pd.DataFrame(columns=["date", "avg_sentiment"])

# 1. Scrape today’s top posts
def scrape_today_posts(subreddits, limit=100):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    posts = []
    for sub in subreddits:
        for post in reddit.subreddit(sub).top(time_filter="day", limit=limit):
            posts.append({
                "date": today,
                "title": post.title,
                "body": post.selftext,
                "text": f"{post.title} {post.selftext}",
                "subreddit": sub,
                "upvotes": post.score,
                "comments": post.num_comments
            })
    return pd.DataFrame(posts)

today_df = scrape_today_posts(subreddits)

# 2. Apply sentiment analysis
if not today_df.empty:
    print(f"✅ Scraped {len(today_df)} posts today.")
    
    today_df["sentiment_score"] = today_df["text"].apply(lambda text: sid.polarity_scores(text)["compound"])
    print("📊 Sentiment score summary:\n", today_df["sentiment_score"].describe())

    # Update post-level sentiment file
    if os.path.exists(post_sentiment_path):
        old_posts_df = pd.read_csv(post_sentiment_path)
        combined_posts_df = pd.concat([old_posts_df, today_df], ignore_index=True)
        combined_posts_df.drop_duplicates(subset=["title", "body", "date", "subreddit"], inplace=True)
    else:
        combined_posts_df = today_df.copy()

    combined_posts_df.to_csv(post_sentiment_path, index=False)
    print(f"✅ Saved post-level sentiment to {post_sentiment_path}")

    # 3. Calculate and save daily average sentiment
    avg_score = today_df["sentiment_score"].mean()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    sentiment_df = sentiment_df[sentiment_df["date"] != today]  # Remove duplicate if exists
    sentiment_df = pd.concat([sentiment_df, pd.DataFrame([{"date": today, "avg_sentiment": avg_score}])], ignore_index=True)
    sentiment_df.to_csv(daily_sentiment_path, index=False)
    print(f"✅ Saved daily average sentiment to {daily_sentiment_path}: {avg_score:.4f}")

else:
    print("⚠️ No posts found for today.")
