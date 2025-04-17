import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import os

# Download VADER lexicon
nltk.download("vader_lexicon")

# Initialize Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# Load Reddit posts dataset
posts_df = pd.read_csv("./data/NLP/reddit_posts.csv")

# Combine title and body for sentiment analysis
posts_df["content"] = posts_df["title"].fillna("") + " " + posts_df["body"].fillna("")

# Apply sentiment analysis
posts_df["sentiment_score"] = posts_df["content"].apply(lambda text: sid.polarity_scores(str(text))["compound"])

# Save the updated dataset
posts_df.to_csv("./data/NLP/processed/reddit_posts_with_sentiment.csv", index=False)

print("✅ Sentiment analysis completed and saved to 'reddit_posts_with_sentiment.csv'")
