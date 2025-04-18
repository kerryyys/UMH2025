import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Load new posts
df_new = pd.read_csv('./data/NLP/raw_unused_data/reddit_posts_2023_2025.csv')

# Ensure 'date' is in datetime format
df_new['date'] = pd.to_datetime(df_new['date'])

# Compute sentiment scores
df_new['sentiment_score'] = df_new['body'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

# Save to a new CSV
df_new.to_csv('./data/NLP/processed/reddit_posts_2023_2025_with_sentiment.csv', index=False)
