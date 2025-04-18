import praw
import pandas as pd
from datetime import datetime, timezone

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id="XBYUpCu5e9O_Mci3EAYEog",
    client_secret="o8-LPf7uKGD_0xiQy3fkBaJrbxe3Vg",
    user_agent="limabijiNLPModel/0.1 by limabiji",
)

# Define the subreddit and the start date
subreddit = reddit.subreddit('BitcoinMarkets')
start_date = datetime(2023, 4, 1, tzinfo=timezone.utc)

# List to hold post data
posts_data = []

# Fetch posts
for submission in subreddit.new(limit=None):
    submission_date = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
    if submission_date >= start_date:
        posts_data.append({
            'title': submission.title,
            'score': submission.score,
            'url': submission.url,
            'num_comments': submission.num_comments,
            'body': submission.selftext,
            'date': submission_date.strftime('%Y-%m-%d %H:%M:%S')
        })

# Convert to DataFrame
df_posts = pd.DataFrame(posts_data)

# Save to CSV
df_posts.to_csv('./data/NLP/processed/reddit_posts_2023_2025.csv', index=False)
