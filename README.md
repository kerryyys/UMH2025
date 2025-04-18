# UMH2025 🚀  
Hi there! We are **Team Lima Biji**, participating in the **UMHackathon 2025** under:

📊 **Domain 2 - Quantitative Trading**

📑 **Slides link**: [View Our Deck](https://www.canva.com/design/DAGkWFnoy34/IumXz3cmGOLTeMXOjEOGaw/edit?utm_content=DAGkWFnoy34&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

---

## 🧠 Introduction

In the fast-paced world of cryptocurrency, market sentiment and behavioral cues from key players like whales are crucial signals for strategy formulation. Our project bridges the power of **Hidden Markov Models (HMM)** and **Natural Language Processing (NLP)** to detect market regimes and predict optimal trading strategies.

---

## 🎯 Project Goal

Our aim is to build an **alpha-generating crypto trading system** that:
- Classifies market conditions (bull, bear, neutral) using **on-chain features**.
- Integrates **Reddit sentiment analysis** to understand public reaction to market shifts.
- Suggests **BUY/SELL/HOLD strategies** based on regime and sentiment.
- Emphasizes **transparency** using **feature attribution** and **data visualization**.

---

## 🧪 Hypotheses & Metrics (To be completed)
> ✍️ Add your hypotheses here, e.g.:
- H1: Reddit sentiment shifts correlate strongly with whale behavior in bearish markets.
- H2: Combining sentiment with on-chain activity outperforms pure technical analysis.

> ✍️ Add metrics here, e.g.:
- Sharpe Ratio
- Strategy Win Rate
- Precision/Recall for sentiment classification
- Regime classification accuracy

---

## 🛠️ Setup & Installation

```
# Clone the repo
git clone https://github.com/kerryyys/UMH2025.git
cd UMH2025

# Create virtual environment
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the pipeline (example)
python models/new_model.py
```
> Note: For NLP sentiment analysis, make sure your Reddit credentials are set up via .env or passed into the praw module.

---

## 🌟 Innovation Highlights
💬 NLP for Whale Behavior Tracking
Scrapes Reddit data to detect how the crowd reacts to sudden market flows.

Uses **VADER Sentiment Analysis** to extract daily sentiment scores.

Integrates whale movement (inflows/outflows) with public opinion.

---

## 🔎 Feature Attribution for Transparency
Uses decision tree explanations & correlation matrices to expose how features drive decisions.

Explains why the model triggers certain BUY/SELL calls.

---

## 📊 Visual Insights
Heatmaps, clustering charts, and sentiment trendlines to explain strategies visually.

---

Model state visualizations (e.g., HMM transition maps, emission probabilities).

---

## 🧪 Feature Engineering
Feature Type | Examples | Description
On-Chain | exchange_inflow, whale_spikes | Real-time behaviors of smart money
Sentiment | avg_sentiment_score, post_volume | Reddit NLP signals aggregated daily
Technical | price, returns, volume | Classical indicators
Engineered | log_return, whale_sentiment_diff | Combined sentiment-behavioral signals

---

## 🧱 Model Architecture (Coming Up Next 🧩)
We combine:

A **Gaussian HMM** for market regime detection.

An **NLP pipeline** to extract public sentiment.

A **strategy recommendation engine** based on regime + sentiment context.

📌 More details and diagrams will be added below 👇

---
## 🗂️ File Structure
```
UMH2025/
├── data/
│   ├── cleaned/
│   │   └── btc 2023-2024/        # Cleaned crypto datasets
│   ├── NLP/
│   │   ├── processed/            # Processed NLP sentiment data
│   │   ├── raw_unused_data/      # Raw unused Reddit post data
│   │   └── reddit_posts.csv      # Collected Reddit post data
│   ├── processed_data/           # Final processed datasets for modeling
│   └── raw_data/
│       ├── crypto_kmeans_clustering_output.csv
│       └── crypto_strategy_output.csv

├── models/
│   ├── NLP/                      # (Reserved for NLP models/scripts)
│   ├── pkl/                      # Serialized model files
│   ├── generate_backtest.py     # Script to simulate strategy based on HMM
│   ├── GMM_Model.py             # Gaussian Mixture Model implementation
│   ├── HMM_Model_2.py           # Another variant of the HMM pipeline
│   ├── HMM_Model.pkl            # Trained HMM model (pickle)
│   ├── HMM_Model.py             # Main HMM modeling script
│   ├── kmeans_model.py          # KMeans clustering model
│   ├── new_model.py             # Combined pipeline (likely used in final integration)
│   ├── test_build_csv.py        # Script for testing CSV outputs
│   └── train_model              # Model training entry point (could be a directory or file)

├── results/                     # Folder to store visualizations or model outputs

├── src/                         # (Reserved for source code files)

├── .gitattributes
├── .gitignore
├── README.md                    # Project documentation (You're working on this!)
├── requirements.txt             # Python dependencies
└── run.bat                      # Batch script to execute project pipeline
```

---

## 📚 Citations
**HMM On-Chain Data**: Credit to [CoinGlass](https://www.coinglass.com/), [CryptoQuant](https://cryptoquant.com/), [Glassnode](https://glassnode.com/)

**Reddit Sentiment Data**: Credit to [Reddit](https://www.reddit.com/) via praw API
