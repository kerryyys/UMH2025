# UMH2025 🚀  
Hi there! We are **Team Lima Biji**, participating in the **UMHackathon 2025** under:

📊 **Domain 2 - Quantitative Trading**

📑 **Slides link**: [View Our Deck](https://www.canva.com/design/DAGkWFnoy34/IumXz3cmGOLTeMXOjEOGaw/edit?utm_content=DAGkWFnoy34&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
Our project integrates financial market data and online user sentiment to enhance crypto market regime detection using Hidden Markov Models (HMM), with the final goal of recommending BUY/SELL/HOLD strategies based on both on-chain data and Reddit discussion patterns.

---

## 🧠 Introduction

Cryptocurrency markets are volatile and sentiment-driven. While traditional models rely purely on numerical indicators, our project attempts to answer:

> "Can combining on-chain whale behavior and Reddit user sentiment create more explainable, adaptive, and realistic trading strategies?"

We propose an explainable ML-driven trading assistant that identifies market regimes and gives contextual investment suggestions supported by public discussions.

---

## 🎯 Project Goal

Our aim is to build an **alpha-generating crypto trading system** that:
- Detect **market regimes** using unsupervised learning (HMM).
- Integrate **Reddit sentiment** to capture behavioral shifts.
- Recommend trading actions (BUY/SELL/HOLD) along with **justifications** derived from sentiment trends.

---

## 🧪 Hypotheses & Metrics
> ✍️ Hypothesis:
- H1: Technical Indicators Improve Regime Detection
- H2: XGBoost Feature Selection Predicts Extreme Price Moves
- H3: Whale-Driven Features Define Market Regimes

> ✍️ Metrics:
- Sharpe Ratio
- Max Drawdown
- Trade Frequency
- Strategy Win Rate

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

## 🧱 Model Architecture
We combine:

A **Gaussian HMM** for market regime detection.
An **NLP pipeline** to extract public sentiment.
A **strategy recommendation engine** based on regime + sentiment context.
![image](https://github.com/user-attachments/assets/cdf75edd-bf0f-42df-869d-2d38e56d9cfc)

### 🧠 Model Architecture (Conceptual View):
            ┌────────────────────┐
            │ On-chain Features │◄─── CryptoQuant / Glassnode / Coinglass
            └────────────────────┘
                       │
                       ▼
            ┌────────────────────┐
            │ Reddit Sentiment  │◄─── NLP pipeline from r/CryptoCurrency, r/BitcoinMarkets
            └────────────────────┘
                       │
                       ▼
            ┌────────────────────┐
            │ Feature Merger     │
            └────────────────────┘
                       │
                       ▼
            ┌────────────────────┐
            │ Gaussian HMM       │
            └────────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ Strategy Decision Engine    │ ──> 🔴 SELL / 🟡 HOLD / 🟢 BUY
         └─────────────────────────────┘

### 🧱 Class Diagram (Simplified Structure)
+---------------------+
| RedditSentiment     |
+---------------------+
| + fetch_posts()     |
| + analyze()         |
| + save_results()    |
+---------------------+

+---------------------+
| FeatureEngineer     |
+---------------------+
| + merge_sources()   |
| + clean_features()  |
+---------------------+

+---------------------+
| HMMTrader           |
+---------------------+
| + train_model()     |
| + predict_regime()  |
| + evaluate()        |
| + generate_signals()|
+---------------------+

+---------------------+
| Visualizer          |
+---------------------+
| + plot_regimes()    |
| + save_backtest()   |
+---------------------+

---
## 🗂️ File Structure
```
UMH2025/
├── archive/                     # Archived or deprecated files

├── data/
│   ├── cleaned/                 # Cleaned datasets (e.g., cleaned/btc 2023-2024/)
│   ├── NLP/
│   │   ├── processed/           # Processed NLP sentiment data
│   │   ├── raw_unused_data/     # Raw unused Reddit post data
│   │   └── reddit_posts.csv     # Collected Reddit post data
│   ├── processed_data/          # Final processed datasets for modeling
│   └── raw_data/
│       ├── crypto_kmeans_clustering_output.csv
│       └── crypto_strategy_output.csv

├── models/                      # Currently unused – reserved for model scripts or checkpoints

├── results/                     # Folder to store visualizations or model outputs

├── src/
│   ├── 0_config/                # Configuration files and constants
│   ├── 1_fetch_data/            # Scripts to fetch or collect raw data
│   ├── 2_merge_data/            # Scripts to merge and align multiple data sources
│   ├── 3_clean_data/            # Scripts to clean and preprocess datasets
│   ├── 4_backtesting/           # Backtesting strategies and evaluation logic
│   ├── NLP/                     # NLP-specific analysis, sentiment scoring, etc.
│   ├── _pycache_/             
│   ├── assets/                  # Static files for Dash app styling
│   │   └── custom.css
│   └── dash/                    # Dash app components
│       ├── app.py               # Main entry point for the Dash dashboard
│       ├── callbacks.py         # Callback functions for interactivity
│       ├── data_loader.py       # Loads and prepares data for visualization
│       └── layout.py            # Dash app layout and structure

├── .gitattributes
├── .gitignore
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── run.bat                      # Batch script to execute project pipeline
├── setup_env.bat                # Batch script to set up the environment
└── ohlcv.csv                    # OHLCV (Open, High, Low, Close, Volume) data


```

---

## 📚 Citations
**HMM On-Chain Data**: Credit to [CoinGlass](https://www.coinglass.com/), [CryptoQuant](https://cryptoquant.com/), [Glassnode](https://glassnode.com/)

**Reddit Sentiment Data**: Credit to [Reddit](https://www.reddit.com/) via praw API
