import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.metrics import silhouette_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# === Feature Set Used ===
features = [
    "gn_distribution_balance_wbtc",
    "gn_addresses_supply_balance_more_100k",
    "gn_distribution_balance_grayscale_trust",
    "cq_reserve_usd",
    "gn_supply_active_3y_5y",
    "gn_derivatives_futures_annualized_basis_3m",
    "gn_supply_active_more_2y_percent",
    "gn_supply_active_more_1y_percent",
    "gn_indicators_hodler_net_position_change",
    "gn_market_realized_volatility_3_months",
    "cq_reserve",
    "gn_market_realized_volatility_6_months",
    "gn_blockchain_utxo_loss_count",
    "gn_supply_active_6m_12m",
    "gn_derivatives_futures_open_interest_crypto_margin_sum",
    "gn_entities_supply_balance_10_100",
    "gn_addresses_min_1_count",
    "gn_supply_illiquid_sum",
    "cq_open_interest",
    "gn_derivatives_futures_estimated_leverage_ratio"
]
# === Load and Prepare Data ===
print("Loading and preparing data...")
df = pd.read_csv("./data/cleaned/btc/2023-2024/btc_features_output.csv", parse_dates=["start_time"])
df['start_time'] = pd.to_datetime(df['start_time'], unit='ms')
df = df.sort_values("start_time").reset_index(drop=True)
df.rename(columns={'unified_close': 'price'}, inplace=True)

for col in df.columns:
    print(col)

# Check for missing values and handle them
print(f"Original data shape: {df.shape}")
missing_pct = df[features + ["price"]].isna().mean() * 100
print(f"Missing values percentage:\n{missing_pct}")

# Drop rows with missing values in the critical features
df = df.dropna(subset=features + ["price"])
print(f"Clean data shape after removing NAs: {df.shape}")

# === Feature Engineering ===
# Add price momentum features
df['price_1d_change'] = df['price'].pct_change(1)
df['price_7d_change'] = df['price'].pct_change(7)
df['price_volatility'] = df['price'].pct_change().rolling(7).std().fillna(0)

# Add volume-related features
if 'volume' in df.columns:
    df['volume_change'] = df['volume'].pct_change().fillna(0)
    df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(7).mean().fillna(df['volume'])
    features.extend(['volume_change', 'volume_ma_ratio'])

# Add price-related features to the feature set
features.extend(['price_1d_change', 'price_7d_change', 'price_volatility'])

# Remove rows after feature engineering that might have NaNs
df = df.dropna(subset=features).reset_index(drop=True)

# === Feature Selection and Scaling ===
print("Scaling features...")
scaler = StandardScaler()
X = scaler.fit_transform(df[features])

# === Train-Test Split ===
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
df_train = df.iloc[:train_size].copy()
df_test = df.iloc[train_size:].copy()

# === BIC Score for Model Selection ===
def compute_bic_score(X, model):
    """Compute the BIC score for the fitted model."""
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_components = model.n_components
    n_parameters = n_components * n_features + n_components * (n_components - 1)  # mean + transition matrix

    log_likelihood = model.score(X)
    bic = -2 * log_likelihood + n_parameters * np.log(n_samples)
    return bic

# === Find Optimal Number of Components ===
print("Finding optimal number of components...")
n_components_range = range(2, 10)
models = []
bic_scores = []
aic_scores = []

for n_components in n_components_range:
    model = GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        n_iter=1000,
        tol=1e-4,
        random_state=42
    )
    model.fit(X_train)
    models.append(model)

    # Compute BIC score
    bic = compute_bic_score(X_train, model)
    bic_scores.append(bic)

    # Compute AIC score0
    log_likelihood = model.score(X_train)
    n_features = X_train.shape[1]
    n_params = n_components * (n_features + n_features + n_components - 1)
    aic = -2 * log_likelihood + 2 * n_params
    aic_scores.append(aic)

    print(f"Components: {n_components}, BIC: {bic:.2f}, AIC: {aic:.2f}")

# Plot BIC and AIC scores
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(n_components_range, bic_scores, '-o')
plt.xlabel('Number of Components')
plt.ylabel('BIC Score')
plt.title('BIC Score vs Number of Components')

plt.subplot(1, 2, 2)
plt.plot(n_components_range, aic_scores, '-o')
plt.xlabel('Number of Components')
plt.ylabel('AIC Score')
plt.title('AIC Score vs Number of Components')
plt.tight_layout()
plt.savefig("./results/model_selection_scores.png")

# Select optimal number of components based on BIC score (lower is better)
optimal_components = n_components_range[np.argmin(bic_scores)]
print(f"Optimal number of components based on BIC: {optimal_components}")

# === Train Gaussian HMM with Optimal Components (5 for this enhanced version) ===
print("Training HMM model with 5 components...")
model = GaussianHMM(
    n_components=5,  # Using 5 regimes for enhanced version
    covariance_type="diag",
    n_iter=2000,
    tol=1e-5,
    random_state=42
)
model.fit(X_train)

# Save the model
joblib.dump(model, 'Enhanced_HMM_Model.pkl')
joblib.dump(scaler, 'Feature_Scaler.pkl')

# === Regime Assignment ===
print("Assigning regimes...")
df_train["Regime"] = model.predict(X_train)
df_train["Return"] = df_train["price"].pct_change().fillna(0)

# Analyze regime characteristics for labeling
regime_stats = pd.DataFrame()
for regime in range(5):
    regime_data = df_train[df_train["Regime"] == regime]
    stats = {
        'avg_return': regime_data["Return"].mean() * 100,
        'volatility': regime_data["Return"].std() * 100,
        'median_return': regime_data["Return"].median() * 100,
        'count': len(regime_data),
        'pct_positive_days': (regime_data["Return"] > 0).mean() * 100
    }
    regime_stats = pd.concat([regime_stats, pd.DataFrame([stats])], ignore_index=True)

# Label regimes based on return and volatility characteristics
regime_stats.index = range(5)
print("\n=== Regime Statistics ===")
print(regime_stats)

# Sort regimes by average return for labeling
sorted_regimes = regime_stats.sort_values('avg_return')
regime_map = {}

# Assign labels to regimes
labels = ["Strong Bear", "Weak Bear", "Neutral", "Weak Bull", "Strong Bull"]
for i, idx in enumerate(sorted_regimes.index):
    regime_map[int(idx)] = labels[i]

print("\n=== Regime Labels ===")
for regime, label in regime_map.items():
    print(f"Regime {regime}: {label} (Avg Return: {regime_stats.loc[regime, 'avg_return']:.2f}%, Volatility: {regime_stats.loc[regime, 'volatility']:.2f}%)")

# === Predict Test Set Regimes ===
print("Predicting test set regimes...")
test_predictions = []
window_size = 30

for i in range(len(X_test)):
    sequence = np.vstack((X_train, X_test[:i+1]))[-window_size:]
    current_state = model.predict(sequence)[-1]
    test_predictions.append(current_state)

df_test["Regime"] = test_predictions
df_test["Market_Regime"] = df_test["Regime"].map(regime_map)

# === Combine Train & Test Sets ===
df_all = pd.concat([df_train, df_test]).reset_index(drop=True)
df_all["Market_Regime"] = df_all["Regime"].map(regime_map)

# === Regime Transition Analysis ===
print("\n=== Regime Transition Matrix ===")
print(pd.DataFrame(model.transmat_,
                   index=[regime_map[i] for i in range(5)],
                   columns=[regime_map[i] for i in range(5)]))

# === Emission Analysis ===
print("\n=== Feature Importance in Each Regime ===")
feature_importance = pd.DataFrame()

for i in range(5):
    means = model.means_[i]
    stds = np.sqrt(model.covars_[i])

    # Z-scores of feature means relative to overall distribution
    importance = pd.Series(means, index=features)
    feature_importance[regime_map[i]] = importance

print(feature_importance)

# Visualize feature importance by regime
plt.figure(figsize=(20, 10))
sns.heatmap(feature_importance, annot=True, cmap="coolwarm", center=0)
plt.title("Feature Importance by Market Regime")
plt.tight_layout()
plt.savefig("./results/feature_importance.png")

# === Regime Duration Analysis ===
df_all['regime_change'] = df_all['Regime'].ne(df_all['Regime'].shift()).astype(int)
df_all['regime_group'] = df_all['regime_change'].cumsum()

regime_durations = df_all.groupby(['regime_group', 'Market_Regime']).size().reset_index(name='duration')
print("\n=== Regime Duration Analysis ===")
duration_stats = regime_durations.groupby('Market_Regime')['duration'].agg(['mean', 'median', 'min', 'max'])
print(duration_stats)

# Plot regime duration distribution
plt.figure(figsize=(12, 6))
sns.boxplot(x='Market_Regime', y='duration', data=regime_durations)
plt.title('Distribution of Regime Durations')
plt.ylabel('Duration (days)')
plt.tight_layout()
plt.savefig("./results/regime_durations.png")

# === Visualization of Market Regimes ===
plt.figure(figsize=(14, 8))
colors = {"Strong Bull": "darkgreen", "Weak Bull": "lightgreen",
          "Neutral": "yellow", "Weak Bear": "salmon", "Strong Bear": "darkred"}

# Create background for regimes
for regime, color in colors.items():
    mask = df_all["Market_Regime"] == regime
    if mask.any():
        plt.scatter(df_all.loc[mask, "start_time"], df_all.loc[mask, "price"],
                    c=color, label=regime, alpha=0.7, s=10)

plt.title("Cryptocurrency Price with Market Regimes")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./results/market_regimes_visualization.png")

# === Confusion Matrix for Regimes ===
# Compute next day regime predictions based on current day
df_all['pred_next_regime'] = df_all['Regime'].shift(1)
df_all['actual_regime'] = df_all['Regime']

# Create confusion matrix
confusion = pd.crosstab(df_all['actual_regime'].dropna(),
                         df_all['pred_next_regime'].dropna(),
                         rownames=['Actual'],
                         colnames=['Predicted'])

# Convert to actual regime names
confusion.index = [regime_map[i] for i in confusion.index]
confusion.columns = [regime_map[i] for i in confusion.columns]

print("\n=== Regime Prediction Confusion Matrix ===")
print(confusion)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
plt.title('Regime Prediction Confusion Matrix')
plt.tight_layout()
plt.savefig("./results/regime_confusion_matrix.png")

# === Regime Prediction Function ===
def predict_regime(features_df, window_size=30):
    """
    Predict the current market regime based on recent feature data

    Parameters:
    -----------
    features_df : DataFrame with the same features used for training
    window_size : How many days of data to use for prediction

    Returns:
    --------
    regime : The predicted market regime
    probabilities : Probability distribution over all regimes
    """
    # Ensure we have the right features
    missing_features = set(features) - set(features_df.columns)
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # Scale the features
    X = scaler.transform(features_df[features].tail(window_size))

    # Predict hidden states
    states = model.predict(X)
    current_state = states[-1]

    # Get transition probabilities from current state
    next_state_probs = model.transmat_[current_state]

    return regime_map[current_state], {regime_map[i]: prob for i, prob in enumerate(next_state_probs)}

# === Model Performance Metrics ===
print("\n=== Model Performance Analysis ===")

# 1. Predictive Accuracy
df_all['correct_prediction'] = (df_all['pred_next_regime'] == df_all['actual_regime'])
accuracy = df_all['correct_prediction'].mean()
print(f"Next-Day Regime Prediction Accuracy: {accuracy:.2%}")

# 2. Feature Importance Analysis
def feature_importance_analysis(model, feature_names):
    """Analyze which features are most important for distinguishing between regimes"""
    feature_importance = np.zeros(len(feature_names))

    for i in range(model.n_components):
        for j in range(model.n_components):
            if i != j:
                # Calculate the difference in means between regimes
                diff = np.abs(model.means_[i] - model.means_[j])
                feature_importance += diff

    # Normalize
    feature_importance = feature_importance / feature_importance.sum()
    return pd.Series(feature_importance, index=feature_names).sort_values(ascending=False)

importance = feature_importance_analysis(model, features)
print("\nFeature Importance for Regime Classification:")
print(importance)

# Plot feature importance
plt.figure(figsize=(12, 6))
importance.plot(kind='bar')
plt.title('Feature Importance for Regime Classification')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig("./results/feature_importance_bar.png")

# 3. Model Stability Analysis
print("\n=== Model Stability Analysis ===")
# Check regime distribution over time
regime_counts = df_all.groupby(pd.Grouper(key='start_time', freq='M'))['Market_Regime'].value_counts().unstack().fillna(0)
regime_counts = regime_counts.div(regime_counts.sum(axis=1), axis=0)

plt.figure(figsize=(16, 6))
regime_counts.plot(kind='area', stacked=True, colormap='viridis')
plt.title('Regime Distribution Over Time')
plt.ylabel('Proportion')
plt.legend(title='Market Regime')
plt.tight_layout()
plt.savefig("./results/regime_distribution_time.png")

# === Feature Correlation within Regimes ===
# Analyze how features correlate differently in different regimes
plt.figure(figsize=(20, 16))
for i, regime in enumerate(regime_map.values()):
    plt.subplot(3, 2, i+1)
    regime_data = df_all[df_all['Market_Regime'] == regime]

    if len(regime_data) > 10:  # Need enough data for correlation
        corr_matrix = regime_data[features].corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, vmin=-1, vmax=1)
        plt.title(f'Feature Correlation in {regime} Regime')
    else:
        plt.text(0.5, 0.5, f"Insufficient data for {regime} regime",
                 horizontalalignment='center', verticalalignment='center')

plt.tight_layout()
plt.savefig("./results/feature_correlations_by_regime.png")

# === Price Change Distribution by Regime ===
plt.figure(figsize=(12, 8))
for regime in regime_map.values():
    regime_returns = df_all[df_all['Market_Regime'] == regime]['Return'].dropna()
    if len(regime_returns) > 10:
        sns.kdeplot(regime_returns, label=regime)

plt.title('Return Distribution by Market Regime')
plt.xlabel('Daily Return')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./results/return_distribution_by_regime.png")

# === Save Labeled Data ===
output_cols = ["start_time", "price", "Regime", "Market_Regime", "Return"] + features
df_all[output_cols].to_csv("./results/hmm_labeled_data.csv", index=False)

print("\n=== Final Regime Analysis ===")
print(f"Current Market Regime: {df_all['Market_Regime'].iloc[-1]}")

# Predict next regime
current_features = df_all[features].tail(30)
next_regime, probabilities = predict_regime(df_all[features], window_size=30)

print(f"Predicted Next Regime: {next_regime}")
print("Transition Probabilities:")
for regime, prob in probabilities.items():
    print(f"  {regime}: {prob:.2%}")

# Final regime summary
regime_summary = df_all.groupby('Market_Regime').agg({
    'Return': ['mean', 'std', 'median'],
    'price': ['count', 'min', 'max']
}).round(4)

print("\n=== Regime Summary Statistics ===")
print(regime_summary)