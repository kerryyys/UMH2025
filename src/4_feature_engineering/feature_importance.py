import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Load and Prepare Data ===
df = pd.read_csv('./data/cleaned/btc/2023-2024/btc_features_output.csv')

# Drop unneeded columns (keep 70 features)
X = df.drop(columns=['start_time', 'date'])

# === 2. Check and clean data types ===
print(X.dtypes)  # Check data types to see if any column is an object (string)

# Convert 'cq_taker_buy_ratio' to numeric (or drop if it's not needed)
X['cq_taker_buy_ratio'] = pd.to_numeric(X['cq_taker_buy_ratio'], errors='coerce')  # Convert to numeric, NaNs for errors

# === 3. Ensure all features are numeric ===
X = X.select_dtypes(include=[np.number])  # Keep only numeric columns

# === 4. Replace inf and NaN values ===
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

# === 5. Fit HMM to the features ===
n_states = 3  # Set number of regimes
model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", random_state=42)
model.fit(X)

# Predict hidden states (regimes)
df['predicted_state'] = model.predict(X)

# === 6. Compute Mutual Information ===
mi_scores = mutual_info_classif(X, df['predicted_state'], discrete_features=False, random_state=42)

# Create DataFrame of feature importance
mi_df = pd.DataFrame({
    'Feature': X.columns,
    'Mutual_Information': mi_scores
}).sort_values(by='Mutual_Information', ascending=False).reset_index(drop=True)

# Print top 20 features
print(mi_df.head(20))

# === 7. (Optional) Visualize Top 20 Features ===
top20 = mi_df.head(20)
plt.figure(figsize=(10, 6))
sns.barplot(data=top20, x='Mutual_Information', y='Feature', palette='viridis')
plt.title("Top 20 Features by Mutual Information with HMM Regimes")
plt.tight_layout()
plt.show()

# === 8. (Optional) Save MI results to CSV ===
mi_df.to_csv('feature_importance_hmm_mi.csv', index=False)
