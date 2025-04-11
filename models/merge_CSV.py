import pandas as pd

# Load CSV files
coinglass = pd.read_csv('./data/coinglass_combined_2024_2025.csv')
merged_crypto = pd.read_csv('./data/merged_cq_data.csv')
glassnode = pd.read_csv('./data/glassnode_crypto_data.csv')
# data_4h = pd.read_csv('./data/data_4h.csv')

# Step 1: Merge coinglass & merged_crypto on start_time
df1 = pd.merge(coinglass, merged_crypto, on='start_time', how='inner')

# Step 2: Merge the result with glassnode on start_time
df2 = pd.merge(df1, glassnode, on='start_time', how='left')

# Step 3: Reset index to merge with data_4h (assuming row alignment, not time alignment)
df2.reset_index(drop=True, inplace=True)
# data_4h.reset_index(drop=True, inplace=True)

# Step 4: Concatenate data_4h with merged dataset by row index
final_df = pd.concat([df2], axis=1)

# Optional: Remove rows with NaNs if needed (e.g., first row might have NaNs)
final_df.dropna(inplace=True)

# Save merged result to a CSV
final_df.to_csv("./data/merged_dataset.csv", index=False)

print("✅ Merged dataset shape:", final_df.shape)
print("📌 Columns:", final_df.columns.tolist())
