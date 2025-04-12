import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize
import os

def preprocess_data(file_path, output_filename):
    # Load data
    df = pd.read_csv(file_path)
    
    # Parse dates explicitly
    df["start_time"] = pd.to_datetime(df["start_time"], unit="ms")

    # Rename for consistency
    df.rename(columns={"active_address": "active_addresses"}, inplace=True)

    # Sort and drop duplicate timestamps
    df = df.sort_values("start_time").drop_duplicates(subset=["start_time"])

    # Winsorize to limit extreme outliers (1% both tails)
    for col in ["active_addresses", "exchange_inflow", "exchange_outflow", "price", "transaction_count"]:
        df[col] = winsorize(df[col], limits=[0.01, 0.01])

    # Handle missing values (forward fill then backfill)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Standardize features
    features = ["active_addresses", "exchange_inflow", "exchange_outflow", "price", "transaction_count"]
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Save processed file
    os.makedirs("data/processed_data", exist_ok=True)
    output_path = os.path.join("data/processed_data", output_filename)
    df.to_csv(output_path, index=False)

    print(f"✅ Processed: {file_path} → {output_path}")
    return df

# Process both datasets
df_gn = preprocess_data("data/raw_data/GN_data.csv", "GN_data_clean.csv")
df_cq = preprocess_data("data/raw_data/CQ_data.csv", "CQ_data_clean.csv")
