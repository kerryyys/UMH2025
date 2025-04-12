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
    cols_to_winsorize = ["active_addresses", "exchange_inflow", "exchange_outflow", "price", "transaction_count"]
    
    # Include SSR_v for GN dataset if it exists
    if "SSR_v" in df.columns:
        cols_to_winsorize.append("SSR_v")
    
    # Include whale_ratio for CQ dataset if it exists
    if "exchange_whale_ratio" in df.columns:
        cols_to_winsorize.append("exchange_whale_ratio")

    # Include reserve_usd for CQ dataset if it exists
    if "reserve_usd" in df.columns:
        cols_to_winsorize.append("reserve_usd")
        
    for col in cols_to_winsorize:
        df[col] = winsorize(df[col], limits=[0.01, 0.01])

    # Handle missing values (forward fill then backfill)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Standardize features (including SSR_v and whale_ratio if present)
    features = ["active_addresses", "exchange_inflow", "exchange_outflow", "price", "transaction_count"]
    
    # Add SSR_v or whale_ratio to features if they exist
    if "SSR_v" in df.columns:
        features.append("SSR_v")
    if "whale_ratio" in df.columns:
        features.append("whale_ratio")
    if "reserve_usd" in df.columns:
        features.append("reserve_usd")
    
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Save processed file
    os.makedirs("data/processed_data", exist_ok=True)
    output_path = os.path.join("data/processed_data", output_filename)
    df.to_csv(output_path, index=False)

    print(f"Processed: {file_path} -> {output_path}")
    return df

def preprocess_data_cg(file_path, output_filename):
    # Load data
    df = pd.read_csv(file_path)
    
    # Parse dates explicitly
    df["start_time"] = pd.to_datetime(df["start_time"], unit="ms")
    
    # Rename columns for consistency
    rename_dict = {
        "open_interest": "open_interest",
        "high open interest": "high_open_interest",
        "low open interest": "low_open_interest",
        "open value of open interest": "open_value_of_open_interest",
        "funding_rate": "funding_rate",
        "high funding rate": "high_funding_rate",
        "low funding rate": "low_funding_rate",
        "open funding rate": "open_funding_rate",
        "longAccount": "long_account",
        "longShortRatio": "long_short_ratio",
        "shortAccount": "short_account"
    }
    
    df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns}, inplace=True)

    # Sort timestamps
    df = df.sort_values("start_time")

    # Winsorize to limit extreme outliers (1% both tails)
    for col in ["open_interest", "high_open_interest", "low_open_interest", "open_value_of_open_interest", "funding_rate",
                "high_funding_rate", "low_funding_rate", "open_funding_rate", "long_account", "long_short_ratio", "short_account"]:
        df[col] = winsorize(df[col], limits=[0.01, 0.01])

    # Handle missing values (forward fill then backfill)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Standardize features
    features = ["open_interest", "high_open_interest", "low_open_interest", "open_value_of_open_interest", "funding_rate",
                "high_funding_rate", "low_funding_rate", "open_funding_rate", "long_account", "long_short_ratio", "short_account"]
    
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Save processed file
    os.makedirs("data/processed_data", exist_ok=True)
    output_path = os.path.join("data/processed_data", output_filename)
    df.to_csv(output_path, index=False)

    print(f"Processed: {file_path} -> {output_path}")
    return df

# Process both datasets
df_gn = preprocess_data("data/raw_data/GN_data.csv", "GN_data_clean.csv")
df_cq = preprocess_data("data/raw_data/CQ_data.csv", "CQ_data_clean.csv")
df_cg = preprocess_data_cg("data/raw_data/coinglass_combined_2024_2025.csv", "CG_data_clean.csv")
