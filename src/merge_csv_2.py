import csv
import pandas as pd

def merge_excluding_columns(base_file, additional_files, output_file, exclude_columns_map):
    # Read the base file as a DataFrame (this will hold all rows)
    df_base = pd.read_csv(base_file)
    
    # Process each additional file
    for file in additional_files:
        exclude_cols = exclude_columns_map.get(file, [])
        print(f"📦 Processing {file} — excluding columns: {exclude_cols}")
        
        # Read the current additional file into DataFrame
        df_additional = pd.read_csv(file)

        # Exclude columns from the additional file based on the exclude_cols list
        df_additional = df_additional.drop(df_additional.columns[exclude_cols], axis=1)

        # Make sure the base file and additional file have the same number of rows, 
        # assuming all the files are aligned by index (same timestamps, etc.)
        df_base = pd.concat([df_base, df_additional], axis=1)

    # Save the result to a new CSV file
    df_base.to_csv(output_file, index=False)
    print(f"✅ Successfully created {output_file}")

# Example usage:
additional_csvs = [
    'data/processed_data/GN_data_clean.csv',
    'data/processed_data/CG_data_clean.csv'
]

# Specify which columns to exclude (0-based index) for each file
exclude_columns = {
    'data/processed_data/GN_data_clean.csv': [0, 1, 2, 3, 4, 5],  # For example, exclude columns 0-5 from GN data
    'data/processed_data/CG_data_clean.csv': [0]  # Exclude column 0 from CG data
}

merge_excluding_columns('data/processed_data/CQ_data_clean.csv', additional_csvs, 'data/processed_data/final_merged.csv', exclude_columns)
