import glob
import pandas as pd
from pathlib import Path


class DataMerger:
    def __init__(self, key_column: str):
        self.key_column = key_column
        self.result_df = None

    def merge_from_prefix(self, directory: str, prefix: str, output_filename: str):
        search_path = Path(directory) / f"{prefix}*.csv"
        csv_files = glob.glob(str(search_path))
        csv_files.sort()

        dataframes = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                if self.key_column not in df.columns:
                    print(
                        f"File '{file}' does not contain the key column '{self.key_column}'. Skipping.")
                    continue
                if df.empty:
                    print(f"File '{file}' is empty. Skipping.")
                    continue
                dataframes.append(df)
            except pd.errors.EmptyDataError:
                print(f"File '{file}' is empty (EmptyDataError). Skipping.")
                continue

        if dataframes:
            merged_df = dataframes[0]
            for df in dataframes[1:]:
                duplicate_columns = [
                    col for col in df.columns if col in merged_df.columns and col != self.key_column]
                if duplicate_columns:
                    df = df.drop(columns=duplicate_columns)
                merged_df = pd.merge(
                    merged_df, df, on=self.key_column, how='inner')

            merged_df.to_csv(f"{output_filename}.csv", index=False)
            print(
                f"Combined CSV saved to '{output_filename}.csv' with {merged_df.shape[1]} columns.")
        else:
            print(
                f"No valid CSV files found in '{directory}' starting with prefix '{prefix}'.")

    def merge_pairwise(self, file1: str, prefix1: str, file2: str, prefix2: str, output_filename: str):
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        if self.key_column not in df1.columns or self.key_column not in df2.columns:
            raise ValueError(
                f"The key column '{self.key_column}' must exist in both files.")

        df1 = df1.rename(columns={
                         col: f"{prefix1}_{col}" for col in df1.columns if col != self.key_column})
        df2 = df2.rename(columns={
                         col: f"{prefix2}_{col}" for col in df2.columns if col != self.key_column})

        df1 = df1.rename(
            columns={f"{prefix1}_{self.key_column}": self.key_column})
        df2 = df2.rename(
            columns={f"{prefix2}_{self.key_column}": self.key_column})

        merged_df = pd.merge(df1, df2, on=self.key_column, how='inner')

        print("Merged DataFrame Columns:")
        for col in merged_df.columns:
            print(col)

        self.result_df = merged_df

    def drop_columns(self, columns_to_remove: list):
        if self.result_df is not None:
            self.result_df = self.result_df.drop(columns=columns_to_remove)

    def save_result(self, output_filename: str):
        if self.result_df is not None:
            self.result_df.to_csv(output_filename, index=False)
            print(f"Final merged DataFrame saved to '{output_filename}'")
