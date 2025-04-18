import glob
from pathlib import Path
from functools import reduce
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
from rich.console import Console

console = Console()


class DataCombiner:
    def __init__(
        self,
        key_column: str,
        always_create_parent: bool = True,
        show_progress: bool = True,
        merge_drop_threshold: int = 200,
    ):
        self.key_column = key_column
        self.always_create_parent = always_create_parent
        self.show_progress = show_progress
        self.merge_drop_threshold = merge_drop_threshold
        self.result_df = None

    def _load_and_prepare(self, file: str, prefix: str, use_filename_as_column_name: bool):
        warnings = []
        try:
            df = pd.read_csv(file)
        except pd.errors.EmptyDataError:
            warnings.append(f"{file} is empty")
            return None, warnings
        if self.key_column not in df.columns or df.empty:
            warnings.append(f"{file} missing '{self.key_column}' or empty")
            return None, warnings
        if use_filename_as_column_name:
            stem = Path(file).stem
            base = stem[len(prefix):]
            non_keys = [c for c in df.columns if c != self.key_column]
            mapping = {non_keys[0]: base} if len(non_keys) == 1 else {
                c: f"{base}_{c}" for c in non_keys
            }
            df = df.rename(columns=mapping)
        return df, warnings

    def _prefix_and_load(self, file: str, prefix: str):
        df = pd.read_csv(file)
        if self.key_column not in df.columns:
            raise ValueError(f"Key '{self.key_column}' not found in {file}")
        rename_map = {
            c: f"{prefix}_{c}" for c in df.columns if c != self.key_column
        }
        return df.rename(columns=rename_map)

    def _merge_two(self, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        dupes = set(left.columns) & set(right.columns) - {self.key_column}
        if dupes:
            right = right.drop(columns=list(dupes))
        return pd.merge(left, right, on=self.key_column, how='inner')

    def combine_from_prefix(
        self,
        directory: str,
        prefix: str,
        output_filename: str,
        use_filename_as_column_name: bool = False,
    ) -> pd.DataFrame:
        files = sorted(glob.glob(str(Path(directory) / f"{prefix}*.csv")))
        dfs = []  # list of (df, file)
        warnings = []

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self._load_and_prepare, f, prefix, use_filename_as_column_name
                ): f
                for f in files
            }
            loader = as_completed(futures)
            if self.show_progress:
                loader = tqdm(loader, total=len(futures), desc="Loading CSVs")
            for future in loader:
                f = futures[future]
                df, warn = future.result()
                warnings.extend(warn)
                if df is not None:
                    dfs.append((df, f))
        for w in warnings:
            console.print(f"[bold red]WARNING:[/bold red] {w}")
        if not dfs:
            raise ValueError(
                f"No valid CSVs in {directory} with prefix {prefix}"
            )

        # Iterative merge with threshold check
        merged, first_file = dfs[0]
        skipped = []
        for df, file in dfs[1:]:
            next_merged = self._merge_two(merged, df)
            dropped = merged.shape[0] - next_merged.shape[0]
            if dropped > self.merge_drop_threshold:
                console.print(
                    f"[bold yellow]SKIP:[/bold yellow] Skipping {file}, merge dropped {dropped} rows > threshold {self.merge_drop_threshold}"
                )
                skipped.append(file)
            else:
                merged = next_merged

        out_path = Path(output_filename)
        if out_path.suffix.lower() != ".csv":
            out_path = out_path.with_suffix(".csv")
        parent = out_path.parent
        if not parent.exists():
            if self.always_create_parent:
                parent.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(f"Directory '{parent}' does not exist")

        merged.to_csv(out_path, index=False)
        self.result_df = merged
        console.print(
            f"[bold green]INFO:[/bold green] Successfully saved combined CSV to {out_path}"
        )
        if skipped:
            console.print(
                f"[bold yellow]INFO:[/bold yellow] Skipped files: {', '.join(skipped)}"
            )
        return merged

    def combine_files(
        self,
        file_prefix_pairs: list[tuple[str, str]],
        output_filename: str,
    ) -> pd.DataFrame:
        if not file_prefix_pairs:
            raise ValueError("No file-prefix pairs provided")
        dfs = []  # list of (df, file)
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self._prefix_and_load, file, prefix
                ): file
                for file, prefix in file_prefix_pairs
            }
            loader = as_completed(futures)
            if self.show_progress:
                loader = tqdm(loader, total=len(futures), desc="Loading files")
            for future in loader:
                file = futures[future]
                df = future.result()
                dfs.append((df, file))

        # Iterative merge with threshold check
        merged, first_file = dfs[0]
        skipped = []
        for df, file in dfs[1:]:
            next_merged = self._merge_two(merged, df)
            dropped = merged.shape[0] - next_merged.shape[0]
            if dropped > self.merge_drop_threshold:
                console.print(
                    f"[bold yellow]SKIP:[/bold yellow] Skipping {file}, merge dropped {dropped} rows > threshold {self.merge_drop_threshold}"
                )
                skipped.append(file)
            else:
                merged = next_merged

        self.result_df = merged
        out_path = Path(output_filename)
        if out_path.suffix.lower() != ".csv":
            out_path = out_path.with_suffix(".csv")
        parent = out_path.parent
        if not parent.exists():
            if self.always_create_parent:
                parent.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(f"Directory '{parent}' does not exist")
        merged.to_csv(out_path, index=False)
        console.print(
            f"[bold green]INFO:[/bold green] Successfully saved combined {len(dfs)} files to {out_path}"
        )
        if skipped:
            console.print(
                f"[bold yellow]INFO:[/bold yellow] Skipped files: {', '.join(skipped)}"
            )
        return merged

    def drop_columns(self, columns_to_remove: list) -> pd.DataFrame:
        if self.result_df is not None:
            self.result_df = self.result_df.drop(columns=columns_to_remove)
        return self.result_df

    def save_result(self, output_filename: str) -> Path:
        if self.result_df is not None:
            dest = Path(output_filename)
            if dest.suffix.lower() != ".csv":
                dest = dest.with_suffix(".csv")
            dest.parent.mkdir(parents=True, exist_ok=True)
            self.result_df.to_csv(dest, index=False)
            console.print(
                f"[bold green]INFO:[/bold green] Successfully saved result to {dest}"
            )
            return dest
        raise ValueError("No result to save")
