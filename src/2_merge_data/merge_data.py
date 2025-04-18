from util import DataCombiner

BASE_DIR = "data"
BASE_PROCESSED_DIR = "data/processed"
CRYPTO = "btc"
YEARS = ["2022-2023", "2023-2024", "2024-2025"]
PROVIDERS = ["glassnode", "coinglass", "cryptoquant"]


combiner = DataCombiner(key_column="start_time")

for year in YEARS:
    for provider in PROVIDERS:
        combiner.combine_from_prefix(
            directory=f"{BASE_DIR}/{CRYPTO}/{year}/{provider}",
            prefix=f"{provider}_",
            output_filename=f"{BASE_PROCESSED_DIR}/{CRYPTO}/{year}/{provider}_combined",
            use_filename_as_column_name=True if provider == "glassnode" else False
        )

    combiner.combine_files(
        file_prefix_pairs=[
            (f"{BASE_PROCESSED_DIR}/{CRYPTO}/{year}/cryptoquant_combined.csv", "cq"),
            (f"{BASE_PROCESSED_DIR}/{CRYPTO}/{year}/coinglass_combined.csv", "cg"),
            (f"{BASE_PROCESSED_DIR}/{CRYPTO}/{year}/glassnode_combined.csv", "gn"),
        ],
        output_filename=f"{BASE_PROCESSED_DIR}/{CRYPTO}/{year}/{CRYPTO}_{year}_combined_data.csv"
    )
