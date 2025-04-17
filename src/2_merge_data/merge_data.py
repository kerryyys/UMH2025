from util import DataCombiner

BASE_DIR = "data"
BASE_PROCESSED_DIR = "data/processed"
CRYPTO = "btc"
YEAR = "2023-2024"

combiner = DataCombiner(key_column="start_time")


# PROVIDER = "glassnode"
# merger.merge_from_prefix(
#     directory=f"{BASE_DIR}/{CRYPTO}/{YEAR}/{PROVIDER}",
#     prefix="glassnode_",
#     output_filename=f"{BASE_PROCESSED_DIR}/{CRYPTO}/{YEAR}/{PROVIDER}_combined",
#     use_filename_as_column_name=True
# )

# PROVIDER = "coinglass"
# merger.merge_from_prefix(
#     directory=f"{BASE_DIR}/{CRYPTO}/{YEAR}/{PROVIDER}",
#     prefix=f"{PROVIDER}_",
#     output_filename=f"{BASE_PROCESSED_DIR}/{CRYPTO}/{YEAR}/{PROVIDER}_combined",
#     use_filename_as_column_name=False
# )

# PROVIDER = "cryptoquant"
# merger.merge_from_prefix(
#     directory=f"{BASE_DIR}/{CRYPTO}/{YEAR}/{PROVIDER}",
#     prefix=f"{PROVIDER}_",
#     output_filename=f"{BASE_PROCESSED_DIR}/{CRYPTO}/{YEAR}/{PROVIDER}_combined",
#     use_filename_as_column_name=False
# )

# merger.merge_from_prefix(
#     directory="data/cryptoquant",
#     prefix="cryptoquant_",
#     output_filename="data/processed_data/cryptoquant_combined"
# )

# merger.merge_from_prefix(
#     directory="data/coinglass",
#     prefix="coinglass_",
#     output_filename="data/processed_data/coinglass_combined"
# )

combiner.combine_files(
    file_prefix_pairs=[
        (f"{BASE_PROCESSED_DIR}/{CRYPTO}/{YEAR}/cryptoquant_combined.csv", "cq"),
        (f"{BASE_PROCESSED_DIR}/{CRYPTO}/{YEAR}/coinglass_combined.csv", "cg"),
        (f"{BASE_PROCESSED_DIR}/{CRYPTO}/{YEAR}/glassnode_combined.csv", "gn"),
    ],
    output_filename=f"{BASE_PROCESSED_DIR}/{CRYPTO}/{YEAR}/{CRYPTO}_{YEAR}_combined_data.csv"
)

# merger.drop_columns(["cg_time", "cq_is_shutdown", "cq_datetime", "cg_t"])

# merger.save_result("data/processed_data/merged_cleaned.csv")
