from util import DataMerger  # Adjust if path differs

merger = DataMerger(key_column="start_time")

merger.merge_from_prefix(
    directory="data/cryptoquant",
    prefix="cryptoquant_",
    output_filename="data/processed_data/cryptoquant_combined"
)

merger.merge_from_prefix(
    directory="data/coinglass",
    prefix="coinglass_",
    output_filename="data/processed_data/coinglass_combined"
)

merger.merge_pairwise(
    file1="data/processed_data/cryptoquant_combined.csv",
    prefix1="cq",
    file2="data/processed_data/coinglass_combined.csv",
    prefix2="cg",
    output_filename="data/processed_data/merged_cryptoquant_coinglass"
)

merger.drop_columns(["cg_time", "cq_is_shutdown", "cq_datetime", "cg_t"])

merger.save_result("data/processed_data/merged_cleaned.csv")
