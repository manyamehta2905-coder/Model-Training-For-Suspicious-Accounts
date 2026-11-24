
import pandas as pd

df = pd.read_csv(
    "training_dataset_original.csv",
    low_memory=False
)

print("\n========== SENDER FEATURES ==========")

for col in df.columns:
    if col.startswith("s_"):
        print(
            col,
            "| Missing:", df[col].isna().sum(),
            "| Unique:", df[col].nunique(),
            "| Mean:", round(df[col].mean(), 2)
            if pd.api.types.is_numeric_dtype(df[col]) else "N/A"
        )

print("\n========== RECEIVER FEATURES ==========")

for col in df.columns:
    if col.startswith("r_"):
        print(
            col,
            "| Missing:", df[col].isna().sum(),
            "| Unique:", df[col].nunique(),
            "| Mean:", round(df[col].mean(), 2)
            if pd.api.types.is_numeric_dtype(df[col]) else "N/A"
        )

print("\n========== CHAIN FEATURES ==========")

for col in df.columns:
    if col.startswith("chain_") or col in [
        "chain_length",
        "total_hops",
        "unique_accounts",
        "total_chain_amount",
        "avg_time_between_txns_sec",
        "max_time_gap_sec",
        "chain_duration_sec",
        "amount_retention_ratio",
        "max_fan_out",
        "max_fan_in",
        "has_cycle"
    ]:
        print(
            col,
            "| Missing:", df[col].isna().sum(),
            "| Unique:", df[col].nunique()
        )

print("\n========== AUDIT COMPLETE ==========")
