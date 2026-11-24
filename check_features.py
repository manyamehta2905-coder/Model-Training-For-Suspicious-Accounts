
import pandas as pd

df = pd.read_csv("training_dataset.csv", low_memory=False)

columns = [
    "timestamp",
    "sender_account",
    "receiver_account",
    "s_in_txn_count",
    "s_out_txn_count",
    "s_total_in_amount",
    "s_total_out_amount",
    "r_in_txn_count",
    "r_out_txn_count",
    "r_total_in_amount",
    "r_total_out_amount",
    "chain_length",
    "total_chain_amount",
    "label_transaction_fraud"
]

print("\n========== FEATURE INSPECTION ==========")

print(df[columns].head(10).to_string(index=False))

print("\n========== MISSING VALUES ==========")
print(df[columns].isnull().sum())

print("\n========== DATE-WISE FEATURE CHECK ==========")

df["timestamp"] = pd.to_datetime(
    df["timestamp"], utc=True, errors="coerce"
)

print(
    df.groupby(df["timestamp"].dt.month)[
        ["s_in_txn_count", "s_out_txn_count",
         "chain_length", "total_chain_amount"]
    ].mean().round(2)
)

