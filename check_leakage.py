
import pandas as pd

df = pd.read_csv("training_dataset.csv", low_memory=False)

# Convert account IDs into consistent strings
for col in ["sender_account", "receiver_account"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64").astype("string")

# Calculate actual totals using the entire dataset
outgoing = df.groupby("sender_account").agg(
    actual_out_count=("txn_id", "count"),
    actual_out_amount=("amount", "sum")
)

incoming = df.groupby("receiver_account").agg(
    actual_in_count=("txn_id", "count"),
    actual_in_amount=("amount", "sum")
)

# Compare stored sender statistics with full-dataset totals
df["actual_sender_out_count"] = df["sender_account"].map(
    outgoing["actual_out_count"]
)

df["actual_sender_out_amount"] = df["sender_account"].map(
    outgoing["actual_out_amount"]
)

df["actual_sender_in_count"] = df["sender_account"].map(
    incoming["actual_in_count"]
)

print("\n========== SENDER FEATURE COMPARISON ==========")

print(df[[
    "sender_account",
    "s_out_txn_count",
    "actual_sender_out_count",
    "s_total_out_amount",
    "actual_sender_out_amount",
    "s_in_txn_count",
    "actual_sender_in_count"
]].head(15).to_string(index=False))

print("\n========== MATCH PERCENTAGES ==========")

print(
    "Outgoing count match:",
    round(
        (df["s_out_txn_count"] == df["actual_sender_out_count"]).mean() * 100,
        2
    ), "%"
)

print(
    "Outgoing amount match:",
    round(
        (
            (df["s_total_out_amount"] - df["actual_sender_out_amount"]).abs()
            < 0.02
        ).mean() * 100,
        2
    ), "%"
)

print(
    "Incoming count match:",
    round(
        (df["s_in_txn_count"] == df["actual_sender_in_count"]).mean() * 100,
        2
    ), "%"
)
