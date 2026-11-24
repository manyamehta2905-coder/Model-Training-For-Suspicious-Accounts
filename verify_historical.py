
import pandas as pd

original = pd.read_csv(
    "training_dataset_original.csv", low_memory=False
)

new = pd.read_csv(
    "training_dataset_historical.csv", low_memory=False
)

for df in [original, new]:
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    for col in ["sender_account", "receiver_account"]:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        ).astype("Int64").astype("string")

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

sample = new.sample(10, random_state=42)

print("========== VERIFYING ALL 8 FEATURES ==========")

for _, row in sample.iterrows():

    previous = original[
        original["timestamp"] < row["timestamp"]
    ]

    sender = row["sender_account"]
    receiver = row["receiver_account"]

    sender_in = previous[previous["receiver_account"] == sender]
    sender_out = previous[previous["sender_account"] == sender]

    receiver_in = previous[previous["receiver_account"] == receiver]
    receiver_out = previous[previous["sender_account"] == receiver]

    checks = [
        row["s_in_txn_count"] == len(sender_in),
        row["s_out_txn_count"] == len(sender_out),
        abs(row["s_total_in_amount"] - sender_in["amount"].sum()) < 0.02,
        abs(row["s_total_out_amount"] - sender_out["amount"].sum()) < 0.02,
        row["r_in_txn_count"] == len(receiver_in),
        row["r_out_txn_count"] == len(receiver_out),
        abs(row["r_total_in_amount"] - receiver_in["amount"].sum()) < 0.02,
        abs(row["r_total_out_amount"] - receiver_out["amount"].sum()) < 0.02
    ]

    print(row["txn_id"], "PASS" if all(checks) else "FAIL")

print("\nVerification complete.")
