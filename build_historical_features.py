
import pandas as pd

print("Loading dataset...", flush=True)

df = pd.read_csv(
    "training_dataset_original.csv",
    low_memory=False
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"], utc=True, errors="coerce"
)

for col in ["sender_account", "receiver_account"]:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    ).astype("Int64").astype("string")

df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

print("Building account transaction history...", flush=True)

# Outgoing transaction events
outgoing = df[
    ["timestamp", "sender_account", "amount"]
].rename(columns={"sender_account": "account"})

outgoing["out_count"] = 1
outgoing["out_amount"] = outgoing["amount"]
outgoing["in_count"] = 0
outgoing["in_amount"] = 0.0

# Incoming transaction events
incoming = df[
    ["timestamp", "receiver_account", "amount"]
].rename(columns={"receiver_account": "account"})

incoming["in_count"] = 1
incoming["in_amount"] = incoming["amount"]
incoming["out_count"] = 0
incoming["out_amount"] = 0.0

events = pd.concat([
    outgoing[["account", "timestamp", "in_count",
              "out_count", "in_amount", "out_amount"]],
    incoming[["account", "timestamp", "in_count",
              "out_count", "in_amount", "out_amount"]]
], ignore_index=True)

events = events.dropna(subset=["account", "timestamp"])

# Combine all activity for each account at each timestamp
events = events.groupby(
    ["account", "timestamp"], as_index=False
).sum()

events = events.sort_values(["account", "timestamp"])

stats = ["in_count", "out_count", "in_amount", "out_amount"]

# Cumulative totals, shifted so current timestamp is excluded
cumulative = events.groupby("account")[stats].cumsum()

events[["hist_" + c for c in stats]] = (
    cumulative.groupby(events["account"]).shift(1).fillna(0)
)

history = events[
    ["account", "timestamp"] + ["hist_" + c for c in stats]
]

print("Applying sender history...", flush=True)

df = df.merge(
    history,
    left_on=["sender_account", "timestamp"],
    right_on=["account", "timestamp"],
    how="left"
)

df["s_in_txn_count"] = df["hist_in_count"].fillna(0)
df["s_out_txn_count"] = df["hist_out_count"].fillna(0)
df["s_total_in_amount"] = df["hist_in_amount"].fillna(0)
df["s_total_out_amount"] = df["hist_out_amount"].fillna(0)

df = df.drop(
    columns=["account"] + ["hist_" + c for c in stats]
)

print("Applying receiver history...", flush=True)

df = df.merge(
    history,
    left_on=["receiver_account", "timestamp"],
    right_on=["account", "timestamp"],
    how="left"
)

df["r_in_txn_count"] = df["hist_in_count"].fillna(0)
df["r_out_txn_count"] = df["hist_out_count"].fillna(0)
df["r_total_in_amount"] = df["hist_in_amount"].fillna(0)
df["r_total_out_amount"] = df["hist_out_amount"].fillna(0)

df = df.drop(
    columns=["account"] + ["hist_" + c for c in stats]
)

df.to_csv("training_dataset_historical.csv", index=False)

print("\n========== COMPLETE ==========")
print("Transactions:", len(df))
print("Saved: training_dataset_historical.csv")
