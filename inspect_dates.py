import pandas as pd

df = pd.read_csv("training_dataset.csv", low_memory=False)

df["timestamp"] = pd.to_datetime(
    df["timestamp"], errors="coerce", utc=True
)

print("Earliest transaction:", df["timestamp"].min())
print("Latest transaction:", df["timestamp"].max())

print("\nTransactions by month:")
print(
    df.groupby(df["timestamp"].dt.to_period("M"))
      .size()
)

print("\nFraud labels by month:")
print(
    df.groupby(df["timestamp"].dt.to_period("M"))[
        "label_transaction_fraud"
    ].agg(["count", "sum"])
)