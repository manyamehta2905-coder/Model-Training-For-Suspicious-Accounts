
import pandas as pd

df = pd.read_csv(
    "training_dataset_historical.csv",
    low_memory=False
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"], utc=True
)

duplicates = df["timestamp"].duplicated(keep=False)

print("Total transactions:", len(df))
print("Transactions sharing timestamps:", duplicates.sum())
print("Unique timestamps:", df["timestamp"].nunique())
print("Largest simultaneous transaction group:",
      df.groupby("timestamp").size().max())

print("\nSample duplicate timestamps:")
print(
    df.loc[duplicates, ["timestamp", "txn_id"]]
      .head(10)
      .to_string(index=False)
)
