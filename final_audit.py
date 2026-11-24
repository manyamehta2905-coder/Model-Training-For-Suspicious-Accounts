
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

pred = pd.read_csv("predictions.csv")
original = pd.read_csv("training_dataset_original.csv")
chains = pd.read_csv("chain_predictions.csv")

print("\n========== DATA INTEGRITY ==========")

print("Prediction rows:", len(pred))
print("Duplicate transaction IDs:", pred["txn_id"].duplicated().sum())
print("Missing transaction IDs:", pred["txn_id"].isna().sum())

print("\nSplit counts:")
print(pred["split"].value_counts())

print("\n========== RECEIVER MODEL AUDIT ==========")

receiver_info = original[
    ["txn_id", "receiver_account"]
].drop_duplicates("txn_id")

merged = pred.merge(
    receiver_info,
    on="txn_id",
    how="left",
    validate="one_to_one"
)

test = merged[merged["split"] == "test"].copy()

print("Total test rows:", len(test))
print("Missing receiver accounts:", test["receiver_account"].isna().sum())
print("Valid receiver accounts:", test["receiver_account"].notna().sum())

valid = test[
    test["receiver_account"].notna()
].dropna(subset=["actual_receiver_mule", "flag_receiver_mule"])

y = valid["actual_receiver_mule"].astype(int)
p = valid["flag_receiver_mule"].astype(int)

print("\nReceiver metrics on valid accounts only:")
print("Rows:", len(valid))
print("Actual mule cases:", int(y.sum()))
print("Predicted mule cases:", int(p.sum()))
print(confusion_matrix(y, p, labels=[0, 1]))
print(classification_report(y, p, zero_division=0))

print("\n========== CHAIN AUDIT ==========")

test_chains = chains[chains["split"] == "test"].copy()

print("Test snapshots:", len(test_chains))
print("Unique chains:", test_chains["chain_id"].nunique())
print("\nSnapshots per chain:")
print(test_chains.groupby("chain_id").size().describe())

print("\n========== AUDIT COMPLETED ==========")
