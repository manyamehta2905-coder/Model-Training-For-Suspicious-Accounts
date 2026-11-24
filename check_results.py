
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("predictions.csv")
chains = pd.read_csv("chain_predictions.csv")

models = {
    "TRANSACTION FRAUD": ("actual_txn_fraud", "flag_txn_fraud"),
    "SENDER MULE": ("actual_sender_mule", "flag_sender_mule"),
    "RECEIVER MULE": ("actual_receiver_mule", "flag_receiver_mule")
}

print("\n========== TRANSACTION & ACCOUNT MODELS ==========")

for name, (actual, predicted) in models.items():

    data = df[df["split"] == "test"].copy()
    data = data.dropna(subset=[actual, predicted])

    y = data[actual].astype(int)
    p = data[predicted].astype(int)

    print(f"\n========== {name} ==========")
    print("Test rows:", len(data))
    print("Actual fraud cases:", int(y.sum()))
    print("Predicted fraud cases:", int(p.sum()))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, p, labels=[0, 1]))

    print("\nClassification Report:")
    print(classification_report(y, p, zero_division=0))

print("\n========== DEVELOPING CHAIN ==========")

data = chains[chains["split"] == "test"].copy()
data = data.dropna(subset=["actual_chain_fraud", "flag_chain"])

y = data["actual_chain_fraud"].astype(int)
p = data["flag_chain"].astype(int)

print("Snapshot rows:", len(data))
print("Unique chains:", data["chain_id"].nunique())
print("Actual fraud snapshots:", int(y.sum()))
print("Predicted fraud snapshots:", int(p.sum()))

print("\nConfusion Matrix:")
print(confusion_matrix(y, p, labels=[0, 1]))

print("\nClassification Report:")
print(classification_report(y, p, zero_division=0))

print("\n========== ALL CHECKS COMPLETED ==========")
