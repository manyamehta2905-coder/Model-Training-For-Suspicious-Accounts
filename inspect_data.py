
import pandas as pd

df = pd.read_csv("training_dataset.csv", low_memory=False)

chains = df.dropna(subset=["chain_id"]).copy()

# Convert both account columns to comparable numeric values
chains["sender_num"] = pd.to_numeric(
    chains["sender_account"], errors="coerce"
)
chains["receiver_num"] = pd.to_numeric(
    chains["receiver_account"], errors="coerce"
)

chains = chains.sort_values(["chain_id", "hop_index"])

results = []

for chain_id, group in chains.groupby("chain_id"):
    group = group.reset_index(drop=True)

    if len(group) < 2:
        continue

    connected = (
        group["receiver_num"].iloc[:-1].to_numpy()
        == group["sender_num"].iloc[1:].to_numpy()
    )

    results.append({
        "chain_id": chain_id,
        "transactions": len(group),
        "connected_links": int(connected.sum()),
        "expected_links": len(group) - 1,
        "fully_connected": bool(connected.all())
    })

report = pd.DataFrame(results)

print("\n========== CORRECTED CONNECTIVITY ==========")
print("Total chains:", len(report))
print("Fully connected:", report["fully_connected"].sum())
print("Not fully connected:", (~report["fully_connected"]).sum())

print("\nSample chains:")
print(report.head(10).to_string(index=False))

print("\n========== LABEL CONSISTENCY ==========")

label_counts = (
    chains.groupby("chain_id")["label_chain_fraud"].nunique()
)

print("Chains with consistent labels:", (label_counts <= 1).sum())
print("Chains with inconsistent labels:", (label_counts > 1).sum())

print("\n========== COMPLETE ==========")

