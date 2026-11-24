
import pandas as pd
from collections import defaultdict

df = pd.read_csv("training_dataset.csv", low_memory=False)
df = df.dropna(subset=["chain_id"]).copy()

# Normalize account IDs to avoid int/float comparison problems
def normalize_account(value):
    if pd.isna(value):
        return None
    return str(int(float(value)))

# Each chain starts as its own component
chain_ids = df["chain_id"].unique().tolist()
parent = {chain: chain for chain in chain_ids}

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    root_a = find(a)
    root_b = find(b)

    if root_a != root_b:
        parent[root_b] = root_a

# Connect chains that share an account
account_to_first_chain = {}

for _, row in df.iterrows():
    chain = row["chain_id"]

    for column in ["sender_account", "receiver_account"]:
        account = normalize_account(row[column])

        if account is None:
            continue

        if account in account_to_first_chain:
            union(chain, account_to_first_chain[account])
        else:
            account_to_first_chain[account] = chain

# Find final connected components
components = defaultdict(list)

for chain in chain_ids:
    components[find(chain)].append(chain)

component_sizes = sorted(
    [len(chains) for chains in components.values()],
    reverse=True
)

print("\n========== CONNECTED NETWORK GROUPS ==========")
print("Total chains:", len(chain_ids))
print("Connected network groups:", len(components))
print("Largest network group:", max(component_sizes))
print("Smallest network group:", min(component_sizes))
print("Average chains per group:", round(
    sum(component_sizes) / len(component_sizes), 2
))

print("\nTop 20 network group sizes:")
print(component_sizes[:20])

# Examine labels in each network group
chain_labels = (
    df.groupby("chain_id")["label_chain_fraud"].first()
)

print("\n========== FRAUD LABEL DISTRIBUTION ==========")

for i, (root, chains_in_group) in enumerate(
    sorted(components.items(), key=lambda x: len(x[1]), reverse=True)[:10],
    start=1
):
    labels = chain_labels.reindex(chains_in_group)

    print(
        f"Group {i}: "
        f"{len(chains_in_group)} chains | "
        f"Fraudulent: {(labels == 1).sum()} | "
        f"Benign: {(labels == 0).sum()}"
    )

print("\n========== COMPLETE ==========")

