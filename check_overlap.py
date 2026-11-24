
import pandas as pd

df = pd.read_csv("training_dataset.csv", low_memory=False)

chains = df.dropna(subset=["chain_id"]).copy()

# Find all accounts associated with each chain
account_chains = pd.concat([
    chains[["chain_id", "sender_account"]].rename(
        columns={"sender_account": "account"}
    ),
    chains[["chain_id", "receiver_account"]].rename(
        columns={"receiver_account": "account"}
    )
]).dropna().drop_duplicates()

# Count how many different chains contain each account
overlap = account_chains.groupby("account")["chain_id"].nunique()

print("========== ACCOUNT OVERLAP ==========")
print("Accounts in chained transactions:", len(overlap))
print("Accounts appearing in multiple chains:", (overlap > 1).sum())
print("Maximum chains per account:", overlap.max())

print("\n========== CHAIN STATISTICS ==========")
print("Total chains:", chains["chain_id"].nunique())
print("Average transactions per chain:",
      round(chains.groupby("chain_id").size().mean(), 2))
print("Largest chain:",
      chains.groupby("chain_id").size().max())

print("\n========== SAMPLE OVERLAPPING ACCOUNTS ==========")

shared = overlap[overlap > 1].sort_values(ascending=False).head(10)

for account in shared.index:
    ids = account_chains.loc[
        account_chains["account"] == account, "chain_id"
    ].unique()

    print("Account:", account, "| Chains:", list(ids))

print("\n========== COMPLETE ==========")
