"""Brute-force check: recompute features for random transactions straight from the raw file
using ONLY rows with timestamp < current timestamp, and compare with features_txn.csv."""
import sys, numpy as np, pandas as pd
import config as C

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
raw = pd.read_csv(C.SOURCE_CSV, low_memory=False)
sc, rc = C.resolve_columns(raw)
raw[C.COL_TS] = pd.to_datetime(raw[C.COL_TS], utc=True)
raw["_s"] = raw[sc].astype(str).str.strip().str.upper()
raw["_r"] = raw[rc].astype(str).str.strip().str.upper()
f = pd.read_csv(C.TXN_FEATURES_CSV, low_memory=False).set_index("txn_id")
rng = np.random.default_rng(1)
bad = 0
# include rows that share a timestamp with another row (hardest case)
dup = raw[raw.duplicated(C.COL_TS, keep=False)].sample(min(50, raw.duplicated(C.COL_TS, keep=False).sum()), random_state=1) if raw.duplicated(C.COL_TS, keep=False).any() else raw.iloc[:0]
sample = pd.concat([raw.sample(N, random_state=1), dup])
for _, t in sample.iterrows():
    past = raw[raw[C.COL_TS] < t[C.COL_TS]]
    row = f.loc[t[C.COL_TXN_ID]]
    for role, acct, col in (("s", t["_s"], "_s"), ("r", t["_r"], "_r")):
        outs = past[past["_s"] == acct]; ins = past[past["_r"] == acct]
        exp = {"in_cnt": len(ins), "out_cnt": len(outs),
               "tot_in": ins[C.COL_AMOUNT].sum(), "tot_out": outs[C.COL_AMOUNT].sum(),
               "distinct_receivers": outs["_r"].nunique(), "distinct_senders": ins["_s"].nunique(),
               "txn_last_1h": int(((past["_s"] == acct) | (past["_r"] == acct)).mul(
                   past[C.COL_TS] >= t[C.COL_TS] - pd.Timedelta(seconds=C.VELOCITY_WINDOW_SEC)).sum())}
        if C.COL_DEVICE in raw: exp["distinct_devices"] = outs[C.COL_DEVICE].nunique()
        for k, v in exp.items():
            got = row[f"{role}_hist_{k}"]
            if not np.isclose(got, v, rtol=1e-6, atol=1e-6):
                bad += 1
                if bad <= 10: print("MISMATCH", t[C.COL_TXN_ID], role, k, "got", got, "expected", v)
    pp = int(((past["_s"] == t["_s"]) & (past["_r"] == t["_r"])).sum())
    if row["pair_prior_txn_count"] != pp:
        bad += 1; print("MISMATCH pair", t[C.COL_TXN_ID], row["pair_prior_txn_count"], pp)
print(f"Checked {len(sample)} transactions ({len(dup)} sharing a timestamp). Mismatches: {bad}")
sys.exit(1 if bad else 0)
