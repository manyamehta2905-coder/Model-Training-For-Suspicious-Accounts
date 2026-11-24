"""
Developing-chain snapshots -> features_chain_snapshots.csv
One row per (chain, hop k): every feature uses ONLY hops 1..k (what a live system could see
when hop k arrives). The label is the chain's final label (known only after the chain ends,
so for training we use chains that are fully finished inside the training window).
Run:  venv\\Scripts\\python.exe build_chain_snapshots.py
"""
import numpy as np, pandas as pd
import config as C


def snapshot_rows(g, sc, rc):
    g = g.sort_values([C.COL_TS] + ([C.COL_HOP] if C.COL_HOP else []) + ["_row"], kind="mergesort")
    t0 = g[C.COL_TS].iloc[0]
    out, prev_ts = [], None
    senders, nodes, out_deg, in_deg = set(), set(), {}, {}
    amts, gaps, has_cycle = [], [], False
    for k, (_, r) in enumerate(g.iterrows(), start=1):
        s, d, a, ts = r["_s"], r["_r"], float(r[C.COL_AMOUNT]), r[C.COL_TS]
        if d in senders or d == s:          # money returned to an account that already sent in this chain
            has_cycle = True
        senders.add(s); nodes.update([s, d])
        out_deg[s] = out_deg.get(s, 0) + 1
        in_deg[d] = in_deg.get(d, 0) + 1
        amts.append(a)
        gaps.append(0.0 if prev_ts is None else (ts - prev_ts).total_seconds())
        prev_ts = ts
        out.append({
            "chain_id": r[C.COL_CHAIN], "hop_index": k, "snapshot_ts": ts, "txn_id": r[C.COL_TXN_ID],
            "chain_start_ts": t0,
            "hops_so_far": k, "unique_accounts_so_far": len(nodes),
            "total_amount_so_far": sum(amts), "mean_amount_so_far": float(np.mean(amts)),
            "max_amount_so_far": max(amts), "first_amount": amts[0], "last_amount": a,
            "last_to_first_amount_ratio": a / amts[0] if amts[0] else np.nan,
            "duration_sec_so_far": (ts - t0).total_seconds(),
            "mean_gap_sec": float(np.mean(gaps[1:])) if k > 1 else np.nan,
            "last_gap_sec": gaps[-1] if k > 1 else np.nan,
            "max_fan_out_so_far": max(out_deg.values()), "max_fan_in_so_far": max(in_deg.values()),
            "has_cycle_so_far": int(has_cycle),
            "last_sender": s, "last_receiver": d,
        })
    return out


def main():
    df = pd.read_csv(C.SOURCE_CSV, low_memory=False)
    sc, rc = C.resolve_columns(df)
    for c in (C.COL_CHAIN, C.LABEL_CHAIN):
        if c not in df.columns:
            raise SystemExit(f"Missing column {c}")
    df[C.COL_TS] = pd.to_datetime(df[C.COL_TS], utc=True)
    df["_s"] = df[sc].astype(str).str.strip().str.upper()
    df["_r"] = df[rc].astype(str).str.strip().str.upper()
    df["_row"] = np.arange(len(df))
    ch = df[df[C.COL_CHAIN].notna()]
    rows = []
    for cid, g in ch.groupby(C.COL_CHAIN, sort=False):
        labs = g[C.LABEL_CHAIN].dropna().unique()
        if len(labs) != 1:
            print(f"warning: chain {cid} has inconsistent labels, skipped"); continue
        snaps = snapshot_rows(g, sc, rc)
        for s_ in snaps:
            s_["label_chain_fraud"] = int(labs[0])
        rows += snaps
    snap = pd.DataFrame(rows)
    snap["chain_total_hops"] = snap.groupby("chain_id")["hop_index"].transform("max")  # for audit only
    snap["chain_end_ts"] = snap.groupby("chain_id")["snapshot_ts"].transform("max")    # for audit only

    # Chain split: assign by chain START; a chain that is not completely finished before its
    # split ends is dropped, so no label from the future leaks into training/validation.
    start = C.split_of(snap["chain_start_ts"]); end = C.split_of(snap["chain_end_ts"])
    snap["split"] = np.where(start == end, start, "dropped_straddles_boundary")
    snap.to_csv(C.CHAIN_SNAP_CSV, index=False)
    per_chain = snap.drop_duplicates("chain_id")
    print(f"Wrote {C.CHAIN_SNAP_CSV}: {len(snap):,} snapshots from {len(per_chain):,} chains")
    print("Chains per split:\n", per_chain["split"].value_counts().to_string())
    print("Fraud chains per split:\n", per_chain.groupby("split")["label_chain_fraud"].sum().to_string())
    print("Hop order: sorted by timestamp"
          + (f" then '{C.COL_HOP}'" if C.COL_HOP else " (set COL_HOP in config.py if a hop column exists)"))


if __name__ == "__main__":
    main()