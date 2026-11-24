
"""
Scores transactions and developing chains using trained models.

Run:
venv\Scripts\python.exe predict.py

Outputs:
1. predictions.csv
2. chain_predictions.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import config as C


TXN_MODELS = {
    "transaction_fraud": "p_txn_fraud",
    "sender_mule": "p_sender_mule",
    "receiver_mule": "p_receiver_mule"
}


def band(probabilities, thresholds):
    return np.where(
        probabilities >= thresholds["high"],
        "HIGH",
        np.where(
            probabilities >= thresholds["medium"],
            "MEDIUM",
            "LOW"
        )
    )


def load_model(name):
    path = os.path.join(C.MODELS_DIR, f"{name}.joblib")

    if not os.path.exists(path):
        raise SystemExit(
            f"ERROR: {path} not found. Run train_models.py first."
        )

    return joblib.load(path)


def check_features(data, features, model_name):
    missing = [f for f in features if f not in data.columns]

    if missing:
        raise SystemExit(
            f"{model_name}: Missing features: {missing}"
        )


def predict_transactions(tx, output_file):

    print("\n========== TRANSACTION PREDICTIONS ==========")

    # Keep original column names for dashboard compatibility.
    keep = [
        "txn_id",
        "timestamp",
        "sender_account",
        "receiver_account",
        "amount",
        "channel",
        "direction",
        "status",
        "chain_id",
        "hop_index",
        "split"
    ]

    available = [c for c in keep if c in tx.columns]
    result = tx[available].copy()

    for model_name, probability_col in TXN_MODELS.items():

        bundle = load_model(model_name)

        features = bundle["numeric"] + bundle["categorical"]

        check_features(tx, features, model_name)

        probabilities = bundle["model"].predict_proba(
            tx[features]
        )[:, 1]

        thresholds = bundle["thresholds"]

        result[probability_col] = probabilities

        result[probability_col.replace("p_", "risk_")] = band(
            probabilities,
            thresholds
        )

        result[probability_col.replace("p_", "flag_")] = (
            probabilities >= thresholds["high"]
        ).astype(int)

        print(
            f"{model_name}: scored {len(result):,} transactions"
        )

    # Actual labels for evaluation.
    labels = [
        (C.LABEL_TXN, "actual_txn_fraud"),
        (C.LABEL_SENDER, "actual_sender_mule"),
        (C.LABEL_RECEIVER, "actual_receiver_mule")
    ]

    for source_col, output_col in labels:
        if source_col in tx.columns:
            result[output_col] = tx[source_col]

    result.to_csv(output_file, index=False)

    print(f"\nSaved: {output_file}")
    print(f"Total transactions: {len(result):,}")

    return result


def predict_chains(output_file):

    print("\n========== CHAIN PREDICTIONS ==========")

    if not os.path.exists(C.CHAIN_SNAP_CSV):
        raise SystemExit(
            f"ERROR: {C.CHAIN_SNAP_CSV} not found.\n"
            "Run build_chain_snapshots.py first."
        )

    ch = pd.read_csv(C.CHAIN_SNAP_CSV, low_memory=False)

    if ch.empty:
        raise SystemExit("ERROR: Chain snapshots file is empty.")

    bundle = load_model("developing_chain")

    features = bundle["numeric"]

    check_features(ch, features, "developing_chain")

    required_columns = [
        "chain_id",
        "hop_index",
        "snapshot_ts",
        "txn_id",
        "last_sender",
        "last_receiver",
        "split",
        "label_chain_fraud"
    ]

    missing = [
        col for col in required_columns
        if col not in ch.columns
    ]

    if missing:
        raise SystemExit(
            f"Chain snapshots missing columns: {missing}"
        )

    probabilities = bundle["model"].predict_proba(
        ch[features]
    )[:, 1]

    thresholds = bundle["thresholds"]

    result = ch[required_columns].copy()

    result["p_chain_fraud"] = probabilities

    result["risk_chain"] = band(
        probabilities,
        thresholds
    )

    result["flag_chain"] = (
        probabilities >= thresholds["high"]
    ).astype(int)

    result["actual_chain_fraud"] = ch["label_chain_fraud"]

    result.to_csv(output_file, index=False)

    print(f"\nSaved: {output_file}")
    print(f"Total chain snapshots: {len(result):,}")
    print(f"Unique chains: {result['chain_id'].nunique():,}")

    print("\nRisk distribution:")
    print(result["risk_chain"].value_counts())

    print("\nSplit distribution:")
    print(result["split"].value_counts())

    return result


def main():

    output_txn = (
        sys.argv[1] if len(sys.argv) > 1
        else "predictions.csv"
    )

    output_chain = (
        sys.argv[2] if len(sys.argv) > 2
        else "chain_predictions.csv"
    )

    if not os.path.exists(C.TXN_FEATURES_CSV):
        raise SystemExit(
            f"ERROR: {C.TXN_FEATURES_CSV} not found. "
            "Run build_features.py first."
        )

    print("Loading transaction features...")

    tx = pd.read_csv(
        C.TXN_FEATURES_CSV,
        low_memory=False
    )

    print(f"Loaded {len(tx):,} transactions.")

    predict_transactions(tx, output_txn)

    predict_chains(output_chain)

    print("\n==========================================")
    print("ALL PREDICTIONS COMPLETED SUCCESSFULLY")
    print("==========================================")
    print(f"1. {output_txn}")
    print(f"2. {output_chain}")


if __name__ == "__main__":
    main()
