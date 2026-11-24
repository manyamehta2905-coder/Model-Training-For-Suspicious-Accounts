"""
Corrected training for the 4 tasks.

Run ONLY after build_features.py and build_chain_snapshots.py.
- Chronological split from config.py
- Explicit allow-lists of features
- Preprocessing fitted on TRAIN only
- Thresholds chosen on VALIDATION only
- TEST evaluated after fitting

Run:
    venv\Scripts\python.exe train_models.py
"""

import json
import os

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

import config as C


HIST = [
    "in_cnt", "out_cnt", "tot_in", "tot_out", "night_ratio",
    "round_ratio", "failed_ratio", "distinct_receivers",
    "distinct_senders", "distinct_devices", "distinct_ips",
    "txn_last_1h",
]

CURRENT = [
    "amount", "log_amount", "hour_local", "dow", "is_round_amount",
    "is_night", "pair_prior_txn_count", "dev_prior_distinct_accts",
    "ip_prior_distinct_accts",
]

CHAIN_FEATS = [
    "hop_index", "unique_accounts_so_far", "total_amount_so_far",
    "mean_amount_so_far", "max_amount_so_far", "first_amount",
    "last_amount", "last_to_first_amount_ratio", "duration_sec_so_far",
    "mean_gap_sec", "last_gap_sec", "max_fan_out_so_far",
    "max_fan_in_so_far", "has_cycle_so_far",
]

NEVER = (
    "label_", "meta_", "chain_id", "txn_id", "chain_total_hops",
    "chain_end_ts", "forward",
)


def feature_sets(cols):
    cats = [c for c in C.CAT_TXN if c in cols]
    if C.USE_STATUS_AS_FEATURE:
        cats += [c for c in ("status", "response_code") if c in cols]

    cur = [c for c in CURRENT if c in cols]
    sender = [f"s_hist_{h}" for h in HIST if f"s_hist_{h}" in cols]
    receiver = [f"r_hist_{h}" for h in HIST if f"r_hist_{h}" in cols]

    return {
        "transaction_fraud": (cur + sender + receiver, cats),
        "sender_mule": (cur + sender, cats),
        "receiver_mule": (cur + receiver, cats),
    }


def assert_clean(features):
    bad = [
        f for f in features
        if f.lower().startswith(NEVER)
        or any(token in f.lower() for token in NEVER)
    ]
    if bad:
        raise SystemExit(f"Forbidden predictors found: {bad}")


def make_model(num, cat, pos_weight):
    transformers = []
    if cat:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
        )

    preprocessor = ColumnTransformer(
        transformers,
        remainder="passthrough",
    )

    classifier = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
    )

    return Pipeline([
        ("pre", preprocessor),
        ("clf", classifier),
    ])


def pick_thresholds(y, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y, probabilities
    )

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    )
    high_threshold = float(thresholds[int(np.nanargmax(f1_values))])

    eligible = np.where(
        recall[:-1] >= C.MEDIUM_RISK_MIN_RECALL
    )[0]
    medium_threshold = (
        float(thresholds[eligible[-1]]) if len(eligible)
        else high_threshold
    )

    return min(medium_threshold, high_threshold), high_threshold


def evaluate(y, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    output = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "negatives": int((1 - y).sum()),
        "threshold": float(threshold),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "pr_auc": (
            float(average_precision_score(y, probabilities))
            if y.sum() else None
        ),
        "roc_auc": (
            float(roc_auc_score(y, probabilities))
            if 0 < y.sum() < len(y) else None
        ),
    }

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    output["confusion_matrix"] = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    output["base_rate"] = float(y.mean()) if len(y) else None

    return output


def run_task(
    name, train, validation, test, numeric, categorical, label,
    weights=None, group=None
):
    assert_clean(numeric + categorical)

    y_train = train[label].astype(int).to_numpy()
    y_validation = validation[label].astype(int).to_numpy()
    y_test = test[label].astype(int).to_numpy()

    if (
        y_train.sum() == 0
        or y_validation.sum() == 0
        or y_test.sum() == 0
    ):
        print(f"[{name}] skipped: a split has no positive examples")
        return None

    if weights is not None:
        weights = np.asarray(weights, dtype=float)

        if len(weights) != len(train):
            raise ValueError(
                f"[{name}] Weight count ({len(weights)}) does not "
                f"match training rows ({len(train)})."
            )

        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError(
                f"[{name}] Sample weights must all be finite and positive. "
                f"Min={np.nanmin(weights)}, NaNs={np.isnan(weights).sum()}"
            )

    positive_weight = (
        (len(y_train) - y_train.sum()) / y_train.sum()
    )

    model = make_model(numeric, categorical, positive_weight)
    model.fit(
        train[numeric + categorical],
        y_train,
        clf__sample_weight=weights,
    )

    validation_probabilities = model.predict_proba(
        validation[numeric + categorical]
    )[:, 1]

    medium_threshold, high_threshold = pick_thresholds(
        y_validation, validation_probabilities
    )

    test_probabilities = model.predict_proba(
        test[numeric + categorical]
    )[:, 1]

    result = {
        "features_numeric": numeric,
        "features_categorical": categorical,
        "label": label,
        "thresholds": {
            "medium": medium_threshold,
            "high": high_threshold,
        },
        "validation": evaluate(
            y_validation, validation_probabilities, high_threshold
        ),
        "test": evaluate(
            y_test, test_probabilities, high_threshold
        ),
    }

    if group is not None:
        account_scores = pd.DataFrame({
            "g": test[group].to_numpy(),
            "y": y_test,
            "p": test_probabilities,
        }).groupby("g").agg(
            y=("y", "max"),
            p=("p", "max"),
        )

        result["test_account_level"] = evaluate(
            account_scores["y"].to_numpy(),
            account_scores["p"].to_numpy(),
            high_threshold,
        )

    try:
        importances = model.named_steps["clf"].feature_importances_
        names = model.named_steps["pre"].get_feature_names_out()
        result["top_features"] = [
            [str(name), float(value)]
            for name, value in sorted(
                zip(names, importances),
                key=lambda item: -item[1],
            )[:10]
        ]
    except Exception:
        pass

    if (
        result["test"]["roc_auc"] is not None
        and result["test"]["roc_auc"] > 0.99
    ):
        print(
            f"WARNING [{name}]: test ROC-AUC > 0.99 -- "
            "check for leakage before trusting this."
        )

    joblib.dump(
        {
            "model": model,
            "numeric": numeric,
            "categorical": categorical,
            "thresholds": result["thresholds"],
        },
        os.path.join(C.MODELS_DIR, f"{name}.joblib"),
    )

    test_metrics = result["test"]
    print(
        f"[{name}] val-thr={high_threshold:.3f} | "
        f"TEST P={test_metrics['precision']:.3f} "
        f"R={test_metrics['recall']:.3f} "
        f"F1={test_metrics['f1']:.3f} "
        f"PR-AUC={test_metrics['pr_auc']:.3f} "
        f"(base rate {test_metrics['base_rate']:.3f})"
    )

    return result


def account_sample_weights(train, account_column):
    """
    Give each account equal total training weight.
    All rows must have a non-missing account ID.
    """
    counts = train[account_column].map(
        train[account_column].value_counts()
    )

    weights = (1.0 / counts).to_numpy(dtype=float)

    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError(
            f"Invalid sample weights for {account_column}. "
            "Check missing account IDs in the training split."
        )

    return weights / weights.mean()


def main():
    os.makedirs(C.MODELS_DIR, exist_ok=True)
    os.makedirs(C.REPORTS_DIR, exist_ok=True)
    os.makedirs("train_test_data", exist_ok=True)

    transactions = pd.read_csv(
        C.TXN_FEATURES_CSV,
        low_memory=False,
    )

    feature_map = feature_sets(set(transactions.columns))
    split_parts = {
        split: transactions[transactions["split"] == split].copy()
        for split in ("train", "val", "test")
    }

    print({
        split: len(data)
        for split, data in split_parts.items()
    })

    for split, data in split_parts.items():
        data[["txn_id", "timestamp"]].to_csv(
            f"train_test_data/{split}_txn_ids.csv",
            index=False,
        )

    results = {}

    # Transaction, sender, and receiver classification tasks.
    tasks = (
        ("transaction_fraud", C.LABEL_TXN, None),
        ("sender_mule", C.LABEL_SENDER, "sender"),
        ("receiver_mule", C.LABEL_RECEIVER, "receiver"),
    )

    for task_name, label, account_column in tasks:
        numeric, categorical = feature_map[task_name]

        # Receiver-mule prediction is undefined when the transaction
        # has no receiver account. Exclude those rows from ALL splits.
        task_parts = {
            split: data.copy()
            for split, data in split_parts.items()
        }

        if account_column is not None:
            for split in ("train", "val", "test"):
                before = len(task_parts[split])
                task_parts[split] = task_parts[split][
                    task_parts[split][account_column].notna()
                ].copy()
                removed = before - len(task_parts[split])

                if removed:
                    print(
                        f"  [{task_name}] excluded {removed:,} rows "
                        f"with missing {account_column} in {split}"
                    )

        weights = None
        if account_column is not None:
            label_counts = (
                transactions.groupby(account_column)[label].nunique()
            )
            inconsistent = int((label_counts > 1).sum())

            if inconsistent:
                print(
                    f"  note [{task_name}]: {inconsistent} accounts "
                    "have a label that changes across rows"
                )

            weights = account_sample_weights(
                task_parts["train"], account_column
            )

        results[task_name] = run_task(
            task_name,
            task_parts["train"],
            task_parts["val"],
            task_parts["test"],
            numeric,
            categorical,
            label,
            weights=weights,
            group=account_column,
        )

    # Developing-chain classification task.
    chain_data = pd.read_csv(
        C.CHAIN_SNAP_CSV,
        low_memory=False,
    )
    print(
        "Chain snapshots by split:",
        chain_data["split"].value_counts().to_dict(),
    )

    chain_parts = {
        split: chain_data[chain_data["split"] == split].copy()
        for split in ("train", "val", "test")
    }

    # Rows without a chain ID cannot receive a chain-level sample weight.
    for split in ("train", "val", "test"):
        before = len(chain_parts[split])
        chain_parts[split] = chain_parts[split][
            chain_parts[split]["chain_id"].notna()
        ].copy()
        removed = before - len(chain_parts[split])

        if removed:
            print(
                f"  [developing_chain] excluded {removed:,} rows "
                f"with missing chain_id in {split}"
            )

    chain_weights = (
        1.0 / chain_parts["train"]
        .groupby("chain_id")["hop_index"]
        .transform("count")
    ).to_numpy(dtype=float)
    chain_weights = chain_weights / chain_weights.mean()

    results["developing_chain"] = run_task(
        "developing_chain",
        chain_parts["train"],
        chain_parts["val"],
        chain_parts["test"],
        [c for c in CHAIN_FEATS if c in chain_data.columns],
        [],
        "label_chain_fraud",
        weights=chain_weights,
        group="chain_id",
    )

    with open(
        os.path.join(C.REPORTS_DIR, "metrics.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2)

    comparison_rows = []
    for model_name, model_result in results.items():
        if model_result:
            comparison_rows.append({
                "model": model_name,
                **{
                    f"test_{metric}": model_result["test"][metric]
                    for metric in (
                        "precision", "recall", "f1", "pr_auc", "roc_auc"
                    )
                },
                "threshold": model_result["thresholds"]["high"],
                "test_positives": model_result["test"]["positives"],
            })

    pd.DataFrame(comparison_rows).to_csv(
        os.path.join(C.REPORTS_DIR, "model_comparison.csv"),
        index=False,
    )

    print(
        "Saved models/, reports/metrics.json, "
        "reports/model_comparison.csv"
    )
    print(
        "Reminder: results are on SYNTHETIC data -- "
        "prototype results only."
    )


if __name__ == "__main__":
    main()
