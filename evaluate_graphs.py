
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

os.makedirs("reports/figures", exist_ok=True)

OUTPUT = "reports/figures"

tasks = [
    {
        "name": "Transaction Fraud",
        "file": "predictions.csv",
        "prob": "p_txn_fraud",
        "flag": "flag_txn_fraud",
        "actual": "actual_txn_fraud",
        "base_rate": 0.048
    },
    {
        "name": "Sender Mule",
        "file": "predictions.csv",
        "prob": "p_sender_mule",
        "flag": "flag_sender_mule",
        "actual": "actual_sender_mule",
        "base_rate": 0.093
    },
    {
        "name": "Receiver Mule",
        "file": "predictions.csv",
        "prob": "p_receiver_mule",
        "flag": "flag_receiver_mule",
        "actual": "actual_receiver_mule",
        "base_rate": 0.081
    },
    {
        "name": "Developing Chain",
        "file": "chain_predictions.csv",
        "prob": "p_chain_fraud",
        "flag": "flag_chain",
        "actual": "actual_chain_fraud",
        "base_rate": 0.552
    }
]

results = []


# --------------------------------------------------
# HELPER: SAVE FIGURE
# --------------------------------------------------

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT, filename),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()


# --------------------------------------------------
# EVALUATE EACH MODEL ON TEST DATA ONLY
# --------------------------------------------------

for task in tasks:

    print("\nEvaluating:", task["name"])

    if not os.path.exists(task["file"]):
        print("File missing:", task["file"])
        continue

    df = pd.read_csv(task["file"], low_memory=False)

    required = [
        task["prob"],
        task["actual"],
        task["flag"]
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        print("Missing columns:", missing)
        continue

    # IMPORTANT: only unseen test data
    if "split" in df.columns:
        df = df[df["split"].astype(str) == "test"].copy()
    else:
        print("No split column. Cannot verify test-only results.")
        continue

    df[task["prob"]] = pd.to_numeric(
        df[task["prob"]], errors="coerce"
    )

    df[task["actual"]] = pd.to_numeric(
        df[task["actual"]], errors="coerce"
    )

    df[task["flag"]] = pd.to_numeric(
        df[task["flag"]], errors="coerce"
    )

    df = df.dropna(subset=required)

    if df.empty:
        print("No valid test rows.")
        continue

    y_true = df[task["actual"]].astype(int).to_numpy()
    y_pred = df[task["flag"]].astype(int).to_numpy()
    y_prob = df[task["prob"]].to_numpy()

    if len(np.unique(y_true)) < 2:
        print("Only one class found. Skipping curves.")
        continue

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    accuracy = np.mean(y_true == y_pred)

    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    results.append({
        "Model": task["name"],
        "Test Rows": len(df),
        "Fraud Base Rate": y_true.mean(),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "PR-AUC": pr_auc,
        "ROC-AUC": roc_auc
    })

    print(
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f} | "
        f"PR-AUC: {pr_auc:.3f}"
    )

    safe_name = (
        task["name"].lower().replace(" ", "_")
    )

    # --------------------------------------------------
    # GRAPH 1: CONFUSION MATRIX
    # --------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-Fraud", "Fraud"]
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax.set_title(
        f"{task['name']} - Confusion Matrix",
        fontsize=13,
        fontweight="bold"
    )

    save_plot(f"{safe_name}_confusion_matrix.png")

    # --------------------------------------------------
    # GRAPH 2: PRECISION-RECALL CURVE
    # --------------------------------------------------

    p_values, r_values, thresholds = precision_recall_curve(
        y_true,
        y_prob
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        r_values,
        p_values,
        linewidth=2.5,
        label=f"PR-AUC = {pr_auc:.3f}"
    )

    ax.axhline(
        y=task["base_rate"],
        linestyle="--",
        label="Fraud base rate"
    )

    ax.set_title(
        f"{task['name']} - Precision Recall Curve",
        fontweight="bold"
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()

    save_plot(f"{safe_name}_pr_curve.png")

    # --------------------------------------------------
    # GRAPH 3: PREDICTED PROBABILITY DISTRIBUTION
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(
        y_prob[y_true == 0],
        bins=25,
        alpha=0.65,
        label="Actual Non-Fraud",
        density=True
    )

    ax.hist(
        y_prob[y_true == 1],
        bins=25,
        alpha=0.65,
        label="Actual Fraud",
        density=True
    )

    ax.set_title(
        f"{task['name']} - Probability Distribution",
        fontweight="bold"
    )

    ax.set_xlabel("Predicted Fraud Probability")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.2)

    save_plot(f"{safe_name}_probability_distribution.png")


# --------------------------------------------------
# GRAPH 4: MODEL COMPARISON
# --------------------------------------------------

if results:

    metrics_df = pd.DataFrame(results)

    metrics_df.to_csv(
        "reports/test_model_metrics.csv",
        index=False
    )

    comparison_metrics = [
        "Precision",
        "Recall",
        "F1",
        "PR-AUC"
    ]

    chart_data = metrics_df.set_index("Model")[
        comparison_metrics
    ]

    ax = chart_data.plot(
        kind="bar",
        figsize=(11, 6),
        width=0.75
    )

    ax.set_title(
        "Model Performance Comparison - Test Set",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    plt.xticks(rotation=15, ha="right")

    save_plot("model_comparison.png")

    print("\n========== TEST METRICS ==========")
    print(metrics_df.round(3).to_string(index=False))

    print("\nMetrics saved: reports/test_model_metrics.csv")


# --------------------------------------------------
# GRAPH 5: DEVELOPING CHAIN RISK BY HOP
# --------------------------------------------------

if os.path.exists("chain_predictions.csv"):

    chain = pd.read_csv(
        "chain_predictions.csv",
        low_memory=False
    )

    if all(
        c in chain.columns
        for c in ["split", "hop_index", "p_chain_fraud"]
    ):

        chain = chain[
            chain["split"].astype(str) == "test"
        ].copy()

        chain["hop_index"] = pd.to_numeric(
            chain["hop_index"], errors="coerce"
        )

        chain["p_chain_fraud"] = pd.to_numeric(
            chain["p_chain_fraud"], errors="coerce"
        )

        chain = chain.dropna(
            subset=["hop_index", "p_chain_fraud"]
        )

        if not chain.empty:

            hop_data = (
                chain.groupby("hop_index")["p_chain_fraud"]
                .mean()
                .reset_index()
            )

            fig, ax = plt.subplots(figsize=(8, 5))

            ax.plot(
                hop_data["hop_index"],
                hop_data["p_chain_fraud"],
                marker="o",
                linewidth=2.5
            )

            ax.set_title(
                "Average Predicted Chain Fraud Risk by Hop",
                fontweight="bold"
            )

            ax.set_xlabel("Hop Number")
            ax.set_ylabel("Average Predicted Fraud Probability")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)

            save_plot("chain_risk_by_hop.png")


print("\n====================================")
print("GRAPH GENERATION COMPLETED")
print("Check reports/figures/")
print("====================================")
