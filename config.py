
"""
Shared settings for the Suspicious Transaction Detection pipeline.
Edit ONLY this file if a column name differs in your dataset.
"""

import re

# ---------- files ----------
SOURCE_CSV = "training_dataset_original.csv"
TXN_FEATURES_CSV = "features_txn.csv"
CHAIN_SNAP_CSV = "features_chain_snapshots.csv"

MODELS_DIR = "models"
REPORTS_DIR = "reports"

# ---------- column names ----------
COL_TS = "timestamp"
COL_AMOUNT = "amount"

COL_SENDER = "sender_account"
COL_RECEIVER = "receiver_account"

COL_TXN_ID = "txn_id"
COL_CHAIN = "chain_id"
COL_HOP = "hop_index"

COL_DEVICE = "device_id"
COL_IP = "ip_address"
COL_STATUS = "status"

# ---------- labels ----------
LABEL_TXN = "label_transaction_fraud"
LABEL_SENDER = "label_sender_is_mule"
LABEL_RECEIVER = "label_receiver_is_mule"
LABEL_CHAIN = "label_chain_fraud"

# ---------- feature definitions ----------
NIGHT_UTC_OFFSET_MIN = 330

NIGHT_START_HOUR = 22
NIGHT_END_HOUR = 5

ROUND_AMOUNT_MULTIPLE = 100
VELOCITY_WINDOW_SEC = 3600

FAILED_VALUE = "FAILED"

# ---------- chronological split ----------
TRAIN_START = "2025-07-01"
TRAIN_END = "2025-11-01"

VAL_START = "2025-11-01"
VAL_END = "2025-12-01"

TEST_START = "2025-12-01"
TEST_END = "2026-01-01"

# ---------- model inputs ----------
USE_STATUS_AS_FEATURE = False

CAT_TXN = [
    "channel",
    "direction",
    "source_type",
    "app_name",
    "currency"
]

# ---------- threshold policy ----------
MEDIUM_RISK_MIN_RECALL = 0.90


def resolve_columns(df):

    required = {
        "COL_SENDER": COL_SENDER,
        "COL_RECEIVER": COL_RECEIVER
    }

    for key, column in required.items():
        if column not in df.columns:
            raise SystemExit(
                f"{key}: column '{column}' not found in dataset."
            )

    return COL_SENDER, COL_RECEIVER


def split_of(ts):
    """
    Return train / val / test / out
    based on UTC timestamps.
    """

    import pandas as pd

    s = pd.Series("out", index=ts.index, dtype=object)

    for name, a, b in (
        ("train", TRAIN_START, TRAIN_END),
        ("val", VAL_START, VAL_END),
        ("test", TEST_START, TEST_END)
    ):

        s[
            (ts >= pd.Timestamp(a, tz="UTC")) &
            (ts < pd.Timestamp(b, tz="UTC"))
        ] = name

    return s
