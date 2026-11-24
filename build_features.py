"""
Build strictly past-only transaction features.

Rule:
- A transaction at time t sees only events with timestamp < t.
- Events with the same timestamp are excluded from historical features.
- Missing receiver accounts have zero receiver-history features.

Run:
    venv\Scripts\python.exe build_features.py
"""

import numpy as np
import pandas as pd
import config as C


def prior_sum(df, key_cols, value):
    """Sum values from strictly earlier timestamps for each key."""
    tmp = df[key_cols + ["_ts", value]].copy()
    tmp["_cum"] = tmp.groupby(key_cols, sort=False, dropna=False)[value].cumsum()

    groups = tmp.groupby(key_cols + ["_ts"], sort=False, dropna=False)
    cum_at_timestamp = groups["_cum"].transform("last")
    value_at_timestamp = groups[value].transform("sum")

    return (cum_at_timestamp - value_at_timestamp).to_numpy()


def first_pair_flag(df, key_col, val_col):
    """1 for the first occurrence of a key-value pair, otherwise 0."""
    return (~df.duplicated([key_col, val_col])).astype(np.int8)


def trailing_count(keys, ts_sec, window):
    """
    Count same-account events in [current_time-window, current_time).
    Input must already be sorted by account and timestamp.
    """
    result = np.zeros(len(keys), dtype=np.int64)
    groups = pd.Series(np.arange(len(keys))).groupby(keys, sort=False)

    for _, positions in groups:
        idx = positions.to_numpy(dtype=np.int64)
        account_times = ts_sec[idx]

        left = np.searchsorted(
            account_times, account_times - window, side="left"
        )
        right = np.searchsorted(
            account_times, account_times, side="left"
        )
        result[idx] = right - left

    return result


def normalize_account(series):
    """Normalize account IDs and convert common missing-value strings to NA."""
    values = series.astype("string").str.strip().str.upper()
    return values.replace({
        "": pd.NA,
        "NAN": pd.NA,
        "NONE": pd.NA,
        "NULL": pd.NA,
        "<NA>": pd.NA,
    })


def main():
    # 1. Load source data
    df = pd.read_csv(C.SOURCE_CSV, low_memory=False)
    sc, rc = C.resolve_columns(df)
    print(f"Using sender='{sc}', receiver='{rc}'")

    required = [
        C.COL_TS, C.COL_AMOUNT, C.COL_TXN_ID,
        C.LABEL_TXN, C.LABEL_SENDER, C.LABEL_RECEIVER
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    # 2. Validate and normalize data
    df[C.COL_TS] = pd.to_datetime(
        df[C.COL_TS], utc=True, errors="coerce"
    )
    df[C.COL_AMOUNT] = pd.to_numeric(
        df[C.COL_AMOUNT], errors="coerce"
    )

    if df[C.COL_TS].isna().any():
        raise SystemExit("Invalid or missing timestamps found.")
    if df[C.COL_AMOUNT].isna().any():
        raise SystemExit("Invalid or missing transaction amounts found.")

    df["_sender"] = normalize_account(df[sc])
    df["_receiver"] = normalize_account(df[rc])
    df["_row"] = np.arange(len(df), dtype=np.int64)

    # 3. Current transaction flags
    local_time = df[C.COL_TS] + pd.Timedelta(
        minutes=C.NIGHT_UTC_OFFSET_MIN
    )
    hour = local_time.dt.hour

    df["_night"] = (
        (hour >= C.NIGHT_START_HOUR) |
        (hour < C.NIGHT_END_HOUR)
    ).astype(np.int8)

    df["_round"] = (
        df[C.COL_AMOUNT] % C.ROUND_AMOUNT_MULTIPLE == 0
    ).astype(np.int8)

    if C.COL_STATUS in df.columns:
        df["_failed"] = (
            df[C.COL_STATUS].astype("string").str.upper()
            == C.FAILED_VALUE
        ).astype(np.int8)
    else:
        df["_failed"] = np.zeros(len(df), dtype=np.int8)

    has_dev = C.COL_DEVICE in df.columns
    has_ip = C.COL_IP in df.columns

    # 4. Build account event table (one outgoing/incoming event per transaction)
    def make_side(account_col, role, counterparty_col):
        events = pd.DataFrame({
            "_row": df["_row"],
            "acct": df[account_col],
            "role": role,
            "_ts": df[C.COL_TS],
            "amt": df[C.COL_AMOUNT],
            "cp": df[counterparty_col],
            "night": df["_night"],
            "rnd": df["_round"],
            "failed": df["_failed"],
        })

        if has_dev and role == "out":
            events["dev"] = normalize_account(df[C.COL_DEVICE])
        else:
            events["dev"] = pd.NA

        if has_ip and role == "out":
            events["ip"] = normalize_account(df[C.COL_IP])
        else:
            events["ip"] = pd.NA

        return events

    events = pd.concat(
        [
            make_side("_sender", "out", "_receiver"),
            make_side("_receiver", "in", "_sender"),
        ],
        ignore_index=True,
    )

    # A missing account cannot have account-level history.
    events = events[events["acct"].notna()].copy()
    events = events.sort_values(
        ["acct", "_ts", "_row", "role"], kind="mergesort"
    ).reset_index(drop=True)

    events["amt_out"] = np.where(
        events["role"] == "out", events["amt"], 0.0
    )
    events["amt_in"] = np.where(
        events["role"] == "in", events["amt"], 0.0
    )
    events["is_out"] = (events["role"] == "out").astype(np.int8)
    events["is_in"] = (events["role"] == "in").astype(np.int8)

    # 5. Strictly-past account history
    key = ["acct"]
    history = {}

    history["in_cnt"] = prior_sum(events, key, "is_in")
    history["out_cnt"] = prior_sum(events, key, "is_out")
    history["tot_in"] = prior_sum(events, key, "amt_in")
    history["tot_out"] = prior_sum(events, key, "amt_out")

    total_count = history["in_cnt"] + history["out_cnt"]

    for feature_name, source_col in (
        ("night", "night"),
        ("round", "rnd"),
        ("failed", "failed"),
    ):
        values = prior_sum(events, key, source_col)
        history[f"{feature_name}_ratio"] = np.divide(
            values,
            total_count,
            out=np.zeros(len(events), dtype=float),
            where=total_count > 0,
        )

    # 6. Distinct counterparties seen before current timestamp
    for flag_name, role in (
        ("cp_new_out", "out"),
        ("cp_new_in", "in"),
    ):
        subset = events[
            (events["role"] == role) & events["cp"].notna()
        ]
        flags = pd.Series(0, index=events.index, dtype=np.int8)
        flags.loc[subset.index] = first_pair_flag(
            subset, "acct", "cp"
        ).to_numpy()
        events[flag_name] = flags

    history["distinct_receivers"] = prior_sum(
        events, key, "cp_new_out"
    )
    history["distinct_senders"] = prior_sum(
        events, key, "cp_new_in"
    )

    # 7. Distinct devices/IPs previously used by each sender account
    for feature_name, source_col, available in (
        ("devices", "dev", has_dev),
        ("ips", "ip", has_ip),
    ):
        if not available:
            continue

        subset = events[
            (events["role"] == "out") & events[source_col].notna()
        ]
        flag_col = f"new_{feature_name}"
        flags = pd.Series(0, index=events.index, dtype=np.int8)
        flags.loc[subset.index] = first_pair_flag(
            subset, "acct", source_col
        ).to_numpy()
        events[flag_col] = flags

        history[f"distinct_{feature_name}"] = prior_sum(
            events, key, flag_col
        )

    # 8. Account activity during the preceding hour
    epoch_seconds = (
        events["_ts"] - pd.Timestamp("1970-01-01", tz="UTC")
    ).dt.total_seconds().to_numpy(dtype=np.int64)

    account_codes = pd.factorize(events["acct"], sort=False)[0]
    history["txn_last_1h"] = trailing_count(
        account_codes, epoch_seconds, C.VELOCITY_WINDOW_SEC
    )

    # 9. Attach account history to transaction rows
    feature_names = list(history.keys())
    for name, values in history.items():
        events[name] = values

    sender_side = (
        events[events["role"] == "out"]
        .set_index("_row")[feature_names]
        .add_prefix("s_hist_")
    )
    receiver_side = (
        events[events["role"] == "in"]
        .set_index("_row")[feature_names]
        .add_prefix("r_hist_")
    )

    sender_side = sender_side[~sender_side.index.duplicated(keep="first")]
    receiver_side = receiver_side[
        ~receiver_side.index.duplicated(keep="first")
    ]

    # 10. Base output
    result = df[
        [C.COL_TXN_ID, C.COL_TS, "_sender", "_receiver", C.COL_AMOUNT]
    ].copy()
    result.columns = ["txn_id", "timestamp", "sender", "receiver", "amount"]
    result.index = df["_row"]

    optional_columns = [
        C.COL_CHAIN, "channel", "direction", "source_type",
        "app_name", "currency", C.COL_STATUS, "response_code",
        C.LABEL_TXN, C.LABEL_SENDER, C.LABEL_RECEIVER, C.LABEL_CHAIN,
    ]
    for col in optional_columns:
        if col in df.columns and col not in result.columns:
            result[col] = df[col].to_numpy()

    result = result.join(sender_side).join(receiver_side)

    # Missing receiver accounts have no receiver-side history.
    missing_receiver = result["receiver"].isna()
    receiver_history_columns = [
        col for col in result.columns if col.startswith("r_hist_")
    ]
    result.loc[missing_receiver, receiver_history_columns] = 0
    print("Missing receiver rows corrected:", int(missing_receiver.sum()))

    # 11. Strictly-past sender/receiver pair history
    ordered = df.sort_values(
        [C.COL_TS, "_row"], kind="mergesort"
    )
    pair = (
        ordered["_sender"].astype("string") + ">" +
        ordered["_receiver"].astype("string")
    )
    pair = pair.where(
        ordered["_sender"].notna() & ordered["_receiver"].notna()
    )

    pair_events = pd.DataFrame({
        "_row": ordered["_row"].to_numpy(),
        "_ts": ordered[C.COL_TS].to_numpy(),
        "pair": pair.to_numpy(),
        "one": np.ones(len(ordered), dtype=np.int8),
    })
    pair_events = pair_events[pair_events["pair"].notna()].copy()
    pair_events = pair_events.sort_values(
        ["pair", "_ts", "_row"], kind="mergesort"
    )

    pair_events["pair_prior_txn_count"] = prior_sum(
        pair_events, ["pair"], "one"
    )
    result["pair_prior_txn_count"] = (
        pair_events.set_index("_row")["pair_prior_txn_count"]
        .reindex(result.index)
        .fillna(0)
    )

    # 12. Previously seen sender accounts per device/IP
    for feature_name, source_col, available in (
        ("dev", C.COL_DEVICE, has_dev),
        ("ip", C.COL_IP, has_ip),
    ):
        if not available:
            continue

        shared = pd.DataFrame({
            "_row": df["_row"],
            "_ts": df[C.COL_TS],
            "key": normalize_account(df[source_col]),
            "acct": df["_sender"],
        })
        shared = shared[
            shared["key"].notna() & shared["acct"].notna()
        ].sort_values(
            ["key", "_ts", "_row"], kind="mergesort"
        )

        shared["new_acct"] = first_pair_flag(
            shared, "key", "acct"
        )
        shared["prior_accts"] = prior_sum(
            shared, ["key"], "new_acct"
        )

        result[f"{feature_name}_prior_distinct_accts"] = (
            shared.set_index("_row")["prior_accts"]
            .reindex(result.index)
            .fillna(0)
        )

    # 13. Current transaction time features
    result["hour_local"] = (
        df[C.COL_TS] +
        pd.Timedelta(minutes=C.NIGHT_UTC_OFFSET_MIN)
    ).dt.hour.to_numpy()
    result["dow"] = df[C.COL_TS].dt.day_of_week.to_numpy()
    result["log_amount"] = np.log1p(
        df[C.COL_AMOUNT].clip(lower=0)
    ).to_numpy()
    result["is_round_amount"] = df["_round"].to_numpy()
    result["is_night"] = df["_night"].to_numpy()

    # 14. Chronological split
    result["split"] = C.split_of(result["timestamp"]).to_numpy()

    # Final safety check: no receiver-history NaNs for missing receivers.
    if receiver_history_columns:
        remaining = int(
            result.loc[missing_receiver, receiver_history_columns]
            .isna().sum().sum()
        )
        if remaining:
            raise SystemExit(
                f"Receiver-history correction failed: {remaining} NaN values remain."
            )

    # 15. Save
    result.to_csv(C.TXN_FEATURES_CSV, index=False)

    print(f"\nWrote {C.TXN_FEATURES_CSV}: {len(result):,} rows")
    print("\nSplit counts:")
    print(result["split"].value_counts())
    sender_history = result["s_hist_in_cnt"] + result["s_hist_out_cnt"]
    print(
        "\nRows with zero prior sender history:",
        int((sender_history == 0).sum()),
    )
    print("\nFeature generation completed.")


if __name__ == "__main__":
    main()
