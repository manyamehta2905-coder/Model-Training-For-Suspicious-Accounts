
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =====================================================
# 1. PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Transaction Risk Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. DESIGN / CSS
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #F5F7FA;
    color: #243247;
}

[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E5EAF0;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

h1, h2, h3 {
    color: #172B4D;
    font-weight: 700;
}

p, label {
    color: #526176;
}

.hero {
    background: #FFFFFF;
    border: 1px solid #E6EBF1;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 22px;
}

.hero-tag {
    color: #168F86;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
}

.hero-title {
    color: #172B4D;
    font-size: 29px;
    font-weight: 700;
    margin-top: 7px;
}

.hero-subtitle {
    color: #758398;
    font-size: 14px;
    margin-top: 5px;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid #E6EBF1;
    border-radius: 12px;
    padding: 18px;
    min-height: 112px;
}

.metric-label {
    color: #718096;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .4px;
}

.metric-value {
    color: #172B4D;
    font-size: 26px;
    font-weight: 700;
    margin-top: 9px;
}

.metric-note {
    color: #8A97A8;
    font-size: 11px;
    margin-top: 4px;
}

.section-heading {
    color: #172B4D;
    font-size: 19px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 12px;
}

div[data-testid="stTabs"] button {
    font-weight: 600;
    color: #526176;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #168F86;
}

.stButton button, .stDownloadButton button {
    border-radius: 8px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #E6EBF1;
    border-radius: 10px;
}

hr {
    border-color: #E6EBF1;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. HELPER FUNCTIONS
# =====================================================

@st.cache_data
def load_csv(path):
    if Path(path).exists():
        return pd.read_csv(path, low_memory=False)
    return None


@st.cache_data
def load_metrics():
    path = Path("reports/metrics.json")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def numeric(df, col, default=0):
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)

    return pd.to_numeric(
        df[col], errors="coerce"
    ).fillna(default)


def money(value):
    return f"₹{value:,.0f}"


def card(label, value, note=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def chart_style(fig, height=340):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="DM Sans",
            color="#526176",
            size=12
        ),
        title_font=dict(
            color="#172B4D",
            size=17
        ),
        margin=dict(l=15, r=15, t=55, b=15),
        height=height,
        legend=dict(
            bgcolor="rgba(255,255,255,0)"
        )
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor="#E6EBF1"
    )
    fig.update_yaxes(
        gridcolor="#EEF1F5",
        zeroline=False
    )
    return fig


# =====================================================
# 4. LOAD DATA
# =====================================================

tx = load_csv("predictions.csv")
chain = load_csv("chain_predictions.csv")
metrics = load_metrics()


# =====================================================
# 5. HEADER
# =====================================================

st.markdown("""
<div class="hero">
    <div class="hero-tag">TRANSACTION MONITORING SYSTEM</div>
    <div class="hero-title">Suspicious Transaction Detection</div>
    <div class="hero-subtitle">
        ML-powered transaction risk analysis and suspicious
        money-flow monitoring
    </div>
</div>
""", unsafe_allow_html=True)


if tx is None:
    st.error("predictions.csv not found. Please run predict.py first.")
    st.stop()

if "timestamp" not in tx.columns:
    st.error("timestamp column is missing from predictions.csv.")
    st.stop()


# =====================================================
# 6. COLUMN MAPPING
# =====================================================

mapping = {
    "transaction_fraud_probability": "p_txn_fraud",
    "transaction_fraud_prediction": "flag_txn_fraud",
    "sender_mule_probability": "p_sender_mule",
    "sender_mule_prediction": "flag_sender_mule",
    "receiver_mule_probability": "p_receiver_mule",
    "receiver_mule_prediction": "flag_receiver_mule",
    "chain_fraud_probability": "p_chain_fraud",
    "chain_fraud_prediction": "flag_chain",
    "sender_account": "sender",
    "receiver_account": "receiver",
    "label_transaction_fraud": "actual_txn_fraud",
    "label_sender_is_mule": "actual_sender_mule",
    "label_receiver_is_mule": "actual_receiver_mule",
    "label_chain_fraud": "actual_chain_fraud"
}

for old, new in mapping.items():
    if old in tx.columns and new not in tx.columns:
        tx[new] = tx[old]


# =====================================================
# 7. RECOVER SPLIT
# =====================================================

if "split" not in tx.columns:
    features = load_csv("features_txn.csv")

    if (
        features is not None
        and "txn_id" in features.columns
        and "split" in features.columns
    ):
        split_map = features[
            ["txn_id", "split"]
        ].drop_duplicates("txn_id")

        tx = tx.merge(
            split_map,
            on="txn_id",
            how="left",
            validate="many_to_one"
        )
    else:
        st.error("Could not recover split from features_txn.csv.")
        st.stop()

if tx["split"].isna().all():
    st.error("Split mapping failed. Check transaction IDs.")
    st.stop()


# =====================================================
# 8. NORMALISE DATA
# =====================================================

for col in [
    "p_txn_fraud",
    "p_sender_mule",
    "p_receiver_mule"
]:
    if col not in tx.columns:
        tx[col] = 0.0
    tx[col] = numeric(tx, col).clip(0, 1)

for col in [
    "flag_txn_fraud",
    "flag_sender_mule",
    "flag_receiver_mule"
]:
    if col not in tx.columns:
        tx[col] = 0
    tx[col] = numeric(tx, col).astype(int)

for col in ["sender", "receiver", "channel", "txn_id"]:
    if col not in tx.columns:
        tx[col] = "Unknown"

tx["amount"] = numeric(tx, "amount")
tx["timestamp"] = pd.to_datetime(
    tx["timestamp"], utc=True, errors="coerce"
)


# =====================================================
# 9. SIDEBAR FILTERS
# =====================================================

st.sidebar.markdown("## Filters")
st.sidebar.caption("Customise transaction analysis")

available_splits = [
    x for x in ["test", "val", "train"]
    if x in tx["split"].astype(str).unique()
]

splits = st.sidebar.multiselect(
    "Data split",
    available_splits,
    default=["test"] if "test" in available_splits else available_splits
)

if any(x in splits for x in ["train", "val"]):
    st.sidebar.info("Train/validation results are not final out-of-sample results.")

channels = sorted(tx["channel"].dropna().astype(str).unique())

selected_channels = st.sidebar.multiselect(
    "Channel",
    channels,
    default=channels
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Risk settings")

high_threshold = st.sidebar.slider(
    "High risk threshold",
    0.0, 1.0, 0.80, 0.01
)

medium_threshold = st.sidebar.slider(
    "Medium risk threshold",
    0.0, 1.0, 0.40, 0.01
)

if medium_threshold > high_threshold:
    st.sidebar.error("Medium threshold must be lower than high threshold.")
    st.stop()

min_probability = st.sidebar.slider(
    "Minimum probability",
    0.0, 1.0, 0.0, 0.01
)

tx["risk_txn_fraud"] = np.select(
    [
        tx["p_txn_fraud"] >= high_threshold,
        tx["p_txn_fraud"] >= medium_threshold
    ],
    ["HIGH", "MEDIUM"],
    default="LOW"
)

risk_options = st.sidebar.multiselect(
    "Risk category",
    ["HIGH", "MEDIUM", "LOW"],
    default=["HIGH", "MEDIUM"]
)

valid_dates = tx["timestamp"].dropna()

date_range = None
if len(valid_dates):
    date_range = st.sidebar.date_input(
        "Date range",
        value=(valid_dates.min().date(), valid_dates.max().date()),
        min_value=valid_dates.min().date(),
        max_value=valid_dates.max().date()
    )


# =====================================================
# 10. FILTER DATA
# =====================================================

base = tx[
    tx["split"].astype(str).isin(splits)
].copy()

if selected_channels:
    base = base[
        base["channel"].astype(str).isin(selected_channels)
    ]
else:
    base = base.iloc[0:0]

if risk_options:
    base = base[
        base["risk_txn_fraud"].isin(risk_options)
    ]
else:
    base = base.iloc[0:0]

base = base[
    base["p_txn_fraud"] >= min_probability
]

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    base = base[
        (base["timestamp"].dt.date >= date_range[0])
        & (base["timestamp"].dt.date <= date_range[1])
    ]


# =====================================================
# 11. NOTICE
# =====================================================

st.caption(
    f"Showing {len(base):,} transactions based on selected filters."
)

st.warning(
    "Research prototype using synthetic data. "
    "Not validated for real-world banking decisions."
)


# =====================================================
# 12. TABS
# =====================================================

overview, alerts_tab, chains_tab, performance_tab = st.tabs([
    "Overview",
    "Transaction Alerts",
    "Chain Analysis",
    "Model Performance"
])


# =====================================================
# TAB 1: OVERVIEW
# =====================================================

with overview:

    st.markdown('<div class="section-heading">System Overview</div>',
                unsafe_allow_html=True)

    total = len(base)
    flagged = int(base["flag_txn_fraud"].sum())

    sender_count = base.loc[
        base["flag_sender_mule"] == 1, "sender"
    ].nunique()

    receiver_count = base.loc[
        base["flag_receiver_mule"] == 1, "receiver"
    ].nunique()

    flagged_value = base.loc[
        base["flag_txn_fraud"] == 1, "amount"
    ].sum()

    cols = st.columns(5)

    values = [
        ("Total Transactions", f"{total:,}", "Transactions in current view"),
        ("Flagged Transactions", f"{flagged:,}", "Predicted as fraud"),
        ("Sender Mule Accounts", f"{sender_count:,}", "Unique flagged senders"),
        ("Receiver Mule Accounts", f"{receiver_count:,}", "Unique flagged receivers"),
        ("Flagged Amount", money(flagged_value), "Value of flagged transactions")
    ]

    for col, item in zip(cols, values):
        with col:
            card(*item)

    st.markdown('<div class="section-heading">Risk Analytics</div>',
                unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        risk_data = (
            base["risk_txn_fraud"]
            .value_counts()
            .reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0)
            .rename_axis("Risk")
            .reset_index(name="Transactions")
        )

        fig = px.pie(
            risk_data,
            names="Risk",
            values="Transactions",
            hole=0.62,
            color="Risk",
            color_discrete_map={
                "HIGH": "#E56B6F",
                "MEDIUM": "#E8B45B",
                "LOW": "#49A88F"
            },
            title="Transaction Risk Distribution"
        )

        st.plotly_chart(
            chart_style(fig),
            use_container_width=True
        )

    with right:
        fig = px.histogram(
            base,
            x="p_txn_fraud",
            nbins=35,
            color_discrete_sequence=["#269D95"],
            title="Fraud Probability Distribution"
        )

        fig.update_layout(
            xaxis_title="Fraud Probability",
            yaxis_title="Number of Transactions"
        )

        st.plotly_chart(
            chart_style(fig),
            use_container_width=True
        )

    st.markdown('<div class="section-heading">Daily Alert Trend</div>',
                unsafe_allow_html=True)

    daily_base = base.dropna(subset=["timestamp"])

    if len(daily_base):
        daily = (
            daily_base.set_index("timestamp")
            .resample("D")["flag_txn_fraud"]
            .sum()
            .reset_index()
        )

        fig = px.line(
            daily,
            x="timestamp",
            y="flag_txn_fraud",
            markers=True,
            color_discrete_sequence=["#269D95"],
            title="Flagged Transactions Over Time"
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Flagged Transactions"
        )

        st.plotly_chart(
            chart_style(fig),
            use_container_width=True
        )
    else:
        st.info("No date information available for this selection.")


# =====================================================
# TAB 2: TRANSACTION ALERTS
# =====================================================

with alerts_tab:

    st.markdown('<div class="section-heading">Prioritised Alerts</div>',
                unsafe_allow_html=True)

    alerts = base.sort_values(
        "p_txn_fraud",
        ascending=False
    )

    display_cols = [
        "txn_id", "timestamp", "sender", "receiver",
        "amount", "channel", "p_txn_fraud",
        "risk_txn_fraud", "p_sender_mule",
        "p_receiver_mule"
    ]

    display_cols = [
        c for c in display_cols if c in alerts.columns
    ]

    st.dataframe(
        alerts[display_cols].head(1000),
        use_container_width=True,
        height=370,
        hide_index=True
    )

    st.download_button(
        "Download filtered alerts CSV",
        alerts[display_cols].to_csv(index=False).encode("utf-8"),
        "transaction_alerts.csv",
        "text/csv"
    )

    st.markdown('<div class="section-heading">Money Flow Analysis</div>',
                unsafe_allow_html=True)

    flow = alerts.head(25).copy()

    flow = flow[
        (flow["sender"].astype(str) != "Unknown")
        & (flow["receiver"].astype(str) != "Unknown")
    ]

    if len(flow):
        source = flow["sender"].astype(str)
        target = flow["receiver"].astype(str)

        nodes = list(pd.unique(
            pd.concat([source, target], ignore_index=True)
        ))

        indexes = {n: i for i, n in enumerate(nodes)}

        fig = go.Figure(go.Sankey(
            node=dict(
                label=nodes,
                pad=15,
                thickness=16,
                color="#269D95"
            ),
            link=dict(
                source=[indexes[x] for x in source],
                target=[indexes[x] for x in target],
                value=flow["amount"].clip(lower=1).tolist(),
                color="rgba(38,157,149,0.25)"
            )
        ))

        fig.update_layout(
            title="Money Flow of Highest-Risk Transactions",
            height=430
        )

        st.plotly_chart(
            chart_style(fig, 430),
            use_container_width=True
        )
    else:
        st.info("No money-flow records available.")

    st.markdown('<div class="section-heading">Transaction Details</div>',
                unsafe_allow_html=True)

    transaction_ids = alerts["txn_id"].astype(str).head(500).tolist()

    if transaction_ids:
        selected_id = st.selectbox(
            "Select Transaction ID",
            transaction_ids
        )

        selected = alerts[
            alerts["txn_id"].astype(str) == selected_id
        ].iloc[0]

        st.json(json.loads(selected.to_json()))


# =====================================================
# TAB 3: CHAIN ANALYSIS
# =====================================================

with chains_tab:

    st.markdown('<div class="section-heading">Money-Flow Chain Intelligence</div>',
                unsafe_allow_html=True)

    if chain is None:
        st.info("chain_predictions.csv not found.")
    else:

        chain_mapping = {
            "chain_fraud_probability": "p_chain_fraud",
            "chain_fraud_prediction": "flag_chain",
            "label_chain_fraud": "actual_chain_fraud"
        }

        for old, new in chain_mapping.items():
            if old in chain.columns and new not in chain.columns:
                chain[new] = chain[old]

        if "split" not in chain.columns:
            st.error("Chain prediction file has no split column.")
        elif "p_chain_fraud" not in chain.columns:
            st.error("Chain fraud probability column missing.")
        elif "chain_id" not in chain.columns or "hop_index" not in chain.columns:
            st.error("chain_id or hop_index missing.")
        else:

            cs = chain[
                chain["split"].astype(str).isin(splits)
            ].copy()

            cs["p_chain_fraud"] = numeric(
                cs, "p_chain_fraud"
            ).clip(0, 1)

            if len(cs):

                last = (
                    cs.sort_values("hop_index")
                    .groupby("chain_id")
                    .tail(1)
                )

                peak = (
                    cs.groupby("chain_id")["p_chain_fraud"]
                    .max()
                    .rename("Peak Risk")
                    .reset_index()
                )

                summary_cols = ["chain_id", "hop_index"]
                if "actual_chain_fraud" in last.columns:
                    summary_cols.append("actual_chain_fraud")

                summary = last[summary_cols].merge(
                    peak, on="chain_id", how="left"
                )

                st.dataframe(
                    summary.sort_values(
                        "Peak Risk", ascending=False
                    ),
                    use_container_width=True,
                    height=300,
                    hide_index=True
                )

                selected_chain = st.selectbox(
                    "Inspect chain",
                    summary["chain_id"].tolist()
                )

                one = cs[
                    cs["chain_id"] == selected_chain
                ].sort_values("hop_index")

                fig = px.line(
                    one,
                    x="hop_index",
                    y="p_chain_fraud",
                    markers=True,
                    color_discrete_sequence=["#269D95"],
                    title=f"Risk Development — Chain {selected_chain}"
                )

                fig.update_layout(
                    xaxis_title="Hop Number",
                    yaxis_title="Fraud Probability"
                )

                st.plotly_chart(
                    chart_style(fig),
                    use_container_width=True
                )

            else:
                st.info("No chain records for the selected split.")


# =====================================================
# TAB 4: MODEL PERFORMANCE
# =====================================================

with performance_tab:

    st.markdown('<div class="section-heading">Model Evaluation</div>',
                unsafe_allow_html=True)

    if not metrics:
        st.info("reports/metrics.json not found.")
    else:

        rows = []

        for name, result in metrics.items():
            if not result or "test" not in result:
                continue

            test = result["test"]

            rows.append({
                "Model": name,
                "Precision": test.get("precision"),
                "Recall": test.get("recall"),
                "F1 Score": test.get("f1"),
                "PR-AUC": test.get("pr_auc"),
                "ROC-AUC": test.get("roc_auc"),
                "Base Rate": test.get("base_rate")
            })

        if rows:

            performance = pd.DataFrame(rows)

            st.dataframe(
                performance,
                use_container_width=True,
                hide_index=True
            )

            chart_data = performance.melt(
                id_vars="Model",
                value_vars=[
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "PR-AUC"
                ],
                var_name="Metric",
                value_name="Score"
            )

            fig = px.bar(
                chart_data,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                range_y=[0, 1],
                title="Comparison of ML Models",
                color_discrete_sequence=[
                    "#269D95",
                    "#6C91C2",
                    "#E8B45B",
                    "#A18BC5"
                ]
            )

            st.plotly_chart(
                chart_style(fig, 430),
                use_container_width=True
            )

            st.caption(
                "Metrics are based on the held-out test split. "
                "Thresholds were selected on validation data."
            )

        else:
            st.warning("No model evaluation results found.")


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Suspicious Transaction Detection | "
    "Machine Learning Research Prototype | "
    "Synthetic Dataset"
)
