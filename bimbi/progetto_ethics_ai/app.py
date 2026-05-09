"""
app.py
~~~~~~
Streamlit front-end for the Ethical CDSS – child abuse anomaly detection.

Data flow (real-data mode):
  data/windows.parquet  ──→  patient selector + fairness analysis
  data/kde_model.joblib ──→  EthicalKDEAnomalyDetector (pre-trained)
  data/prior_rules.json ──→  Italian text rules panel + symbolic penalties

Run with:
    streamlit run app.py

Prerequisites:
    Execute notebooks/01_data_exploration.ipynb first to generate the three
    artifact files in data/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from util.ethics_metrics import CostModel, calculate_di, calculate_spd
from util.models import EthicalKDEAnomalyDetector

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
DATA_DIR = _HERE / "data"
MODEL_PATH  = DATA_DIR / "kde_model.joblib"
PARQUET_PATH = DATA_DIR / "windows.parquet"
RULES_PATH  = DATA_DIR / "prior_rules.json"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Ethical CDSS – Child Abuse Detection",
    page_icon="hospital",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading patient windows...")
def load_windows() -> pd.DataFrame:
    """
    Read the pre-computed sliding-window DataFrame from Parquet.

    The Parquet file is written via pa.Table.from_arrays() with explicit
    primitive types to avoid PyArrow extension-type registration conflicts.
    As a result, datetime columns (DATAORA_ACCETTAZIONE) are stored as ISO
    8601 strings and are converted back to datetime64 here on load.
    """
    df = pd.read_parquet(PARQUET_PATH)
    time_col = "DATAORA_ACCETTAZIONE"
    if time_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    return df


@st.cache_resource(show_spinner="Loading KDE model...")
def load_detector() -> EthicalKDEAnomalyDetector:
    """Deserialise the fitted EthicalKDEAnomalyDetector from disk."""
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Loading clinical rules...")
def load_rules() -> list[dict]:
    """Read the Italian prior-knowledge rules from JSON."""
    with open(RULES_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner="Computing anomaly scores for all windows...")
def compute_all_scores(parquet_mtime: float) -> pd.Series:
    """
    Return the pre-computed ``anomaly_score`` column from the Parquet file,
    or recompute on the fly if that column is absent.

    ``parquet_mtime`` is passed purely to invalidate the Streamlit cache when
    the Parquet file is replaced (e.g., after re-running the notebook).
    """
    window_df = load_windows()
    if "anomaly_score" in window_df.columns:
        return window_df["anomaly_score"]
    detector = load_detector()
    return detector.get_anomaly_signal(window_df)


def _parquet_mtime() -> float:
    """Return the modification timestamp of the Parquet file (for cache keying)."""
    try:
        return PARQUET_PATH.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def _data_ready() -> bool:
    return MODEL_PATH.exists() and PARQUET_PATH.exists() and RULES_PATH.exists()


# ---------------------------------------------------------------------------
# Helper – reconstruct per-visit timeline from a flattened window row
# ---------------------------------------------------------------------------

def _reconstruct_timeline(window_row: pd.Series, window_len: int = 3) -> pd.DataFrame:
    """
    Unpack a flattened window row back into individual visit rows.

    Looks for columns ending in ``_t``, ``_t-1``, ``_t-2`` (newest→oldest)
    and reconstructs one dict per visit, preserving all base column names.
    """
    visits: list[dict] = []
    for i in range(window_len):
        suffix = "_t" if i == 0 else f"_t-{i}"
        visit: dict = {"visit": f"t{'+0' if i == 0 else f'-{i}'}"}
        for col_name in window_row.index:
            if col_name.endswith(suffix):
                base = col_name[: -len(suffix)]
                visit[base] = window_row[col_name]
        if len(visit) > 1:  # more than just the 'visit' key
            visits.append(visit)
    return pd.DataFrame(visits) if visits else pd.DataFrame()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("System Parameters")

    alert_threshold = st.slider(
        "Alert Threshold",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.1,
        help="Anomaly signal >= this value triggers a clinical alert.",
    )

    cost_alarm = st.slider(
        "Cost of False Alarm (FP)",
        min_value=1,
        max_value=20,
        value=1,
        step=1,
        help="Investigative cost per false positive.",
    )

    cost_missed = st.slider(
        "Cost of Missed Abuse (FN)",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
        help="Ethical/social cost per missed abuse case.",
    )

    st.divider()

    if _data_ready():
        det = load_detector()
        st.caption(
            f"Model: Gaussian KDE  \n"
            f"Bandwidth: {det.bandwidth}  \n"
            f"Prior rules: {len(det.prior_rules)}  \n"
            f"Fitted: {det._fitted}"
        )
    else:
        st.warning("Artifacts not found. Run the training notebook first.", icon="warning")

    st.divider()
    st.caption("Project: Ethics AI – CDSS v0.2 (real data)")

# ---------------------------------------------------------------------------
# Setup guard — show instructions until artifacts exist
# ---------------------------------------------------------------------------

if not _data_ready():
    st.error(
        "**Required data artifacts are missing.**\n\n"
        "Please run the training notebook first:\n\n"
        "```\n"
        "jupyter notebook notebooks/01_data_exploration.ipynb\n"
        "```\n\n"
        "The notebook will create:\n"
        "- `data/kde_model.joblib`\n"
        "- `data/windows.parquet`\n"
        "- `data/prior_rules.json`"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Load artifacts (cached after first call)
# ---------------------------------------------------------------------------

window_df = load_windows()
detector  = load_detector()
rules     = load_rules()
all_scores = compute_all_scores(_parquet_mtime())

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_dashboard, tab_xai, tab_fairness = st.tabs([
    "CDSS Dashboard",
    "Explainability (XAI)",
    "Fairness Assessment",
])

# ── Tab 1: Dashboard ─────────────────────────────────────────────────────────

with tab_dashboard:
    st.header("CDSS Dashboard – Patient Anomaly Screening")

    # ── Patient Selector ──────────────────────────────────────────────────────
    patient_ids = sorted(window_df["ID_PAZIENTE"].unique())
    col_sel, col_info = st.columns([2, 1])

    with col_sel:
        selected_patient = st.selectbox(
            "Select Patient ID",
            options=patient_ids,
            help="Choose a patient from the loaded dataset to screen.",
        )

    # Retrieve the most recent window for the selected patient.
    patient_windows = window_df[window_df["ID_PAZIENTE"] == selected_patient]
    patient_row_df  = patient_windows.sort_values("DATAORA_ACCETTAZIONE").iloc[[-1]]
    patient_row     = patient_row_df.iloc[0]

    with col_info:
        st.metric("Windows available", len(patient_windows))
        anchor_time = patient_row.get("DATAORA_ACCETTAZIONE", "N/A")
        st.metric("Most recent visit", str(anchor_time)[:10] if anchor_time != "N/A" else "N/A")

    st.divider()

    # ── Run Detection ─────────────────────────────────────────────────────────
    run_btn = st.button("Run Detection", type="primary")

    col_left, col_right = st.columns([2, 1])

    if run_btn:
        # Pass a single-row DataFrame — get_anomaly_signal requires pd.DataFrame.
        signal = detector.get_anomaly_signal(patient_row_df)
        score  = float(signal.iloc[0])

        with col_left:
            st.subheader("Anomaly Signal")
            c_score, c_thresh = st.columns(2)
            c_score.metric("Anomaly Score", f"{score:.4f}")
            c_thresh.metric("Alert Threshold", f"{alert_threshold:.1f}")

            if score >= alert_threshold:
                st.error(
                    "**ALERT** – Anomaly score exceeds threshold. "
                    "Clinical safeguarding review is recommended.",
                )
            else:
                st.success("Score below threshold. No immediate alert.")

            # Bar chart: numeric features from the most recent visit (_t columns).
            numeric_t_cols = [
                c for c in patient_row_df.columns
                if c.endswith("_t")
                and pd.api.types.is_numeric_dtype(patient_row_df[c])
            ]
            if numeric_t_cols:
                feature_vals = patient_row_df[numeric_t_cols].iloc[0]
                display_labels = [c.replace("_t", "") for c in numeric_t_cols]
                median_val = feature_vals.median()
                colors = [
                    "#e74c3c" if v > median_val else "#3498db"
                    for v in feature_vals
                ]
                fig, ax = plt.subplots(figsize=(7, max(3, len(numeric_t_cols) * 0.4)))
                ax.barh(display_labels, feature_vals, color=colors)
                ax.set_xlabel("Feature Value (most recent visit)")
                ax.set_title("Patient Feature Profile – Window Anchor t")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        with col_right:
            st.subheader("Prior Rules Status")
            st.caption("Italian clinical conditions matched against patient record.")
            for rule in rules:
                condition_text = str(rule.get("clinical_conditions", "")).strip().lower()
                # Replicate _evaluate_rule logic: case-insensitive substring search
                # over all string-typed cells in the patient row.
                matched = (
                    bool(condition_text)
                    and any(
                        isinstance(v, str) and condition_text in v.lower()
                        for v in patient_row.values
                    )
                )
                icon = "red_circle" if matched else "large_green_circle"
                label = "MATCHED" if matched else "not matched"
                st.markdown(
                    f"**[{rule.get('rule_id', '?')}]** "
                    f"`{rule.get('clinical_conditions', '')}` – {label}  \n"
                    f"penalty +{rule.get('penalty_weight', 0.0):.1f} | "
                    f"{rule.get('description', '')}"
                )
                st.markdown("---")
    else:
        with col_left:
            st.info("Select a patient above and press **Run Detection**.")

    # ── Cost Optimisation ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Threshold Optimisation via Asymmetric Cost Model")

    scores_arr = all_scores.values.astype(float)
    # Without ground-truth labels, use the top 10% of scores as the "suspected" class.
    heuristic_labels = (scores_arr >= np.percentile(scores_arr, 90)).astype(int)

    cost_model = CostModel(cost_alarm=cost_alarm, cost_missed=cost_missed)
    best = cost_model.optimize_threshold(scores_arr, heuristic_labels)
    df_costs = best["all_results"]

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.plot(df_costs["threshold"], df_costs["total_cost"], lw=2, color="#2c3e50")
    ax2.axvline(
        best["threshold"], color="#e74c3c", linestyle="--",
        label=f"Optimal theta = {best['threshold']:.2f}",
    )
    ax2.axvline(
        alert_threshold, color="#f39c12", linestyle=":",
        label=f"Current threshold = {alert_threshold:.2f}",
    )
    ax2.set_xlabel("Decision Threshold")
    ax2.set_ylabel("Total Cost")
    ax2.set_title("Cost Landscape across Thresholds (top-10% heuristic labels)")
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    c1, c2, c3 = st.columns(3)
    c1.metric("Optimal Threshold", f"{best['threshold']:.3f}")
    c2.metric("Minimum Total Cost", f"{best['total_cost']:.1f}")
    c3.metric("Recall at Optimal", f"{best['recall']:.2%}")

# ── Tab 2: Explainability (XAI) ───────────────────────────────────────────────

with tab_xai:
    st.header("Explainability – Defeasible Reasoning Report")

    # Reuse patient selected in Tab 1 (selectbox persists within session).
    st.subheader(f"Patient {selected_patient} – Visit Timeline")

    timeline_df = _reconstruct_timeline(patient_row, window_len=3)
    if not timeline_df.empty:
        with st.expander("Reconstructed Window Timeline (newest → oldest)", expanded=True):
            st.dataframe(timeline_df, use_container_width=True)
    else:
        st.info("Could not reconstruct timeline from window columns.")

    # Compute score for the selected patient (without re-running the full pipeline).
    signal_xai = detector.get_anomaly_signal(patient_row_df)
    score_xai  = float(signal_xai.iloc[0])
    st.metric("Anomaly Score", f"{score_xai:.4f}")

    # Identify which text rules fired.
    triggered_rules = []
    for rule in rules:
        condition_text = str(rule.get("clinical_conditions", "")).strip().lower()
        if condition_text and any(
            isinstance(v, str) and condition_text in v.lower()
            for v in patient_row.values
        ):
            triggered_rules.append(rule)

    if triggered_rules:
        st.warning(
            f"{len(triggered_rules)} Italian clinical rule(s) triggered "
            f"for this patient window."
        )
        for r in triggered_rules:
            st.markdown(
                f"- **[{r['rule_id']}]** `{r['clinical_conditions']}` – "
                f"{r['description']} *(+{r['penalty_weight']:.1f} penalty)*"
            )
    else:
        st.info("No Italian text rules matched this patient window.")

    st.divider()
    st.subheader("LLM Defeasible Report")

    if st.button("Generate LLM Report (requires OPENAI_API_KEY)"):
        if not os.getenv("OPENAI_API_KEY"):
            st.warning("OPENAI_API_KEY is not set. Cannot query the LLM.")
        else:
            from util.llm_wrapper import generate_xai_report
            # Build a visit list from the reconstructed timeline for the LLM.
            timeline_records = (
                timeline_df.to_dict("records") if not timeline_df.empty
                else [{"note": "Window data not available"}]
            )
            with st.spinner("Querying LLM..."):
                report = generate_xai_report(
                    patient_timeline=timeline_records,
                    anomaly_score=score_xai,
                    triggered_rules=triggered_rules,
                )
            st.markdown(report)
    else:
        st.caption(
            "Press the button above to generate a defeasible clinical explanation "
            "via the OpenAI API. An example report structure is shown in the project README."
        )

# ── Tab 3: Fairness Assessment ────────────────────────────────────────────────

with tab_fairness:
    st.header("Fairness Assessment – Demographic Parity")
    st.markdown(
        "Evaluates whether the anomaly detector flags patients at equal rates "
        "across demographic groups (Statistical Parity / Disparate Impact)."
    )

    # ── Detect available demographic column ───────────────────────────────────
    # Sliding windows suffix columns as _t; try the most recent visit's value.
    DEMO_CANDIDATES = ["SESSO_t", "NAZIONALITA_t", "SESSO", "NAZIONALITA"]
    demo_col = next((c for c in DEMO_CANDIDATES if c in window_df.columns), None)

    if demo_col is None:
        st.warning(
            "No demographic column found in windows.parquet. "
            "Ensure SESSO or NAZIONALITA is present in the raw data and that "
            "create_sliding_windows is called with numeric_only=False."
        )
        st.stop()

    # Drop rows where the demographic column is NaN.
    demo_series = window_df[demo_col].fillna("Unknown")
    group_counts = demo_series.value_counts()
    st.caption(
        f"Demographic column: **{demo_col}** — "
        f"{group_counts.to_dict()}"
    )

    # ── Compute binary predictions using the current alert threshold ──────────
    scores_fair = all_scores.values.astype(float)
    predictions = (scores_fair >= alert_threshold).astype(int)
    groups = demo_series.values

    # Use the two most prevalent groups for the parity calculation.
    top_two = group_counts.index[:2].tolist()
    if len(top_two) < 2:
        st.warning("Need at least two distinct demographic groups for parity analysis.")
        st.stop()

    privileged_group   = top_two[0]
    unprivileged_group = top_two[1]

    # Filter to the two groups (ignore Others for cleaner metrics).
    mask_two = np.isin(groups, top_two)
    pred_two   = predictions[mask_two]
    groups_two = groups[mask_two]
    true_labels_placeholder = np.zeros(len(pred_two), dtype=int)  # unsupervised

    spd = calculate_spd(
        true_labels_placeholder, pred_two, groups_two,
        privileged_group, unprivileged_group,
    )
    di = calculate_di(
        true_labels_placeholder, pred_two, groups_two,
        privileged_group, unprivileged_group,
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric(
        "Statistical Parity Difference",
        f"{spd:+.4f}",
        help="Ideal = 0. Negative → unprivileged group receives fewer alerts.",
    )
    col_b.metric(
        "Disparate Impact (DI)",
        f"{di:.4f}" if not np.isnan(di) else "N/A",
        help="Ideal = 1.0. Below 0.8 fails the 4/5ths rule.",
    )
    col_c.metric(
        "4/5ths Rule",
        "PASS" if (not np.isnan(di) and di >= 0.8) else "FAIL",
    )

    # ── Positive rate bar chart ───────────────────────────────────────────────
    pos_rates = {
        g: float(np.mean(pred_two[groups_two == g]))
        for g in top_two
    }
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    ax3.bar(list(pos_rates.keys()), list(pos_rates.values()),
            color=["#3498db", "#e74c3c"], width=0.5)
    ax3.set_ylim(0, 1)
    ax3.set_ylabel("Alert Rate (fraction flagged)")
    ax3.set_title(f"Alert Rates by {demo_col}  |  threshold = {alert_threshold:.1f}")
    for g, rate in pos_rates.items():
        ax3.text(
            list(pos_rates.keys()).index(g), rate + 0.02,
            f"{rate:.1%}", ha="center", fontweight="bold",
        )
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # ── Fairness vs. threshold sensitivity ────────────────────────────────────
    st.subheader("Fairness Metrics vs. Decision Threshold")

    thresholds = np.linspace(
        float(np.percentile(scores_fair, 5)),
        float(np.percentile(scores_fair, 95)),
        40,
    )
    spd_vals, di_vals = [], []

    for t in thresholds:
        p_t = (scores_fair[mask_two] >= t).astype(int)
        spd_t = calculate_spd(
            true_labels_placeholder, p_t, groups_two,
            privileged_group, unprivileged_group,
        )
        di_t = calculate_di(
            true_labels_placeholder, p_t, groups_two,
            privileged_group, unprivileged_group,
        )
        spd_vals.append(spd_t)
        di_vals.append(di_t if not np.isnan(di_t) else 0.0)

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(11, 3))

    ax4a.plot(thresholds, spd_vals, color="#8e44ad", lw=2)
    ax4a.axhline(0, color="grey", linestyle="--", lw=1)
    ax4a.axvline(alert_threshold, color="#f39c12", linestyle=":", label="Current threshold")
    ax4a.set_xlabel("Decision Threshold")
    ax4a.set_ylabel("SPD")
    ax4a.set_title("Statistical Parity Difference vs. Threshold")
    ax4a.legend()

    ax4b.plot(thresholds, di_vals, color="#27ae60", lw=2)
    ax4b.axhline(0.8, color="#e74c3c", linestyle="--", lw=1, label="4/5ths rule (0.8)")
    ax4b.axvline(alert_threshold, color="#f39c12", linestyle=":", label="Current threshold")
    ax4b.set_xlabel("Decision Threshold")
    ax4b.set_ylabel("DI")
    ax4b.set_title("Disparate Impact vs. Threshold")
    ax4b.legend()

    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    with st.expander("Metric Definitions"):
        st.markdown(f"""
| Metric | Formula | Ideal |
|---|---|---|
| **SPD** | P(alert \\| {unprivileged_group}) − P(alert \\| {privileged_group}) | 0 |
| **DI**  | P(alert \\| {unprivileged_group}) / P(alert \\| {privileged_group}) | 1.0 (≥ 0.8 passes 4/5ths rule) |

A **negative SPD** means the unprivileged group is flagged *less often* —
in an abuse-detection context, this means potential cases may be systematically
missed. A **DI below 0.8** warrants regulatory review.

Current groups: **privileged = {privileged_group}**, **unprivileged = {unprivileged_group}**
(based on the two most frequent values in `{demo_col}`).
        """)
