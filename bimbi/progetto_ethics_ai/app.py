"""
app.py
~~~~~~
Streamlit front-end for the Ethical CDSS – child abuse anomaly detection.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from util.ethics_metrics import CostModel, calculate_di, calculate_spd
from util.models import EthicalKDEAnomalyDetector

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Ethical CDSS – Child Abuse Detection",
    page_icon="🏥",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar – global hyper-parameters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ System Parameters")

    bandwidth = st.slider(
        "KDE Bandwidth",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Controls smoothness of the density estimate. "
             "Lower → sharper, higher → smoother.",
    )

    cost_alarm = st.slider(
        "Cost of False Alarm (FP)",
        min_value=1,
        max_value=20,
        value=1,
        step=1,
        help="Investigative cost incurred per false positive.",
    )

    cost_missed = st.slider(
        "Cost of Missed Abuse (FN)",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
        help="Social / ethical cost incurred per missed abuse case.",
    )

    alert_threshold = st.slider(
        "Alert Threshold",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1,
        help="Anomaly signal above this value triggers a clinical alert.",
    )

    st.divider()
    st.caption("Project: Ethics AI – CDSS v0.1")

# ---------------------------------------------------------------------------
# Mock data helpers
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "age_months",
    "num_visits_90d",
    "days_since_last_visit",
    "triage_code",
    "injury_severity_score",
    "num_different_injuries",
]

PRIOR_RULES = [
    {
        "feature": "num_visits_90d",
        "condition": "gt",
        "threshold": 3,
        "description": "More than 3 ER visits in 90 days is a sentinel red flag.",
    },
    {
        "feature": "days_since_last_visit",
        "condition": "lt",
        "threshold": 7,
        "description": "Repeated visits within 7 days may indicate non-accidental injury.",
    },
    {
        "feature": "num_different_injuries",
        "condition": "gt",
        "threshold": 2,
        "description": "Multiple distinct injury sites are associated with abuse.",
    },
]


@st.cache_resource
def get_trained_detector(bw: float) -> EthicalKDEAnomalyDetector:
    """Train a KDE detector on synthetic normal-pathway data."""
    rng = np.random.default_rng(42)
    n_normal = 400

    X_normal = np.column_stack([
        rng.normal(36, 12, n_normal),     # age_months
        rng.poisson(1.2, n_normal),        # num_visits_90d (low for normal)
        rng.normal(60, 20, n_normal),      # days_since_last_visit
        rng.choice([1, 2, 3], n_normal),   # triage_code (1=urgent…3=minor)
        rng.normal(5, 2, n_normal),        # injury_severity_score
        rng.poisson(0.8, n_normal),        # num_different_injuries
    ])
    X_normal = np.clip(X_normal, 0, None)

    detector = EthicalKDEAnomalyDetector(bandwidth=bw)
    detector.set_prior_knowledge(PRIOR_RULES)
    detector.fit(X_normal)
    return detector


def generate_mock_patient(suspicious: bool = True) -> np.ndarray:
    """Return a single-window feature vector for a synthetic patient."""
    rng = np.random.default_rng()
    if suspicious:
        return np.array([[
            18,                          # age_months (toddler)
            rng.integers(4, 8),          # num_visits_90d – high
            rng.integers(1, 6),          # days_since_last_visit – very recent
            1,                           # triage_code – urgent
            rng.uniform(8, 15),          # injury_severity_score – high
            rng.integers(3, 6),          # num_different_injuries – multiple
        ]])
    else:
        return np.array([[
            48,                          # age_months
            rng.integers(0, 2),          # num_visits_90d – low
            rng.integers(30, 120),       # days_since_last_visit – distant
            3,                           # triage_code – minor
            rng.uniform(1, 4),           # injury_severity_score – low
            rng.integers(0, 2),          # num_different_injuries – few
        ]])


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_dashboard, tab_xai, tab_fairness = st.tabs([
    "🏥 CDSS Dashboard",
    "🔍 Explainability (XAI)",
    "⚖️ Fairness Assessment",
])

# ── Tab 1: Dashboard ────────────────────────────────────────────────────────

with tab_dashboard:
    st.header("CDSS Dashboard – Real-Time Anomaly Screening")

    col_left, col_right = st.columns([2, 1])

    with col_right:
        patient_type = st.radio(
            "Simulate Patient Type",
            ["Suspicious Pattern", "Normal Pattern"],
        )
        run_btn = st.button("▶ Run Detection", type="primary")

    detector = get_trained_detector(bandwidth)

    if run_btn:
        suspicious = patient_type == "Suspicious Pattern"
        X_patient = generate_mock_patient(suspicious=suspicious)
        signal = detector.get_anomaly_signal(X_patient, feature_names=FEATURE_NAMES)
        score = float(signal[0])

        with col_left:
            st.subheader("Anomaly Signal")
            col_score, col_thresh = st.columns(2)
            col_score.metric("Anomaly Score", f"{score:.4f}")
            col_thresh.metric("Alert Threshold", f"{alert_threshold:.1f}")

            if score >= alert_threshold:
                st.error(
                    "🚨 **ALERT** – Anomaly score exceeds threshold. "
                    "Clinical review is recommended.",
                    icon="🚨",
                )
            else:
                st.success(
                    "✅ Score below threshold. No immediate alert.",
                    icon="✅",
                )

            # Feature bar chart
            feature_vals = X_patient[0]
            fig, ax = plt.subplots(figsize=(7, 3))
            bars = ax.barh(FEATURE_NAMES, feature_vals,
                           color=["#e74c3c" if v > np.median(feature_vals) else "#3498db"
                                  for v in feature_vals])
            ax.set_xlabel("Feature Value")
            ax.set_title("Patient Window – Feature Values")
            plt.tight_layout()
            st.pyplot(fig)

        with col_right:
            st.subheader("Prior Rules Status")
            for rule in PRIOR_RULES:
                col = rule["feature"]
                idx = FEATURE_NAMES.index(col)
                val = float(feature_vals[idx])
                violated = (
                    (rule["condition"] == "gt"  and val >  rule["threshold"]) or
                    (rule["condition"] == "gte" and val >= rule["threshold"]) or
                    (rule["condition"] == "lt"  and val <  rule["threshold"]) or
                    (rule["condition"] == "lte" and val <= rule["threshold"]) or
                    (rule["condition"] == "eq"  and val == rule["threshold"])
                )
                icon = "🔴" if violated else "🟢"
                st.markdown(f"{icon} **{col}** = `{val:.1f}` "
                            f"(threshold {rule['condition']} {rule['threshold']})")
    else:
        with col_left:
            st.info("Configure parameters in the sidebar and press **▶ Run Detection**.")

    # Cost optimisation section
    st.divider()
    st.subheader("Threshold Optimisation via Cost Model")

    rng = np.random.default_rng(0)
    mock_signals     = rng.uniform(0, 8, 200)
    mock_true_labels = (mock_signals + rng.normal(0, 1.5, 200) > 4).astype(int)

    cost_model = CostModel(cost_alarm=cost_alarm, cost_missed=cost_missed)
    best = cost_model.optimize_threshold(mock_signals, mock_true_labels)
    df_costs = best["all_results"]

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.plot(df_costs["threshold"], df_costs["total_cost"], lw=2, color="#2c3e50")
    ax2.axvline(best["threshold"], color="#e74c3c", linestyle="--",
                label=f"Optimal θ = {best['threshold']:.2f}")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Total Cost")
    ax2.set_title("Cost Landscape across Thresholds")
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)

    st.metric("Optimal Threshold", f"{best['threshold']:.3f}")
    st.metric("Minimum Total Cost", f"{best['total_cost']:.1f}")

# ── Tab 2: Explainability ───────────────────────────────────────────────────

with tab_xai:
    st.header("Explainability – Defeasible Reasoning Report")
    st.caption(
        "In a live deployment, click **Generate Report** to query the LLM. "
        "A pre-generated example is shown below."
    )

    mock_timeline = [
        {"date": "2024-11-10", "chief_complaint": "Bruised arm",
         "triage_code": 2, "discharge_diagnosis": "Contusion"},
        {"date": "2024-11-15", "chief_complaint": "Head bump",
         "triage_code": 1, "discharge_diagnosis": "Mild concussion"},
        {"date": "2024-11-19", "chief_complaint": "Burns on hand",
         "triage_code": 1, "discharge_diagnosis": "Second-degree burns"},
    ]

    with st.expander("Patient Timeline", expanded=True):
        st.table(pd.DataFrame(mock_timeline))

    mock_report = """
**Prima-Facie Concerns**
The patient presents three distinct ER visits within nine days, each involving
a different anatomical site (arm, head, hand) with escalating severity.
This constellation — multiple injury types, short inter-visit interval, toddler
age — matches WHO and NICE criteria for a non-accidental injury pattern.

**Triggered Clinical Red Flags**
- ✅ WHO guideline §4.2: ≥ 3 ER visits within 90 days.
- ✅ NICE CG89: Repeated head injury in a pre-verbal child.
- ✅ AAP Red Book: Burns in a dependent anatomical site with inconsistent history.

**Plausible Alternative Explanations**
- The family may reside in a high-risk physical environment (e.g. building site).
- Underlying coagulation disorder could explain bruising frequency.
- Caregiver anxiety may drive repeated ER attendance for minor incidents.

**Balanced Clinical Recommendation**
The CDSS alert warrants a multi-disciplinary safeguarding review (paediatrician,
social worker, and named nurse for child protection). This is NOT a definitive
finding of abuse. A full history, home assessment, and skeletal survey should
inform the clinical decision. Document all findings per local safeguarding policy.
""".strip()

    if st.button("💬 Generate LLM Report (requires OPENAI_API_KEY)"):
        if not os.getenv("OPENAI_API_KEY"):
            st.warning(
                "OPENAI_API_KEY is not set. Showing pre-generated example instead.",
                icon="⚠️",
            )
            st.markdown(mock_report)
        else:
            from util.llm_wrapper import generate_xai_report
            with st.spinner("Querying LLM…"):
                report = generate_xai_report(
                    patient_timeline=mock_timeline,
                    anomaly_score=5.82,
                    triggered_rules=PRIOR_RULES,
                )
            st.markdown(report)
    else:
        st.markdown("**Pre-generated example report:**")
        st.markdown(mock_report)

# ── Tab 3: Fairness ─────────────────────────────────────────────────────────

with tab_fairness:
    st.header("Fairness Assessment – Demographic Parity")

    st.markdown(
        "This panel evaluates whether the anomaly detector flags patients at "
        "equal rates across demographic groups (Statistical Parity / Disparate Impact)."
    )

    rng = np.random.default_rng(7)
    n = 300

    # Simulate predictions and demographics
    groups       = rng.choice(["Group A (Majority)", "Group B (Minority)"], n, p=[0.6, 0.4])
    true_labels  = rng.integers(0, 2, n)
    # Introduce slight bias: minority group flagged slightly more often
    pred_probs   = np.where(groups == "Group B (Minority)",
                            rng.uniform(0.45, 0.95, n),
                            rng.uniform(0.30, 0.85, n))
    predictions  = (pred_probs >= 0.5).astype(int)

    spd = calculate_spd(true_labels, predictions, groups,
                        "Group A (Majority)", "Group B (Minority)")
    di  = calculate_di(true_labels, predictions, groups,
                       "Group A (Majority)", "Group B (Minority)")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Statistical Parity Difference (SPD)", f"{spd:+.4f}",
                 help="Ideal = 0. Negative → minority receives fewer positives.")
    col_b.metric("Disparate Impact (DI)", f"{di:.4f}",
                 help="Ideal = 1.0. Below 0.8 → fails 4/5ths rule.")
    col_c.metric(
        "4/5ths Rule",
        "PASS ✅" if di >= 0.8 else "FAIL ❌",
    )

    # Bar chart: positive rates per group
    group_labels = ["Group A (Majority)", "Group B (Minority)"]
    pos_rates = [
        float(np.mean(predictions[groups == g])) for g in group_labels
    ]

    fig3, ax3 = plt.subplots(figsize=(6, 3))
    colors = ["#3498db", "#e74c3c"]
    bars = ax3.bar(group_labels, pos_rates, color=colors, width=0.5)
    ax3.axhline(pos_rates[0], color="#3498db", linestyle="--", alpha=0.5)
    ax3.set_ylim(0, 1)
    ax3.set_ylabel("Positive Prediction Rate")
    ax3.set_title("Disparate Impact – Positive Rates by Group")
    for bar, rate in zip(bars, pos_rates):
        ax3.text(bar.get_x() + bar.get_width() / 2, rate + 0.02,
                 f"{rate:.2%}", ha="center", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig3)

    # Threshold sensitivity
    st.subheader("Fairness vs. Threshold Sensitivity")
    thresholds = np.linspace(0.2, 0.9, 30)
    spd_vals, di_vals = [], []
    for t in thresholds:
        preds_t = (pred_probs >= t).astype(int)
        spd_vals.append(
            calculate_spd(true_labels, preds_t, groups,
                          "Group A (Majority)", "Group B (Minority)")
        )
        di_t = calculate_di(true_labels, preds_t, groups,
                            "Group A (Majority)", "Group B (Minority)")
        di_vals.append(di_t if not np.isnan(di_t) else 0.0)

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(10, 3))

    ax4a.plot(thresholds, spd_vals, color="#8e44ad", lw=2)
    ax4a.axhline(0, color="grey", linestyle="--")
    ax4a.set_xlabel("Decision Threshold")
    ax4a.set_ylabel("SPD")
    ax4a.set_title("SPD vs. Threshold")

    ax4b.plot(thresholds, di_vals, color="#27ae60", lw=2)
    ax4b.axhline(0.8, color="#e74c3c", linestyle="--", label="4/5ths rule (0.8)")
    ax4b.set_xlabel("Decision Threshold")
    ax4b.set_ylabel("DI")
    ax4b.set_title("Disparate Impact vs. Threshold")
    ax4b.legend()

    plt.tight_layout()
    st.pyplot(fig4)

    with st.expander("ℹ️ Metric Definitions"):
        st.markdown("""
| Metric | Formula | Ideal value |
|---|---|---|
| **SPD** | P(Ŷ=1&#124;minority) − P(Ŷ=1&#124;majority) | 0 |
| **DI**  | P(Ŷ=1&#124;minority) / P(Ŷ=1&#124;majority) | 1.0 (≥ 0.8 passes 4/5ths rule) |

A **negative SPD** means the minority group is flagged *less often*, which in
an abuse-detection setting could be harmful (missed cases).
A **DI below 0.8** triggers regulatory scrutiny under US EEOC guidelines.
        """)
