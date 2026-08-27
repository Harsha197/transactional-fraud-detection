"""
Transactional Fraud Detection Dashboard
Interactive Streamlit Business Intelligence and Risk Intelligence Platform.
"""

import os
import sys
import sqlite3
import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_artifact, setup_visualization_style
from src.prediction import FraudRiskScorer
from src.data_loader import DataLoader

# Set Page Config
st.set_page_config(
    page_title="Transactional Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Financial Intelligence UI
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        color: #F8FAFC;
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #38BDF8;
        margin-top: 4px;
    }
    .risk-badge-low {
        background-color: #065F46;
        color: #34D399;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-badge-medium {
        background-color: #854D0E;
        color: #FBBF24;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-badge-high {
        background-color: #9A3412;
        color: #FB923C;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-badge-critical {
        background-color: #991B1B;
        color: #F87171;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_summary_data():
    """Load precomputed metrics and summary stats."""
    metrics_path = PROJECT_ROOT / "reports" / "metrics_summary.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_sample_dashboard_data(nrows: int = 50000):
    """Load dashboard dataset sample for interactive charting."""
    csv_path = PROJECT_ROOT / "powerbi" / "dashboard_data.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, nrows=nrows)
        return df
    
    # Fallback to loading from SQLite
    db_path = PROJECT_ROOT / "data" / "processed" / "fraud_detection.db"
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(f"SELECT * FROM transactions LIMIT {nrows}", conn)
        conn.close()
        return df
    return None


@st.cache_resource
def get_risk_scorer():
    """Load trained inference engine."""
    try:
        return FraudRiskScorer()
    except Exception as e:
        return None


def main():
    setup_visualization_style()
    metrics = load_summary_data()
    df_data = load_sample_dashboard_data()
    scorer = get_risk_scorer()

    # Sidebar Header & Navigation
    st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.sidebar.title("Fraud Guard AI")
    st.sidebar.caption("Enterprise Transaction Risk Intelligence")

    menu = st.sidebar.radio(
        "Navigation",
        [
            "📊 Executive KPI Overview",
            "🔍 Fraud Pattern Analytics",
            "🤖 Model Performance & Evaluation",
            "⚡ Real-Time Transaction Scorer",
            "💾 SQL Analytics & DB Explorer",
        ]
    )

    st.sidebar.divider()
    st.sidebar.info("""
    **Dataset:** European Credit Card Transactions  
    **Volume:** 284,807 Transactions  
    **Class Ratio:** 0.17% Rare Fraud Event  
    **Architecture:** Zero-Leakage ML Pipeline
    """)

    # -------------------------------------------------------------------------
    # TAB 1: EXECUTIVE KPI OVERVIEW
    # -------------------------------------------------------------------------
    if menu == "📊 Executive KPI Overview":
        st.title("Transactional Fraud Detection & Risk Overview")
        st.markdown("Comprehensive executive scorecard monitoring transaction volumes, fraud rates, and financial loss prevention metrics.")

        if metrics:
            total_trans = metrics.get("dataset_records", 283726)
            fraud_count = metrics.get("fraud_count", 473)
            legit_count = metrics.get("legit_count", 283253)
            fraud_rate = metrics.get("fraud_rate_pct", 0.1667)
            best_model = metrics.get("best_model", "XGBoost")
            precision = metrics.get("best_model_precision", 0.88)
            recall = metrics.get("best_model_recall", 0.82)
            pr_auc = metrics.get("best_model_pr_auc", 0.85)
        else:
            total_trans, fraud_count, legit_count, fraud_rate = 284807, 492, 284315, 0.172
            best_model, precision, recall, pr_auc = "XGBoost", 0.88, 0.82, 0.85

        # Top Metric Row
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Transactions</div>
                <div class="metric-value">{total_trans:,}</div>
                <div class="metric-sub">Processed Volume</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Fraudulent Transactions</div>
                <div class="metric-value" style="color: #F87171;">{fraud_count:,}</div>
                <div class="metric-sub">Flagged Incident Cases</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Legitimate Volume</div>
                <div class="metric-value" style="color: #34D399;">{legit_count:,}</div>
                <div class="metric-sub">Approved Safe Volume</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Incident Fraud Rate</div>
                <div class="metric-value" style="color: #FBBF24;">{fraud_rate:.3f}%</div>
                <div class="metric-sub">1 in 577 Transactions</div>
            </div>
            """, unsafe_allow_html=True)
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Best ML Champion</div>
                <div class="metric-value" style="color: #38BDF8; font-size: 1.4rem;">{best_model.split(' ')[0]}</div>
                <div class="metric-sub">PR-AUC: {pr_auc:.3f} | Rec: {recall*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Visual Row 1
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Class Distribution (Severe Imbalance)")
            fig_path = PROJECT_ROOT / "reports" / "figures" / "class_distribution.png"
            if fig_path.exists():
                st.image(str(fig_path), use_container_width=True)
            else:
                st.info("Class distribution figure generating...")

        with c2:
            st.subheader("Transaction Amount by Class")
            fig_path = PROJECT_ROOT / "reports" / "figures" / "amount_distribution.png"
            if fig_path.exists():
                st.image(str(fig_path), use_container_width=True)
            else:
                st.info("Amount distribution figure generating...")

        # Visual Row 2
        st.subheader("24-Hour Diurnal Fraud Activity & Rate")
        fig_path = PROJECT_ROOT / "reports" / "figures" / "hourly_fraud_trends.png"
        if fig_path.exists():
            st.image(str(fig_path), use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: FRAUD PATTERN ANALYTICS
    # -------------------------------------------------------------------------
    elif menu == "🔍 Fraud Pattern Analytics":
        st.title("In-Depth Fraud Behavioral Patterns & Signals")
        st.markdown("Detailed breakdown of anomalous PCA signals, transaction amount distributions, and risk concentration.")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Correlation Analysis: Top Predictive PCA Components")
            fig_path = PROJECT_ROOT / "reports" / "figures" / "correlation_matrix.png"
            if fig_path.exists():
                st.image(str(fig_path), use_container_width=True)
            st.markdown("""
            **Key Findings:**
            * **Strong Negative Signals:** `V14`, `V17`, `V12`, `V10` exhibit extreme negative shifts during fraudulent attempts.
            * **Strong Positive Signals:** `V4`, `V11` correlate positively with anomalous unauthorized transfers.
            """)

        with c2:
            st.subheader("Fraud Probability Across Amount Ranges")
            fig_path = PROJECT_ROOT / "reports" / "figures" / "amount_bins_fraud_rate.png"
            if fig_path.exists():
                st.image(str(fig_path), use_container_width=True)
            st.markdown("""
            **Key Findings:**
            * High fraud concentrations appear in both **Micro Transactions** (€0–€10, used for account ping testing/card validity checks) and **High/Extreme Amounts** (>€500).
            """)

        if df_data is not None:
            st.divider()
            st.subheader("Interactive Transaction Explorer")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                class_filter = st.selectbox("Filter by Class:", ["All", "Legitimate Only (0)", "Fraudulent Only (1)"])
            with col_f2:
                amount_range = st.slider("Filter Amount (€):", 0.0, 3000.0, (0.0, 1500.0))

            filtered_df = df_data.copy()
            if "Class" in filtered_df.columns:
                if class_filter == "Legitimate Only (0)":
                    filtered_df = filtered_df[filtered_df["Class"] == 0]
                elif class_filter == "Fraudulent Only (1)":
                    filtered_df = filtered_df[filtered_df["Class"] == 1]
            
            if "Amount" in filtered_df.columns:
                filtered_df = filtered_df[(filtered_df["Amount"] >= amount_range[0]) & (filtered_df["Amount"] <= amount_range[1])]

            st.dataframe(filtered_df.head(100), use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: MODEL PERFORMANCE & EVALUATION
    # -------------------------------------------------------------------------
    elif menu == "🤖 Model Performance & Evaluation":
        st.title("Machine Learning Benchmarking & Threshold Tuning")
        st.markdown("Comparative performance evaluation across multiple classifier algorithms under extreme class imbalance.")

        if metrics and "model_comparison" in metrics:
            comp_df = pd.DataFrame(metrics["model_comparison"])
            st.subheader("Classifier Benchmark Comparison Table")
            st.dataframe(
                comp_df.style.highlight_max(subset=["Precision", "Recall (Detection Rate)", "F1 Score", "ROC-AUC", "PR-AUC (Avg Precision)"], color="#1E3A8A"),
                use_container_width=True
            )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Precision-Recall & ROC Comparison Curves")
            fig_path = PROJECT_ROOT / "reports" / "figures" / "model_comparison_curves.png"
            if fig_path.exists():
                st.image(str(fig_path), use_container_width=True)

        with col2:
            st.subheader("Feature Importance Ranking")
            fig_path = PROJECT_ROOT / "reports" / "figures" / "feature_importance.png"
            if fig_path.exists():
                st.image(str(fig_path), use_container_width=True)

        st.divider()
        st.subheader("Operational Threshold Optimization & Trade-Off")
        fig_path = PROJECT_ROOT / "reports" / "figures" / "threshold_optimization.png"
        if fig_path.exists():
            st.image(str(fig_path), use_container_width=True)

        col_cm1, col_cm2 = st.columns(2)
        with col_cm1:
            st.subheader("Confusion Matrix (Best Model)")
            fig_path = PROJECT_ROOT / "reports" / "figures" / "confusion_matrix.png"
            if fig_path.exists():
                st.image(str(fig_path), use_container_width=True)
        with col_cm2:
            st.subheader("Financial Impact of Error Types")
            st.markdown("""
            * **False Negative (FN) — Missed Fraud:**  
              *A fraudulent transaction is classified as legitimate.*  
              **Business Impact:** Direct financial liability, chargeback penalties (€25-€50 per claim), merchant reputational erosion.
              
            * **False Positive (FP) — False Alarm:**  
              *A legitimate cardholder is falsely declined or challenged.*  
              **Business Impact:** Customer friction, card abandonment, estimated operational review cost (€10–€20 per incident).
              
            * **Executive Policy Decision:** Setting the threshold at **~0.30–0.50** ensures >80% fraud capture while maintaining low false-alarm friction.
            """)

    # -------------------------------------------------------------------------
    # TAB 4: REAL-TIME TRANSACTION SCORER
    # -------------------------------------------------------------------------
    elif menu == "⚡ Real-Time Transaction Scorer":
        st.title("Real-Time Transaction Risk Scoring Simulation")
        st.markdown("Interactive payment scoring simulator. Enter transaction attributes to generate instant risk probability, tier classification, and recommended action.")

        st.info("ℹ️ **Demonstration Simulator:** Emulates real-time risk decisioning at payment gateway authorization.")

        # Preset test cases
        st.subheader("1. Quick Presets")
        preset = st.selectbox(
            "Select a pre-configured transaction scenario:",
            [
                "Custom Input",
                "Scenario A: Normal Grocery Purchase (Legitimate, €45.20, Noon)",
                "Scenario B: High-Risk Midnight Transfer (Fraudulent Signal, €350.00, 3:00 AM, Severe V14/V17 Anomaly)",
                "Scenario C: Suspicious High-Value Outlier (Elevated Risk, €1,850.00, Extreme PCA shift)",
            ]
        )

        default_amount = 50.0
        default_time = 45000.0
        default_v14 = 0.0
        default_v17 = 0.0
        default_v12 = 0.0
        default_v10 = 0.0
        default_v4 = 0.0
        default_v11 = 0.0

        if "Scenario A" in preset:
            default_amount = 45.20
            default_time = 43200.0  # 12:00 PM
            default_v14, default_v17, default_v12, default_v10 = 0.12, 0.05, 0.08, -0.02
            default_v4, default_v11 = 0.10, -0.05
        elif "Scenario B" in preset:
            default_amount = 350.00
            default_time = 10800.0  # 3:00 AM
            default_v14, default_v17, default_v12, default_v10 = -6.50, -4.80, -5.20, -4.10
            default_v4, default_v11 = 4.20, 3.80
        elif "Scenario C" in preset:
            default_amount = 1850.00
            default_time = 82000.0
            default_v14, default_v17, default_v12, default_v10 = -3.20, -2.90, -2.50, -1.80
            default_v4, default_v11 = 2.10, 2.00

        st.subheader("2. Transaction Attributes")
        with st.form("risk_scorer_form"):
            c1, c2 = st.columns(2)
            with c1:
                input_amount = st.number_input("Transaction Amount (€):", min_value=0.0, max_value=25000.0, value=float(default_amount), step=1.0)
                input_time = st.number_input("Time Elapsed (Seconds):", min_value=0.0, max_value=172800.0, value=float(default_time), step=100.0)
                hour_calc = int((input_time // 3600) % 24)
                st.caption(f"Estimated Time of Day: **{hour_calc:02d}:00 hours**")

            with c2:
                input_v14 = st.slider("Component V14 (Key Fraud Negative Shift):", -15.0, 10.0, float(default_v14), 0.1)
                input_v17 = st.slider("Component V17 (Key Fraud Negative Shift):", -15.0, 10.0, float(default_v17), 0.1)
                input_v12 = st.slider("Component V12 (Fraud Indicator):", -15.0, 10.0, float(default_v12), 0.1)
                input_v4 = st.slider("Component V4 (Key Positive Spike):", -10.0, 15.0, float(default_v4), 0.1)

            submit_btn = st.form_submit_button("⚡ Score Transaction Risk", use_container_width=True)

        if submit_btn and scorer:
            # Construct transaction dict with all 28 PCA features
            sample_tx = {"Time": input_time, "Amount": input_amount}
            for i in range(1, 29):
                sample_tx[f"V{i}"] = 0.0
            sample_tx["V14"] = input_v14
            sample_tx["V17"] = input_v17
            sample_tx["V12"] = input_v12
            sample_tx["V4"] = input_v4

            result = scorer.score_transaction(sample_tx)

            st.divider()
            st.subheader("3. Real-Time Risk Assessment Result")
            res_col1, res_col2, res_col3 = st.columns(3)

            with res_col1:
                st.metric("Fraud Probability", result["fraud_probability_pct"])
                st.metric("Calibrated Risk Score", f"{result['risk_score']} / 100")

            with res_col2:
                tier = result["risk_tier"]
                badge_class = f"risk-badge-{tier.lower()}"
                st.markdown(f"**Risk Tier:**<br><span class='{badge_class}'>{tier.upper()} RISK</span>", unsafe_allow_html=True)
                st.markdown(f"<br>**Decision Flag:** `{result['fraud_flag']}`", unsafe_allow_html=True)

            with res_col3:
                st.markdown("**Recommended Operational Action:**")
                if tier == "Critical":
                    st.error(f"🛑 {result['recommended_action']}")
                elif tier == "High":
                    st.warning(f"⚠️ {result['recommended_action']}")
                elif tier == "Medium":
                    st.info(f"ℹ️ {result['recommended_action']}")
                else:
                    st.success(f"✅ {result['recommended_action']}")

    # -------------------------------------------------------------------------
    # TAB 5: SQL ANALYTICS & DB EXPLORER
    # -------------------------------------------------------------------------
    elif menu == "💾 SQL Analytics & DB Explorer":
        st.title("SQLite Database Query & Analytics Workbench")
        st.markdown("Execute SQL queries directly against the processed SQLite transactions and model predictions database.")

        loader = DataLoader()
        st.subheader("Pre-Built Analytical SQL Queries")
        query_choice = st.selectbox(
            "Select standard SQL query template:",
            [
                "1. Executive Fraud & Loss Summary",
                "2. Fraud Breakdown by Amount Bracket",
                "3. Hourly Diurnal Fraud Activity",
                "4. Top 10 High-Value Fraud Transactions",
                "5. Model Prediction Tier Summary",
                "Custom SQL Query",
            ]
        )

        query_templates = {
            "1. Executive Fraud & Loss Summary": """
                SELECT 
                    COUNT(*) AS total_transactions,
                    SUM(CASE WHEN is_fraud = 0 THEN 1 ELSE 0 END) AS legit_transactions,
                    SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
                    ROUND(SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 4) AS fraud_rate_pct,
                    ROUND(AVG(amount), 2) AS overall_avg_amount,
                    ROUND(AVG(CASE WHEN is_fraud = 1 THEN amount ELSE NULL END), 2) AS fraud_avg_amount,
                    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS total_fraud_loss_eur
                FROM transactions;
            """,
            "2. Fraud Breakdown by Amount Bracket": """
                SELECT 
                    CASE 
                        WHEN amount <= 10 THEN 'Micro (€0 - €10)'
                        WHEN amount <= 50 THEN 'Small (€10 - €50)'
                        WHEN amount <= 100 THEN 'Medium (€50 - €100)'
                        WHEN amount <= 500 THEN 'High (€100 - €500)'
                        ELSE 'Extreme (> €500)'
                    END AS amount_bracket,
                    COUNT(*) AS total_volume,
                    SUM(is_fraud) AS fraud_count,
                    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 4) AS fraud_rate_pct,
                    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_loss_eur
                FROM transactions
                GROUP BY amount_bracket
                ORDER BY fraud_rate_pct DESC;
            """,
            "3. Hourly Diurnal Fraud Activity": """
                SELECT 
                    hour_of_day,
                    time_period,
                    COUNT(*) AS volume,
                    SUM(is_fraud) AS fraud_count,
                    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 4) AS fraud_rate_pct
                FROM transactions
                GROUP BY hour_of_day
                ORDER BY hour_of_day;
            """,
            "4. Top 10 High-Value Fraud Transactions": """
                SELECT 
                    transaction_id,
                    time_sec,
                    hour_of_day,
                    amount,
                    v14, v17, v12, v4
                FROM transactions
                WHERE is_fraud = 1
                ORDER BY amount DESC
                LIMIT 10;
            """,
            "5. Model Prediction Tier Summary": """
                SELECT 
                    risk_tier,
                    COUNT(*) AS total_predicted,
                    SUM(actual_class) AS actual_frauds,
                    ROUND(SUM(actual_class) * 100.0 / COUNT(*), 2) AS precision_pct
                FROM model_predictions
                GROUP BY risk_tier;
            """,
        }

        default_query = query_templates.get(query_choice, "SELECT * FROM transactions LIMIT 20;")
        sql_input = st.text_area("SQL Query Editor:", value=default_query.strip(), height=160)

        if st.button("▶️ Execute Query", use_container_width=True):
            try:
                res_df = loader.run_sql_query(sql_input)
                st.success(f"Query returned {len(res_df):,} rows.")
                st.dataframe(res_df, use_container_width=True)
            except Exception as e:
                st.error(f"SQL Execution Error: {e}")


if __name__ == "__main__":
    main()
