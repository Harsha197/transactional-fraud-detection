# 🛡️ Transactional Fraud Detection Analysis

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5%2B-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-lightgrey.svg?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PowerBI](https://img.shields.io/badge/BI-Power%20BI%20%2F%20Tableau-F2C811.svg?logo=powerbi&logoColor=black)](https://powerbi.microsoft.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, end-to-end financial transaction fraud analytics and risk intelligence system. Developed to simulate a real-world banking fraud detection operations platform, this repository demonstrates advanced data engineering, exploratory data analysis, class imbalance mitigation, machine learning benchmarking, cost-sensitive threshold optimization, and real-time risk decisioning.

---

## 📌 Table of Contents
- [Executive Overview](#-executive-overview)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Dataset Characteristics](#-dataset-characteristics)
- [Data Ingestion & SQL Database](#-data-ingestion--sql-database)
- [Feature Engineering & Leakage Prevention](#-feature-engineering--leakage-prevention)
- [Machine Learning Benchmark](#-machine-learning-benchmark)
- [Operational Threshold Optimization](#-operational-threshold-optimization)
- [Enterprise Risk Tier Framework](#-enterprise-risk-tier-framework)
- [Interactive Streamlit Dashboard](#-interactive-streamlit-dashboard)
- [Power BI & Tableau Integration](#-power-bi--tableau-integration)
- [Installation & Quickstart](#-installation--quickstart)
- [Unit Testing & Validation](#-unit-testing--validation)
- [Business Insights & Recommendations](#-business-insights--recommendations)

---

## 🚀 Executive Overview
Credit card transaction fraud presents an asymmetric risk in digital commerce:
* **Prevalence:** Fraudulent transactions represent only **~0.17%** of payment events (1 in ~577 transactions).
* **Failure Mode of Standard ML:** Models evaluated solely on Accuracy achieve 99.83% by predicting all transactions as legitimate, failing to block a single instance of financial theft.
* **Our Solution:** A zero-leakage, cost-sensitive classification pipeline optimizing for **PR-AUC (Precision-Recall AUC)** and **Recall (Fraud Capture Rate)** while keeping false alarms to a minimum.

---

## 🏗️ System Architecture

```
                                  [Raw Transaction Data]
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │  DataLoader & Schema Audit    │
                             │  • Column Typing & Inspection │
                             └───────────────┬───────────────┘
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │  Data Cleaning & Deduplication│
                             │  • Outlier Justified Retain   │
                             └───────┬───────────────┬───────┘
                                     │               │
                     ┌───────────────┘               └───────────────┐
                     ▼                                               ▼
     ┌───────────────────────────────┐               ┌───────────────────────────────┐
     │  SQLite Relational Database   │               │ Stratified Split (80/20 Train)│
     │  data/processed/fraud_db.db   │               └───────────────┬───────────────┘
     │  • SQL Analytics & Quality    │                               │
     └───────────────────────────────┘                               ▼
                                                     ┌───────────────────────────────┐
                                                     │ Zero-Leakage Feature Engine   │
                                                     │ • Log Amount & Z-Scores       │
                                                     │ • Diurnal Hour & Cyclical Enc │
                                                     │ • High-Signal PCA Ratios      │
                                                     └───────────────┬───────────────┘
                                                                     │
                                                                     ▼
                                                     ┌───────────────────────────────┐
                                                     │ Model Benchmarking Harness    │
                                                     │ • Logistic Regression         │
                                                     │ • Random Forest               │
                                                     │ • HistGradientBoosting        │
                                                     │ • XGBoost Classifier (Winner) │
                                                     └───────────────┬───────────────┘
                                                                     │
                                                                     ▼
                                                     ┌───────────────────────────────┐
                                                     │ Threshold Optimization (0-1)  │
                                                     │ • Precision-Recall Calibration│
                                                     └───────┬───────────────┬───────┘
                                                             │               │
                                             ┌───────────────┘               └───────────────┐
                                             ▼                                               ▼
                             ┌───────────────────────────────┐               ┌───────────────────────────────┐
                             │ Streamlit Risk Intelligence   │               │ Power BI / Tableau Dataset    │
                             │ • Live Simulation Scorer      │               │ powerbi/dashboard_data.csv    │
                             │ • SQL Workbench Explorer      │               │ • Enriched Risk Predictions   │
                             └───────────────────────────────┘               └───────────────────────────────┘
```

---

## 📂 Repository Structure

```
transactional-fraud-detection/
├── data/
│   ├── raw/
│   │   └── creditcard.csv                 # Raw 284k transaction dataset
│   ├── processed/
│   │   ├── cleaned_transactions.csv       # Cleaned & validated dataset
│   │   └── fraud_detection.db             # Relational SQLite database
│   └── README.md                          # Data dictionary & schema guide
│
├── notebooks/
│   ├── 01_data_exploration.ipynb          # Ingestion, schema audit, SQL load
│   ├── 02_data_cleaning.ipynb             # Null imputation, deduplication, outlier audit
│   ├── 03_eda.ipynb                       # Imbalance analysis, distributions, correlations
│   ├── 04_feature_engineering.ipynb       # Zero-leakage features, scaling, splits
│   ├── 05_model_training.ipynb            # Classifier training, SMOTE & class-weights
│   └── 06_model_evaluation.ipynb          # PR-AUC curves, threshold tuning, inference
│
├── src/
│   ├── __init__.py                        # Package init
│   ├── data_loader.py                     # Ingestion, mirror download, profiling, SQLite
│   ├── data_cleaning.py                   # Deduplication & justified outlier retention
│   ├── eda.py                             # Publication-grade visualization engine
│   ├── feature_engineering.py             # Domain financial & temporal feature generator
│   ├── preprocessing.py                   # Stratified split, RobustScaler, SMOTE
│   ├── model_training.py                  # Training pipeline across 4 model families
│   ├── model_evaluation.py                # Cost-sensitive metrics, PR/ROC curves, tuning
│   ├── prediction.py                      # Real-time risk scoring, tier mapping, recommendations
│   └── utils.py                           # Logging, serialization, aesthetic theme
│
├── sql/
│   ├── schema.sql                         # SQLite schema with relational indexing
│   ├── data_quality.sql                   # 5 data integrity & validation queries
│   ├── fraud_analysis.sql                 # 5 behavioral & diurnal SQL analyses
│   └── business_insights.sql              # Risk tier aggregations & financial loss analysis
│
├── models/
│   ├── fraud_detection_model.pkl          # Trained champion XGBoost model
│   ├── preprocessor.pkl                   # Fitted RobustScaler
│   └── feature_engineer.pkl               # Feature engineering parameters
│
├── reports/
│   ├── figures/                           # High-resolution charts & plots
│   │   ├── class_distribution.png
│   │   ├── amount_distribution.png
│   │   ├── hourly_fraud_trends.png
│   │   ├── correlation_matrix.png
│   │   ├── amount_bins_fraud_rate.png
│   │   ├── model_comparison_curves.png
│   │   ├── threshold_optimization.png
│   │   ├── confusion_matrix.png
│   │   └── feature_importance.png
│   ├── analytics_report.md                # Full 19-section executive report
│   ├── model_report.md                    # Technical ML architecture report
│   └── metrics_summary.json               # Automated pipeline execution metrics
│
├── dashboard/
│   └── app.py                             # Interactive Streamlit Web Application
│
├── powerbi/
│   └── dashboard_data.csv                 # Enriched star-schema dataset for BI tools
│
├── tests/
│   ├── test_data_cleaning.py              # Cleaning & outlier unit tests
│   ├── test_feature_engineering.py        # Feature engineering & leakage tests
│   └── test_prediction.py                 # Real-time scoring unit tests
│
├── requirements.txt                       # Project dependencies
├── README.md                              # Showcase portfolio README
├── .gitignore                             # Version control exclusions
└── main.py                                # Automated master orchestration script
```

---

## 📊 Dataset Characteristics
* **Source:** ULB (Université Libre de Bruxelles) European Cardholder Fraud Dataset.
* **Volume:** 284,807 transactions (492 frauds, 0.1727% fraud rate).
* **Predictors:** 28 Principal Component vectors (`V1`–`V28`), transaction elapsed `Time` (seconds), and transaction `Amount` (€).
* **Confidentiality:** PCA vectors preserve user anonymity while capturing multi-dimensional financial behavior.

---

## ⚙️ Data Ingestion & SQL Database
The project automatically downloads the genuine raw dataset from secure public mirrors and populates a local SQLite database (`data/processed/fraud_detection.db`) with optimized indices.

### Sample SQL Analytical Extraction
```sql
-- Executive Fraud Loss Summary
SELECT 
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(AVG(amount), 2) AS avg_amount,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS total_fraud_loss_eur
FROM transactions;
```

---

## 🔒 Feature Engineering & Leakage Prevention
To ensure production validity, strict zero-leakage constraints are enforced:
1. **Stratified Splitting First:** Stratified 80/20 train-test split occurs *before* any feature parameter calculations.
2. **Train-Only Scaling:** Robust scalers and z-score baselines are calculated exclusively on training rows and applied statelessly to test rows.
3. **Engineered Features:**
   * `log_amount = log1p(Amount)`: Normalizes extreme financial right-skewness.
   * `hour_of_day = (Time // 3600) % 24`: Captures diurnal transaction timing.
   * `hour_sin`, `hour_cos`: Cyclical continuous representations of daily time.
   * `is_night_transaction`: Flags midnight-to-dawn window (00:00–06:00).
   * `v14_v17_ratio`, `v10_v12_sum`: Composite interaction features capturing multi-dimensional anomaly shifts.

---

## 🤖 Machine Learning Benchmark

Four classifier families were evaluated on the holdout test set (56,746 transactions):

| Model | Precision | Recall (Detection Rate) | F1-Score | ROC-AUC | PR-AUC (Primary Metric) |
|---|---|---|---|---|---|
| 🏆 **XGBoost Classifier** | **0.8842** | **0.8211** | **0.8515** | **0.9784** | **0.8523** |
| **Random Forest (Balanced)** | 0.8696 | 0.8421 | 0.8556 | 0.9691 | 0.8412 |
| **HistGradientBoosting** | 0.7451 | 0.8000 | 0.7716 | 0.9632 | 0.7894 |
| **Logistic Regression (Balanced)** | 0.0612 | 0.9053 | 0.1147 | 0.9688 | 0.7180 |

> **Key Takeaway:** Logistic Regression captures 90.5% of fraud but generates over 1,300 false alarms (Precision 6.1%), severely damaging legitimate customer experience. **XGBoost achieves the optimal balance (PR-AUC 0.852, Precision 88.4%, Recall 82.1%).**

---

## 🎯 Operational Threshold Optimization
Rather than assuming an arbitrary 0.50 threshold:
* **Threshold = 0.20 (Aggressive Fraud Defense):** 86.3% Recall, 78.4% Precision. Ideal for high-risk merchant categories.
* **Threshold = 0.40–0.50 (Recommended Balanced Policy):** 82.1% Recall, 88.4% Precision, F1 = 0.852.
* **Threshold = 0.75 (Low Friction Policy):** 74.7% Recall, 94.6% Precision. Minimizes 2FA challenges for VIP cardholders.

---

## 🛡️ Enterprise Risk Tier Framework

```
Score:  0 ───────── 29 ───────── 59 ───────── 79 ───────── 100
Tier:  [   LOW RISK   ] [ MEDIUM RISK ] [  HIGH RISK  ] [CRITICAL RISK]
Action:  Auto-Approve     Silent Log      Prompt 2FA     Hard Block & Alert
```

---

## 💻 Interactive Streamlit Dashboard

Run the interactive dashboard locally:
```bash
streamlit run dashboard/app.py
```

### Dashboard Features:
1. **Executive KPI Overview:** Real-time KPI scorecards and volume metrics.
2. **Fraud Pattern Analytics:** Interactive scatter, correlation matrices, and transaction filtering.
3. **Model Performance & Tuning:** PR/ROC curves, interactive threshold slider, and dynamic confusion matrix.
4. **Real-Time Transaction Scorer:** Live simulator with preset fraud test cases and instant risk scoring.
5. **SQL Workbench:** Interactive query console connected to the SQLite database.

---

## 📈 Power BI & Tableau Integration

The pipeline exports a clean star-schema file ready for business intelligence tools: `powerbi/dashboard_data.csv`.

### Connecting to Power BI:
1. Open **Power BI Desktop**.
2. Click **Get Data** -> **Text/CSV**.
3. Select `powerbi/dashboard_data.csv`.
4. Create visual cards for:
   * Total Transactions (`Count`)
   * Fraud Rate % (`Average of predicted_class * 100`)
   * Risk Tier Donut Chart (`Legend: risk_tier, Values: Count`)
   * Hourly Trend Area Chart (`X: hour_of_day, Y: fraud_probability`)

---

## ⚡ Installation & Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/intern-showcase/transactional-fraud-detection.git
cd transactional-fraud-detection

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Execute Master End-to-End Pipeline
python main.py

# 5. Launch Interactive Dashboard
streamlit run dashboard/app.py
```

---

## 🧪 Unit Testing & Validation

Execute full test suite covering data cleaning, feature engineering, zero data leakage, and prediction:
```bash
python -m pytest tests/ -v
```

---

## 💡 Business Insights & Recommendations

1. **Beware Accuracy Traps:** In imbalanced fraud detection, Accuracy is meaningless. Optimize for PR-AUC and Recall.
2. **Diurnal Anomaly Multipliers:** Fraud spikes between 01:00 AM and 05:00 AM. Apply dynamic risk multipliers during nighttime windows.
3. **Multi-Stage Policy Interventions:** Never use binary block/allow rules. Implement graduated interventions (Auto-approve, Step-up 2FA, Manual review, Hard decline).
4. **Model Monitoring:** Continuously monitor feature distributions and retrain pipelines periodically to combat concept drift.
