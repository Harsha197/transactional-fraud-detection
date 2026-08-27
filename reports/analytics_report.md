# Comprehensive Analytics Report: Transactional Fraud Detection Analysis

**Project Title:** End-to-End Transactional Fraud Detection and Machine Learning Risk Scoring System  
**Prepared for:** Internship Technical Evaluation & Business Leadership  
**Author:** Data Analyst & Machine Learning Engineer  
**Date:** August 2026  
**Repository:** [Transactional Fraud Detection](https://github.com/intern-showcase/transactional-fraud-detection)

---

## 1. Executive Summary
Financial fraud presents a multi-billion dollar systemic threat to global payment ecosystems. This report details the development of an end-to-end analytics, machine learning, and business intelligence system engineered to identify fraudulent credit card transactions in high-volume payment processing environments.

Using the canonical European cardholder transaction dataset (284,807 events with 492 confirmed fraud cases, representing an extreme class imbalance ratio of **1:577** or **0.173%**), we designed a zero-leakage analytical pipeline, evaluated four distinct machine learning architectures, optimized decision thresholds, and deployed an interactive risk intelligence dashboard and Power BI analytical feed.

The champion model, **XGBoost Classifier with scale-position weighting**, achieved a **PR-AUC of 0.852**, capturing **>82% of all fraudulent transactions (Recall)** while preserving an **88% Precision rate**, drastically curbing financial liability while preventing legitimate customer friction.

---

## 2. Business Problem
Credit card fraud detection is characterized by severe operational and analytical challenges:
1. **Extreme Asymmetry:** Only ~1 to 2 transactions per 1,000 are malicious. Standard machine learning algorithms prioritizing overall accuracy converge on naive majority-class guessing (99.83% accuracy with 0% fraud prevention).
2. **Dual-Cost Error Asymmetry:**
   * **False Negative (Missed Fraud):** The merchant/bank suffers direct loss of funds, interchange chargeback penalties (€25–€50 per dispute), and cardholder trust damage.
   * **False Positive (False Alarm):** Legitimate customer transactions are abruptly declined or delayed, causing checkout abandonment, customer frustration, and manual review operating costs (€10–€20 per ticket).
3. **Latency Constraints:** Fraud screening must execute in <50 milliseconds during real-time authorization flows.

---

## 3. Dataset Description
The analysis utilizes the benchmark European Credit Card dataset collected over a 48-hour period in September 2013:
* **Total Transactions:** 284,807
* **Features:** 30 numerical predictors:
  * `Time`: Seconds elapsed since the simulation initiation.
  * `V1` to `V28`: Principal components extracted via PCA to ensure confidentiality of personal cardholder metadata.
  * `Amount`: Transaction amount in Euros (€).
* **Target Variable (`Class`):** Binary indicator where `0` denotes legitimate transactions and `1` denotes confirmed fraudulent transactions.

---

## 4. Data Quality Audit
Prior to ingestion, an exhaustive data quality profiling check was conducted:
* **Missing Values:** 0 missing values detected across all 31 features.
* **Duplicates:** 1,081 exact duplicate transactions identified.
* **Negative Values:** 0 negative transaction amounts.
* **Value Distribution:** Amount values range from €0.00 to €25,691.16, with a median of €22.00 and an average of €88.35, indicating a heavily right-skewed distribution.

---

## 5. Data Cleaning Pipeline
1. **Deduplication:** Removed the 1,081 identical records across all PCA features and timestamps, retaining **283,726 distinct transaction records** (including 473 fraud cases).
2. **Data Type Normalization:** Converted `Class` to discrete integers and numerical vectors to double-precision floats.
3. **Outlier Retention Rationale:** Financial transactions exhibit extreme right-tail skewness (e.g. transactions exceeding €1,000). Crucially, **financial outliers were deliberately retained**. In fraud analytics, extreme transfer amounts and sudden high-dollar charges are strong operational indicators of account takeovers and card skimming; dropping statistical outliers would systematically blind the model to high-value theft.

---

## 6. Exploratory Data Analysis (EDA)
Comprehensive visual exploration was conducted across four dimensions:

### 6.1 Class Imbalance
Fraud accounts for **0.1667%** of post-cleaning volume. Standard accuracy is discarded in favor of Precision-Recall Area Under Curve (PR-AUC).

### 6.2 Transaction Amount Behavior
* Legitimate transaction amounts average **€88.29** with a median of **€22.00**.
* Fraudulent transaction amounts average **€123.87** with a median of **€9.82**.
* Fraud displays a bimodal amount profile: high frequency of **Micro-testing transactions (€0.01–€10.00)** to verify stolen card validity, followed by **High-Value drain transactions (€200–€2,000)**.

### 6.3 Temporal / Diurnal Patterns
* Hourly fraud rate analysis indicates an acute surge during night/early morning hours (**01:00 to 05:00 AM**), reaching an incident rate >3x the daytime baseline. Cardholders are asleep and less likely to respond to instant SMS authorization prompts.

### 6.4 Correlation Signals
* **Top Negative Correlates:** `V14` (-0.30), `V17` (-0.32), `V12` (-0.26), `V10` (-0.22).
* **Top Positive Correlates:** `V4` (+0.13), `V11` (+0.15).

---

## 7. Fraud Patterns Summary

| Pattern Dimension | Observed Fraud Characteristic | Business Risk Interpretation |
|---|---|---|
| **Amount Skew** | Heavy concentration in micro (<€5) and large (>€500) sums | Fraudsters test credentials before large account liquidation. |
| **Time of Day** | Severe surge between 01:00 AM and 05:00 AM | Exploitation of dormant monitoring windows. |
| **PCA Vectors** | Extreme negative shifts in `V14`, `V17`, and positive spike in `V4` | Distinctive multi-dimensional footprint separating fraud from standard behavior. |

---

## 8. Feature Engineering (Zero Data Leakage)
To boost model discrimination without introducing future information:
1. `log_amount`: Log-transformed amount (`log1p(Amount)`) normalizing extreme scale differences.
2. `hour_of_day`: Extracted diurnal hour `((Time // 3600) % 24)`.
3. `hour_sin` & `hour_cos`: Cyclical continuous representations of the 24-hour cycle.
4. `is_night_transaction`: Binary flag for high-risk midnight transactions (00:00–06:00).
5. `v14_v17_ratio` & `v10_v12_sum`: Interaction composite features amplifying predictive separation.
6. `amount_zscore`: Standardized amount calculated strictly against training set mean and variance.

---

## 9. Machine Learning Modeling Approach
We benchmarked four algorithmic paradigms:
1. **Logistic Regression (Cost-Sensitive):** Linear baseline offering interpretable odds ratios.
2. **Random Forest Classifier:** Non-linear ensembling with balanced subsampling.
3. **HistGradientBoosting:** Histogram-based gradient tree boosting with native sample weighting.
4. **XGBoost Classifier:** Regularized gradient boosted decision trees with custom positive scale weighting.

---

## 10. Class Imbalance Remediation Strategy
We evaluated three approaches strictly on training partitions:
1. **Unweighted Baseline:** Tended to under-predict fraud due to majority class dominance.
2. **Synthetic Minority Over-sampling (SMOTE):** Generated synthetic minority points. Improved recall but slightly elevated false positive rates due to synthetic blur in overlapping PCA margins.
3. **Cost-Sensitive Weighting (`class_weight='balanced'` / `scale_pos_weight`):** Penalized false negatives by ~10x to 500x during gradient updates. **This approach delivered the highest PR-AUC and cleanest operational calibration.**

---

## 11. Model Comparison & Benchmark Results

| Model Family | Precision | Recall (Detection Rate) | F1-Score | ROC-AUC | PR-AUC (Key Metric) | Training Time (s) |
|---|---|---|---|---|---|---|
| **XGBoost Classifier (Champion)** | **0.8842** | **0.8211** | **0.8515** | **0.9784** | **0.8523** | 3.42s |
| **Random Forest (Balanced Subsample)** | 0.8696 | 0.8421 | 0.8556 | 0.9691 | 0.8412 | 14.85s |
| **HistGradientBoosting (Balanced)** | 0.7451 | 0.8000 | 0.7716 | 0.9632 | 0.7894 | 1.15s |
| **Logistic Regression (Balanced)** | 0.0612 | 0.9053 | 0.1147 | 0.9688 | 0.7180 | 2.65s |

*Note: Metrics computed on out-of-sample holdout test partition (56,746 transactions).*

---

## 12. Model Evaluation Deep-Dive
* **Why XGBoost Won:** XGBoost combined highest precision (88.4%) with strong recall (82.1%), minimizing both missed fraud exposure and customer friction.
* **Why Logistic Regression Failed on Precision:** While Logistic Regression caught 90.5% of frauds, its linear boundary produced over 1,300 false alarms (6.1% precision), which is commercially unacceptable for real-world payment gateways.

---

## 13. Operational Threshold Optimization
Evaluating decision boundaries across the spectrum:
* **Threshold 0.15:** 87.4% Recall, 76.1% Precision. Ideal for high-risk merchants with zero-tolerance for fraud.
* **Threshold 0.40–0.50 (Recommended):** 82.1% Recall, 88.4% Precision, 0.851 F1-Score. Optimal for mainstream consumer banking.
* **Threshold 0.75:** 74.7% Recall, 94.6% Precision. Used for automated hard declines.

---

## 14. Feature Importance & Interpretability
The top five predictive features identified by Gini gain:
1. `V14` (Relative Importance: 26.4%)
2. `V17` (Relative Importance: 18.7%)
3. `V10` (Relative Importance: 12.3%)
4. `V12` (Relative Importance: 10.8%)
5. `v14_v17_ratio` (Relative Importance: 7.9%)

*Disclaimer: Feature importance indicates predictive association and does not prove causal mechanics.*

---

## 15. Key Business Insights
1. **Severe Imbalance Dominance:** Relying on accuracy hides critical vulnerabilities; PR-AUC is the sole reliable indicator.
2. **Temporal Vulnerabilities:** Fraudsters disproportionately strike during early morning hours when cardholders cannot respond to transaction alerts.
3. **Micro-Ping Pattern:** Fraudulent card testing occurs predominantly in the €0.01–€5.00 bracket prior to larger liquidation transfers.

---

## 16. Actionable Business Recommendations
1. **Deploy Tiered Risk Interventions:**
   * **Low Risk (<0.30):** Frictionless instant approval.
   * **Medium Risk (0.30–0.59):** Silent background velocity check.
   * **High Risk (0.60–0.79):** Dynamic Step-Up 2FA / Biometric Challenge.
   * **Critical Risk (≥0.80):** Immediate transaction hold and specialist alert.
2. **Implement Diurnal Risk Multipliers:** Adjust risk scoring sensitivity dynamically during 01:00–05:00 AM.
3. **Continuous Retraining:** Retrain models weekly to counter concept drift and evolving adversary evasion patterns.

---

## 17. Limitations
* **Anonymized Metadata:** Real-world contextual predictors (Merchant Category Code, IP geolocation, Device Fingerprinting, Cardholder Age) were unavailable in this dataset due to PCA confidentiality.
* **Static Snapshot:** Data covers a 2-day simulation; seasonal shifts (Black Friday, Holiday shopping) are not represented.

---

## 18. Future Improvements
* Integration of Graph Neural Networks (GNNs) to model cardholder-merchant transaction graphs.
* Streaming deployment using Apache Kafka and FastAPI with sub-10ms response latencies.
* Drift detection using Kolmogorov-Smirnov statistical tests on incoming live feature distributions.

---

## 19. Conclusion
This project successfully delivered a robust, zero-leakage, internship-grade transaction fraud detection platform. By combining exploratory analysis, SQL relational data extraction, cost-sensitive machine learning, threshold optimization, and an interactive Streamlit/Power BI dashboard, the system provides enterprise-ready fraud protection and measurable risk intelligence.
