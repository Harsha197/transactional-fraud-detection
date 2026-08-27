# Technical Machine Learning & Risk Calibration Report

## Project: Transactional Fraud Detection System
**Author:** Data Analyst & Machine Learning Engineer  
**Dataset:** European Credit Card Transactions (284,807 events)  
**Evaluation Standard:** Zero-Leakage Cross-Validation & PR-AUC Primary Optimization

---

## 1. Machine Learning Architecture Overview
The fraud detection system uses an ensemble and gradient-boosted decision tree architecture combined with cost-sensitive loss weighting to overcome severe class imbalance (~0.17% fraud prevalence).

```
Raw Transaction Feed
        │
        ▼
┌───────────────────────────────────────┐
│ Stratified Partitioning (80/20)       │
│ Train: 226,980 | Test: 56,746         │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ Feature Engineering Pipeline          │
│ • log1p(Amount) & Amount Z-Score      │
│ • Diurnal Hour & Cyclical Sin/Cos     │
│ • High-Impact PCA Interactions        │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ Zero-Leakage Robust Scaling           │
│ Fitted strictly on Training Partition │
└───────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────┐
│ Model Training Harness                                 │
│ 1. Logistic Regression (Cost-Sensitive Baseline)       │
│ 2. Random Forest (Balanced Subsample Ensembling)       │
│ 3. HistGradientBoosting (Histogram Regularization)     │
│ 4. XGBoost Classifier (Scale Pos Weight Tuned)         │
└────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────┐
│ Threshold Optimization & Risk Calibration              │
│ Optimal Threshold Selection: 0.30 - 0.50               │
│ Risk Tiers: Low, Medium, High, Critical                │
└────────────────────────────────────────────────────────┘
```

---

## 2. Algorithm Comparison & Rationale

| Model | Algorithmic Family | Pros | Limitations | Primary Business Role |
|---|---|---|---|---|
| **Logistic Regression (Balanced)** | Linear Generalized Model | Fully interpretable, fast inference (<1ms), clear feature odds ratios. | Linear boundary cannot capture complex feature interactions. | Interpretable Baseline |
| **Random Forest** | Bagging Ensemble of Decision Trees | Captures non-linearities, robust to variance, provides Gini feature importance. | Slower inference, larger memory footprint. | Non-linear Benchmark |
| **HistGradientBoosting** | Histogram-based Gradient Boosting | Extremely fast training, handles large datasets efficiently, native class weights. | Less tunable than standalone XGBoost. | Fast Gradient Boosting |
| **XGBoost Classifier** | Extreme Gradient Boosted Trees | State-of-the-art predictive power, fine-grained tree regularization, handles scale_pos_weight. | Requires careful hyperparameter tuning. | **Champion Production Model** |

---

## 3. Evaluation Metric Philosophy

In standard classification tasks, Accuracy is standard. However, in transactional fraud detection with a 99.83% negative class:
$$\text{Accuracy} = \frac{TN + TP}{Total}$$
A trivial model predicting "Legitimate" for every transaction achieves **99.83% accuracy** while capturing **0% of fraud**, resulting in complete financial exposure.

### Primary Metrics Prioritized:
1. **PR-AUC (Precision-Recall Area Under Curve / Average Precision):** Evaluates precision across all recall operating points without being artificially inflated by millions of true negatives.
2. **Recall (Fraud Detection Rate):** Measures the proportion of actual fraudulent transactions detected ($TP / (TP + FN)$).
3. **Precision (Positive Predictive Value):** Measures the proportion of flagged transactions that are genuinely fraudulent ($TP / (TP + FP)$).
4. **F1-Score:** Harmonic mean balancing fraud capture with customer friction.

---

## 4. Operational Threshold Calibration

Rather than enforcing the arbitrary 0.50 probability threshold:
* **Lower Threshold (0.15–0.30):** High-Recall Mode. Captures >85% of fraud at the expense of additional 2FA challenges.
* **Balanced Threshold (~0.40–0.50):** Maximum F1-Score. Optimal balance for standard payment gateway routing.
* **Higher Threshold (>0.70):** Ultra-High Precision Mode. Used for automated hard declines with near-zero false alarms.

---

## 5. Enterprise Risk Tier Framework

| Risk Tier | Probability Range | Calibrated Score | Recommended Operational Policy |
|---|---|---|---|
| **Low** | 0.00 – 0.29 | 0 – 29 | Auto-approve transaction; standard fraud telemetry. |
| **Medium** | 0.30 – 0.59 | 30 – 59 | Background risk logging; evaluate customer velocity limits. |
| **High** | 0.60 – 0.79 | 60 – 79 | Prompt Step-Up Authentication (SMS OTP / Biometric 2FA). |
| **Critical** | 0.80 – 1.00 | 80 – 100 | Immediate payment decline; trigger live fraud investigator alert. |
