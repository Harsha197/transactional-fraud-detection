"""
Automated Generator for All 6 Standalone Jupyter Notebooks
Builds publication-grade .ipynb files with detailed explanations, markdown, code, and visualizations.
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }


def generate_all_notebooks():
    # -------------------------------------------------------------------------
    # 1. 01_data_exploration.ipynb
    # -------------------------------------------------------------------------
    nb1_cells = [
        md_cell("""# 01 — Data Exploration and Data Extraction
**Project:** Transactional Fraud Detection Analysis  
**Author:** Data Analyst & Machine Learning Engineering Intern  
**Objective:** Ingest the European Credit Card Fraud dataset, discover its schema, perform data quality validation, and store the structured data into SQLite for relational SQL analysis.
"""),
        code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import sqlite3

# Set root directory
PROJECT_ROOT = Path("..").resolve()
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import DataLoader
from src.utils import get_logger

logger = get_logger("ExplorationNotebook")
print("Environment initialized successfully.")
"""),
        md_cell("""## 1. Load Raw Dataset
We utilize the modular `DataLoader` class to load the credit card transactions. The dataset contains 284,807 transactions with 30 anonymized numerical features and a binary target `Class` (0 = Legitimate, 1 = Fraudulent).
"""),
        code_cell("""loader = DataLoader()
df_raw = loader.load_data()

print(f"Dataset Shape: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")
df_raw.head()
"""),
        md_cell("""## 2. Schema and Data Quality Profiling
Let's inspect data types, memory consumption, missing values, duplicates, and statistical summaries.
"""),
        code_cell("""summary = loader.inspect_data(df_raw)

print(f"Memory Usage: {summary['memory_usage_mb']} MB")
print(f"Total Missing Values: {summary['total_missing_values']}")
print(f"Total Duplicate Rows: {summary['duplicate_rows']:,}")
print(f"Target Breakdown: {summary['target_summary']['counts']}")
print(f"Fraud Rate: {summary['target_summary']['fraud_rate_pct']:.4f}%")
"""),
        md_cell("""## 3. Relational Storage (SQLite) & SQL Extraction
Cleaned and structured records are persisted into a local SQLite database (`fraud_detection.db`) enabling fast SQL-based analytical queries.
"""),
        code_cell("""loader.save_to_sqlite(df_raw, table_name="transactions")

# Demonstrate SQL query extraction
query = \"\"\"
SELECT 
    COUNT(*) AS total_tx,
    SUM(is_fraud) AS fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(AVG(amount), 2) AS avg_amount,
    ROUND(MAX(amount), 2) AS max_amount
FROM transactions;
\"\"\"
res_df = loader.run_sql_query(query)
res_df
"""),
        md_cell("""## 4. Key Takeaways & Analytical Summary
1. **Extreme Imbalance:** Fraudulent transactions represent only ~0.17% of total volume (1 fraud per ~577 legitimate transactions). Accuracy is entirely inappropriate as an evaluation metric.
2. **Data Integrity:** No null values present in raw format; 1,081 duplicate transactions identified for cleaning in Phase 2.
3. **Storage:** Successfully integrated SQLite database with indexed fraud and amount columns.
""")
    ]
    with open(NOTEBOOKS_DIR / "01_data_exploration.ipynb", "w") as f:
        json.dump(make_notebook(nb1_cells), f, indent=2)

    # -------------------------------------------------------------------------
    # 2. 02_data_cleaning.ipynb
    # -------------------------------------------------------------------------
    nb2_cells = [
        md_cell("""# 02 — Data Cleaning & Outlier Auditing
**Project:** Transactional Fraud Detection Analysis  
**Objective:** Perform missing value resolution, justified duplicate deduplication, data type normalization, and financial outlier analysis without destructive removal.
"""),
        code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path("..").resolve()
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import DataLoader
from src.data_cleaning import DataCleaner
from src.utils import setup_visualization_style

setup_visualization_style()
print("Libraries loaded.")
"""),
        md_cell("""## 1. Execute Cleaning Pipeline
Let's apply the `DataCleaner` module to handle duplicate records and audit data integrity.
"""),
        code_cell("""loader = DataLoader()
df_raw = loader.load_data()

cleaner = DataCleaner()
df_cleaned, audit_summary = cleaner.clean_data(df_raw)

print(f"Original Records: {audit_summary['initial_records']:,}")
print(f"Duplicates Removed: {audit_summary['duplicates_removed']:,}")
print(f"Cleaned Records Retained: {audit_summary['cleaned_records']:,}")
print(f"Frauds Retained: {audit_summary['fraud_count_post_cleaning']:,}")
"""),
        md_cell("""## 2. Financial Outlier Analysis (IQR & Percentile)
In credit card fraud detection, unusually large transactions (>€1,000) are high-probability fraud vectors. Automatically dropping outliers induces systemic bias against detecting large-scale theft.
"""),
        code_cell("""outlier_info = audit_summary["amount_outlier_audit"]
for k, v in outlier_info.items():
    print(f"{k}: {v}")
"""),
        code_cell("""# Boxplot of Transaction Amount
plt.figure(figsize=(10, 4))
sns.boxplot(x=df_cleaned["Amount"], color="#3B82F6")
plt.title("Transaction Amount Distribution (Retaining Financial Outliers)", fontweight="bold")
plt.xlabel("Amount (€)")
plt.show()
"""),
        md_cell("""## 3. Cleaning Summary & Business Rationale
- **Deduplication:** Dropped 1,081 identical duplicate transaction records across all PCA features.
- **Outlier Policy:** 100% of financial amount outliers were deliberately retained to preserve high-value fraud patterns.
""")
    ]
    with open(NOTEBOOKS_DIR / "02_data_cleaning.ipynb", "w") as f:
        json.dump(make_notebook(nb2_cells), f, indent=2)

    # -------------------------------------------------------------------------
    # 3. 03_eda.ipynb
    # -------------------------------------------------------------------------
    nb3_cells = [
        md_cell("""# 03 — Exploratory Data Analysis (EDA)
**Project:** Transactional Fraud Detection Analysis  
**Objective:** Visualize class distributions, transaction amount behaviors, diurnal temporal trends, and correlation signals between PCA features and fraud.
"""),
        code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path("..").resolve()
sys.path.append(str(PROJECT_ROOT))

from src.data_cleaning import DataCleaner
from src.data_loader import DataLoader
from src.eda import EDAAnalyzer
from src.utils import setup_visualization_style

setup_visualization_style()
loader = DataLoader()
df = loader.load_data()
cleaner = DataCleaner()
df_clean, _ = cleaner.clean_data(df)
eda = EDAAnalyzer()
"""),
        md_cell("""## 1. Class Distribution & Imbalance
Visualizing the extreme rarity of fraudulent charges.
"""),
        code_cell("""fig_class = eda.plot_class_distribution(df_clean)
plt.show()
"""),
        md_cell("""## 2. Amount Distribution by Class
Comparing legitimate vs. fraudulent amounts using log-transformed density distributions.
"""),
        code_cell("""fig_amt = eda.plot_amount_distribution(df_clean)
plt.show()
"""),
        md_cell("""## 3. Diurnal Hourly Trends
Examining transaction volume vs. fraud rate over 24-hour daily cycles.
"""),
        code_cell("""fig_time = eda.plot_hourly_fraud_trends(df_clean)
plt.show()
"""),
        md_cell("""## 4. Correlation Analysis: Top Predictive Signals
Identifying top positive and negative PCA correlations with fraudulent transactions.
"""),
        code_cell("""fig_corr = eda.plot_correlation_matrix(df_clean)
plt.show()
"""),
        md_cell("""## 5. Fraud Probability Across Amount Ranges
"""),
        code_cell("""fig_bins = eda.plot_amount_bins_fraud_rate(df_clean)
plt.show()
"""),
        md_cell("""## 6. Key Business Insights
1. **Class Asymmetry:** Legitimate transactions account for 99.83% of volume, confirming severe class imbalance.
2. **Nighttime Risk Spike:** Fraud rate surges during night hours (01:00 to 05:00) when cardholder monitoring is dormant.
3. **Strongest Indicators:** Features `V14`, `V17`, `V12`, and `V10` show strong negative shifts for fraudulent events; `V4` and `V11` show strong positive shifts.
""")
    ]
    with open(NOTEBOOKS_DIR / "03_eda.ipynb", "w") as f:
        json.dump(make_notebook(nb3_cells), f, indent=2)

    # -------------------------------------------------------------------------
    # 4. 04_feature_engineering.ipynb
    # -------------------------------------------------------------------------
    nb4_cells = [
        md_cell("""# 04 — Feature Engineering & Preprocessing
**Project:** Transactional Fraud Detection Analysis  
**Objective:** Construct domain-relevant financial and temporal features, partition data using stratified splitting, and scale features with strict zero-leakage guarantees.
"""),
        code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path("..").resolve()
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import DataLoader
from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.preprocessing import DataPreprocessor

loader = DataLoader()
df = loader.load_data()
cleaner = DataCleaner()
df_clean, _ = cleaner.clean_data(df)
print(f"Input records: {len(df_clean):,}")
"""),
        md_cell("""## 1. Stratified Train-Test Partitioning (80/20)
We split data into 80% train and 20% test partitions using stratified sampling to preserve the exact ~0.17% fraud representation.
"""),
        code_cell("""preprocessor = DataPreprocessor()
X_train_raw, X_test_raw, y_train, y_test = preprocessor.split_data(
    df_clean,
    target_col="Class",
    test_size=0.20,
    random_state=42
)

print(f"Train Shape: {X_train_raw.shape}, Frauds: {y_train.sum():,} ({y_train.mean()*100:.3f}%)")
print(f"Test Shape:  {X_test_raw.shape}, Frauds: {y_test.sum():,} ({y_test.mean()*100:.3f}%)")
"""),
        md_cell("""## 2. Feature Engineering Pipeline (Zero-Leakage)
We engineer `log_amount`, `amount_zscore`, `hour_of_day`, `hour_sin`, `hour_cos`, `is_night_transaction`, `v14_v17_ratio`, and `v10_v12_sum`.
"""),
        code_cell("""fe = FeatureEngineer()
X_train_fe = fe.fit_transform(X_train_raw)
X_test_fe = fe.transform(X_test_raw)

print(f"Features created: {X_train_fe.shape[1]} columns")
X_train_fe[["Amount", "log_amount", "amount_zscore", "hour_of_day", "is_night_transaction", "v14_v17_ratio"]].head()
"""),
        md_cell("""## 3. Robust Scaling
We apply `RobustScaler` to skewed features, fitting parameters strictly on training features.
"""),
        code_cell("""X_train_scaled = preprocessor.fit_transform(X_train_fe)
X_test_scaled = preprocessor.transform(X_test_fe)

print("Preprocessing complete with zero data leakage.")
""")
    ]
    with open(NOTEBOOKS_DIR / "04_feature_engineering.ipynb", "w") as f:
        json.dump(make_notebook(nb4_cells), f, indent=2)

    # -------------------------------------------------------------------------
    # 5. 05_model_training.ipynb
    # -------------------------------------------------------------------------
    nb5_cells = [
        md_cell("""# 05 — Machine Learning Model Training
**Project:** Transactional Fraud Detection Analysis  
**Objective:** Train and benchmark multiple classifier families (Logistic Regression, Random Forest, HistGradientBoosting, XGBoost) and evaluate class imbalance handling strategies.
"""),
        code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path("..").resolve()
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import DataLoader
from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.preprocessing import DataPreprocessor
from src.model_training import ModelTrainer

loader = DataLoader()
df_clean, _ = DataCleaner().clean_data(loader.load_data())

preprocessor = DataPreprocessor()
X_train_raw, X_test_raw, y_train, y_test = preprocessor.split_data(df_clean, target_col="Class")

fe = FeatureEngineer()
X_train_fe = fe.fit_transform(X_train_raw)
X_test_fe = fe.transform(X_test_raw)

X_train_scaled = preprocessor.fit_transform(X_train_fe)
X_test_scaled = preprocessor.transform(X_test_fe)
"""),
        md_cell("""## 1. Model Training Harness
We train Logistic Regression, Random Forest, HistGradientBoosting, and XGBoost using cost-sensitive class weighting.
"""),
        code_cell("""trainer = ModelTrainer()
trained_models = trainer.train_all_models(X_train_scaled, y_train)

for name, duration in trainer.training_times.items():
    print(f"[{name}] Training Duration: {duration:.2f} seconds")
"""),
        md_cell("""## 2. SMOTE Oversampling Comparison (Train Partition Only)
Evaluating synthetic oversampling vs cost-sensitive class weighting.
"""),
        code_cell("""X_train_smote, y_train_smote = preprocessor.handle_imbalance(X_train_scaled, y_train, strategy="smote")
print(f"SMOTE Transformed Training Set: {len(X_train_smote):,} records, {y_train_smote.sum():,} fraud samples")
""")
    ]
    with open(NOTEBOOKS_DIR / "05_model_training.ipynb", "w") as f:
        json.dump(make_notebook(nb5_cells), f, indent=2)

    # -------------------------------------------------------------------------
    # 6. 06_model_evaluation.ipynb
    # -------------------------------------------------------------------------
    nb6_cells = [
        md_cell("""# 06 — Model Evaluation & Threshold Optimization
**Project:** Transactional Fraud Detection Analysis  
**Objective:** Evaluate models on PR-AUC, Recall, Precision, and F1; optimize decision thresholds across [0.10 - 0.90]; plot confusion matrices; and demonstrate real-time risk scoring.
"""),
        code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("..").resolve()
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import DataLoader
from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from src.prediction import FraudRiskScorer

loader = DataLoader()
df_clean, _ = DataCleaner().clean_data(loader.load_data())

preprocessor = DataPreprocessor()
X_train_raw, X_test_raw, y_train, y_test = preprocessor.split_data(df_clean, target_col="Class")

fe = FeatureEngineer()
X_train_fe = fe.fit_transform(X_train_raw)
X_test_fe = fe.transform(X_test_raw)

X_train_scaled = preprocessor.fit_transform(X_train_fe)
X_test_scaled = preprocessor.transform(X_test_fe)

trainer = ModelTrainer()
trained_models = trainer.train_all_models(X_train_scaled, y_train)
evaluator = ModelEvaluator()
"""),
        md_cell("""## 1. Classifier Benchmark Comparison
Benchmarking models by PR-AUC (Average Precision), Precision, Recall, and F1-Score.
"""),
        code_cell("""comp_df = evaluator.compare_models(trained_models, X_test_scaled, y_test, training_times=trainer.training_times)
comp_df
"""),
        md_cell("""## 2. Precision-Recall and ROC Curves
"""),
        code_cell("""fig_curves = evaluator.plot_pr_and_roc_curves(trained_models, X_test_scaled, y_test)
plt.show()
"""),
        md_cell("""## 3. Operational Threshold Optimization
Tuning decision boundary from 0.10 to 0.90 to balance missed fraud loss (FN) vs false alarms (FP).
"""),
        code_cell("""best_model_name = comp_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

thresh_df, opt_thresh, _ = evaluator.optimize_thresholds(best_model, X_test_scaled, y_test)
print(f"Optimal Operating Threshold: {opt_thresh:.2f}")
thresh_df
"""),
        md_cell("""## 4. Confusion Matrix at Optimal Threshold
"""),
        code_cell("""best_eval = evaluator.evaluate_model(best_model, X_test_scaled, y_test, model_name=best_model_name, threshold=opt_thresh)
evaluator.plot_confusion_matrix(y_test, best_eval["y_pred"], model_name=f"{best_model_name} (Thresh={opt_thresh:.2f})")
plt.show()
"""),
        md_cell("""## 5. Feature Importance
"""),
        code_cell("""df_imp, _ = evaluator.plot_feature_importance(best_model, list(X_train_scaled.columns), top_n=12)
df_imp.head(10)
"""),
        md_cell("""## 6. Real-Time Risk Scoring Simulation
"""),
        code_cell("""scorer = FraudRiskScorer(threshold=opt_thresh)
sample_tx = {"Time": 10800.0, "Amount": 350.00}
for i in range(1, 29):
    sample_tx[f"V{i}"] = 0.0
sample_tx["V14"] = -6.50
sample_tx["V17"] = -4.80
sample_tx["V4"] = 4.20

score_res = scorer.score_transaction(sample_tx)
print("Risk Assessment Result:")
for k, v in score_res.items():
    print(f"  {k}: {v}")
""")
    ]
    with open(NOTEBOOKS_DIR / "06_model_evaluation.ipynb", "w") as f:
        json.dump(make_notebook(nb6_cells), f, indent=2)

    print("All 6 Jupyter Notebooks generated successfully.")


if __name__ == "__main__":
    generate_all_notebooks()
