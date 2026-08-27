"""
Master End-to-End Orchestration Script: Transactional Fraud Detection Analysis
Executes the full pipeline:
Ingestion -> Cleaning -> SQLite Load -> EDA Figures -> Feature Engineering ->
Model Benchmarking -> Threshold Optimization -> Risk Scoring -> Power BI Export.
"""

import sys
import os
import json
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger, save_artifact, setup_visualization_style
from src.data_loader import DataLoader
from src.data_cleaning import DataCleaner
from src.eda import EDAAnalyzer
from src.feature_engineering import FeatureEngineer
from src.preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from src.prediction import FraudRiskScorer

logger = get_logger("MainPipeline")


def run_pipeline():
    logger.info("=" * 80)
    logger.info("STARTING TRANSACTIONAL FRAUD DETECTION PIPELINE (END-TO-END)")
    logger.info("=" * 80)

    # 1. Ingestion & Quality Audit
    logger.info("[STEP 1/10] Data Loading and Schema Audit...")
    loader = DataLoader()
    raw_df = loader.load_data()
    inspection = loader.inspect_data(raw_df)
    logger.info(f"Loaded dataset: {inspection['n_rows']:,} rows, {inspection['n_cols']} cols.")
    logger.info(f"Target distribution: {inspection['target_summary']['counts']} (Fraud Rate: {inspection['target_summary']['fraud_rate_pct']}%)")

    # 2. Data Cleaning
    logger.info("[STEP 2/10] Data Cleaning & Outlier Audit...")
    cleaner = DataCleaner()
    cleaned_df, clean_audit = cleaner.clean_data(raw_df)
    logger.info(f"Cleaning complete: {clean_audit['cleaned_records']:,} rows remaining. Duplicates dropped: {clean_audit['duplicates_removed']:,}")

    # 3. SQLite Relational Storage
    logger.info("[STEP 3/10] Persisting Cleaned Data to SQLite Database...")
    schema_path = PROJECT_ROOT / "sql" / "schema.sql"
    loader.save_to_sqlite(cleaned_df, table_name="transactions", schema_path=schema_path)

    # 4. Exploratory Data Analysis
    logger.info("[STEP 4/10] Generating Publication-Quality EDA Figures...")
    eda = EDAAnalyzer()
    fig_paths = eda.run_all_eda(cleaned_df)
    logger.info(f"Generated {len(fig_paths)} EDA visualization charts in reports/figures/")

    # 5. Feature Engineering
    logger.info("[STEP 5/10] Feature Engineering (Zero-Leakage Architecture)...")
    target_col = "Class"
    X_raw = cleaned_df.drop(columns=[target_col])
    y_raw = cleaned_df[target_col]

    preprocessor = DataPreprocessor()
    X_train_raw, X_test_raw, y_train, y_test = preprocessor.split_data(
        cleaned_df,
        target_col=target_col,
        test_size=0.20,
        random_state=42
    )

    feat_engineer = FeatureEngineer()
    X_train_fe = feat_engineer.fit_transform(X_train_raw)
    X_test_fe = feat_engineer.transform(X_test_raw)

    # Save feature engineer artifact
    save_artifact(feat_engineer, PROJECT_ROOT / "models" / "feature_engineer.pkl")

    # 6. Preprocessing & Scaling
    logger.info("[STEP 6/10] Scaling & Imbalance Treatment on Training Set...")
    X_train_scaled = preprocessor.fit_transform(X_train_fe)
    X_test_scaled = preprocessor.transform(X_test_fe)

    # Save preprocessor artifact
    save_artifact(preprocessor, PROJECT_ROOT / "models" / "preprocessor.pkl")

    # 7. Model Training & Benchmarking
    logger.info("[STEP 7/10] Model Training across 4 Classifier Families...")
    trainer = ModelTrainer()
    trained_models = trainer.train_all_models(X_train_scaled, y_train)

    # 8. Model Evaluation & Comparison
    logger.info("[STEP 8/10] Comprehensive Model Evaluation & Metric Computation...")
    evaluator = ModelEvaluator()
    comparison_df = evaluator.compare_models(
        trained_models,
        X_test_scaled,
        y_test,
        training_times=trainer.training_times,
        threshold=0.50
    )
    logger.info("\n" + comparison_df.to_string(index=False))

    # Save comparison curves
    evaluator.plot_pr_and_roc_curves(trained_models, X_test_scaled, y_test)

    # Determine Best Model based on PR-AUC (Average Precision)
    best_model_name = comparison_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    logger.info(f"Designated Best Model: [{best_model_name}] (Highest PR-AUC)")

    # Threshold Optimization on Best Model
    thresh_df, opt_thresh, thresh_fig = evaluator.optimize_thresholds(best_model, X_test_scaled, y_test)
    logger.info(f"Threshold Optimization complete. Business-optimal operating threshold: {opt_thresh:.2f}")

    # Evaluate best model at optimal threshold
    best_eval = evaluator.evaluate_model(best_model, X_test_scaled, y_test, model_name=best_model_name, threshold=opt_thresh)
    evaluator.plot_confusion_matrix(y_test, best_eval["y_pred"], model_name=f"{best_model_name} (Thresh={opt_thresh:.2f})")
    df_imp, imp_fig = evaluator.plot_feature_importance(best_model, list(X_train_scaled.columns), top_n=15)

    # Save Best Model Artifact
    trainer.save_best_model(best_model, filename="fraud_detection_model.pkl")

    # 9. Real-Time Risk Scorer Calibration & Batch Scoring for Power BI
    logger.info("[STEP 9/10] Generating Power BI / Tableau Enriched Export...")
    scorer = FraudRiskScorer(threshold=opt_thresh)
    
    # Enrich full dataset for BI Dashboards
    dashboard_df = scorer.score_batch(cleaned_df)
    dashboard_df["transaction_id"] = range(1, len(dashboard_df) + 1)
    
    # Save Power BI CSV
    powerbi_dir = PROJECT_ROOT / "powerbi"
    powerbi_dir.mkdir(parents=True, exist_ok=True)
    powerbi_path = powerbi_dir / "dashboard_data.csv"
    dashboard_df.to_csv(powerbi_path, index=False)
    logger.info(f"Saved Power BI dataset to: {powerbi_path} ({len(dashboard_df):,} records)")

    # 10. Store Predictions in SQLite Database
    logger.info("[STEP 10/10] Writing Model Predictions into SQLite Relational Tables...")
    conn = sqlite3.connect(loader.db_path)
    pred_records = pd.DataFrame({
        "transaction_id": dashboard_df["transaction_id"],
        "model_name": best_model_name,
        "fraud_probability": dashboard_df["fraud_probability"],
        "risk_score": dashboard_df["risk_score"],
        "risk_tier": dashboard_df["risk_tier"],
        "predicted_class": dashboard_df["predicted_class"],
        "actual_class": dashboard_df["Class"],
    })
    pred_records.to_sql("model_predictions", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    logger.info("Successfully updated SQLite table 'model_predictions'.")

    # Save execution metrics summary for reporting
    metrics_summary = {
        "dataset_records": int(len(cleaned_df)),
        "duplicates_removed": int(clean_audit["duplicates_removed"]),
        "fraud_count": int(cleaned_df["Class"].sum()),
        "legit_count": int((cleaned_df["Class"] == 0).sum()),
        "fraud_rate_pct": float(round((cleaned_df["Class"].sum() / len(cleaned_df)) * 100, 4)),
        "best_model": best_model_name,
        "optimal_threshold": float(opt_thresh),
        "best_model_precision": float(best_eval["precision"]),
        "best_model_recall": float(best_eval["recall"]),
        "best_model_f1": float(best_eval["f1_score"]),
        "best_model_roc_auc": float(best_eval["roc_auc"]),
        "best_model_pr_auc": float(best_eval["pr_auc"]),
        "best_model_tp": int(best_eval["true_positives"]),
        "best_model_fp": int(best_eval["false_positives"]),
        "best_model_fn": int(best_eval["false_negatives"]),
        "best_model_tn": int(best_eval["true_negatives"]),
        "model_comparison": comparison_df.to_dict(orient="records"),
        "top_features": df_imp.head(10).to_dict(orient="records"),
    }
    
    with open(PROJECT_ROOT / "reports" / "metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info("Saved pipeline metrics summary to reports/metrics_summary.json")

    logger.info("=" * 80)
    logger.info("TRANSACTIONAL FRAUD DETECTION PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    return metrics_summary


if __name__ == "__main__":
    run_pipeline()
