"""
Data Cleaning and Validation Module
Handles missing values, duplicate record deduplication with justification,
type enforcement, and financial outlier audit without destructive data removal.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

from src.utils import get_logger, get_project_root

logger = get_logger("DataCleaning")


class DataCleaner:
    """
    Production-ready data cleaning and integrity assurance pipeline
    tailored specifically for financial transactional datasets.
    """

    def __init__(self, processed_dir: Optional[Path | str] = None):
        self.processed_dir = Path(processed_dir) if processed_dir else get_project_root() / "data" / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.cleaned_csv_path = self.processed_dir / "cleaned_transactions.csv"

    def audit_outliers(self, df: pd.DataFrame, col: str = "Amount") -> Dict[str, Any]:
        """
        Perform rigorous IQR and percentile outlier auditing.
        Financial Justification: Outliers in transaction amounts must NOT be dropped blindly,
        as high-value anomalous charges are high-probability fraud vectors.
        """
        if col not in df.columns:
            return {}

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = max(0, q1 - 1.5 * iqr)
        upper_bound = q3 + 1.5 * iqr
        
        p95 = df[col].quantile(0.95)
        p99 = df[col].quantile(0.99)
        p999 = df[col].quantile(0.999)

        outliers_iqr = df[df[col] > upper_bound]
        fraud_in_outliers = int(outliers_iqr["Class"].sum()) if "Class" in df.columns else None

        audit_result = {
            "column": col,
            "q1": round(float(q1), 2),
            "q3": round(float(q3), 2),
            "iqr": round(float(iqr), 2),
            "upper_bound_iqr": round(float(upper_bound), 2),
            "p95": round(float(p95), 2),
            "p99": round(float(p99), 2),
            "p99_9": round(float(p999), 2),
            "outlier_count_iqr": int(len(outliers_iqr)),
            "outlier_pct_iqr": round(float(len(outliers_iqr) / len(df) * 100), 2),
            "fraud_cases_in_iqr_outliers": fraud_in_outliers,
            "treatment_decision": "RETAINED. Large transaction amounts contain critical fraud signal and dropping them induces systemic bias against high-value theft detection."
        }
        return audit_result

    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute full data cleaning workflow.
        Returns:
            Tuple of (cleaned_dataframe, audit_summary_dictionary)
        """
        initial_rows = len(df)
        initial_cols = df.shape[1]
        logger.info(f"Starting data cleaning pipeline on {initial_rows:,} records...")

        df_clean = df.copy()

        # 1. Missing Value Auditing & Handling
        missing_counts = df_clean.isnull().sum()
        total_missing = int(missing_counts.sum())
        missing_handling_log = "No missing values detected."

        if total_missing > 0:
            logger.warning(f"Detected {total_missing} missing values across columns.")
            # For numeric features, impute with median to prevent outlier distortion
            num_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                if df_clean[col].isnull().sum() > 0:
                    med_val = df_clean[col].median()
                    df_clean[col].fillna(med_val, inplace=True)
            missing_handling_log = f"Imputed {total_missing} missing values using robust column medians."
        
        # 2. Duplicate Detection & Handling
        duplicate_count = int(df_clean.duplicated().sum())
        duplicates_removed = 0
        if duplicate_count > 0:
            logger.info(f"Found {duplicate_count:,} duplicate records. Auditing duplicate composition...")
            # Deduplicate exact matches across all columns
            df_clean = df_clean.drop_duplicates().reset_index(drop=True)
            duplicates_removed = duplicate_count
            logger.info(f"Removed {duplicates_removed:,} exact duplicate rows. Retained {len(df_clean):,} distinct records.")

        # 3. Data Type Normalization
        if "Class" in df_clean.columns:
            df_clean["Class"] = df_clean["Class"].astype(int)
        if "Time" in df_clean.columns:
            df_clean["Time"] = df_clean["Time"].astype(float)
        if "Amount" in df_clean.columns:
            df_clean["Amount"] = df_clean["Amount"].astype(float)

        # 4. Outlier Analysis Audit
        amount_outlier_audit = self.audit_outliers(df_clean, col="Amount")

        # 5. Save Cleaned Dataset
        df_clean.to_csv(self.cleaned_csv_path, index=False)
        logger.info(f"Cleaned dataset saved to: {self.cleaned_csv_path}")

        audit_summary = {
            "initial_records": initial_rows,
            "cleaned_records": len(df_clean),
            "duplicates_removed": duplicates_removed,
            "total_missing_imputed": total_missing,
            "missing_handling_log": missing_handling_log,
            "amount_outlier_audit": amount_outlier_audit,
            "final_columns": list(df_clean.columns),
            "fraud_count_post_cleaning": int(df_clean["Class"].sum()) if "Class" in df_clean.columns else None,
            "legit_count_post_cleaning": int((df_clean["Class"] == 0).sum()) if "Class" in df_clean.columns else None,
        }

        return df_clean, audit_summary
