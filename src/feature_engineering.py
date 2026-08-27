"""
Feature Engineering Pipeline
Constructs domain-relevant financial and temporal features,
engineered with strict separation of train/test statistics to prevent data leakage.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

from src.utils import get_logger

logger = get_logger("FeatureEngineering")


class FeatureEngineer:
    """
    Transforms raw transactional data into high-signal feature matrices.
    Maintains train-fitted statistics for leakage-free validation and production inference.
    """

    def __init__(self):
        self.amount_mean: float = 0.0
        self.amount_std: float = 1.0
        self.fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """
        Compute baseline training distribution parameters (e.g. mean, std for z-scoring).
        """
        if "Amount" in df.columns:
            self.amount_mean = float(df["Amount"].mean())
            self.amount_std = float(df["Amount"].std()) if df["Amount"].std() > 0 else 1.0
        self.fitted = True
        logger.info(f"FeatureEngineer fitted on training data: Amount Mean={self.amount_mean:.2f}, Std={self.amount_std:.2f}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate engineered features on incoming dataframe.
        """
        df_out = df.copy()

        # 1. Amount Transformations
        if "Amount" in df_out.columns:
            df_out["log_amount"] = np.log1p(df_out["Amount"])
            # Z-Score using training parameters (avoids test-set snooping)
            mean_val = self.amount_mean if self.fitted else df_out["Amount"].mean()
            std_val = self.amount_std if self.fitted else (df_out["Amount"].std() if df_out["Amount"].std() > 0 else 1.0)
            df_out["amount_zscore"] = (df_out["Amount"] - mean_val) / std_val

        # 2. Temporal Features
        if "Time" in df_out.columns:
            df_out["hour_of_day"] = ((df_out["Time"] // 3600) % 24).astype(int)
            df_out["day_number"] = ((df_out["Time"] // 86400) + 1).astype(int)
            
            # Cyclical encoding of Hour of Day for continuous circular representation
            df_out["hour_sin"] = np.sin(2 * np.pi * df_out["hour_of_day"] / 24.0)
            df_out["hour_cos"] = np.cos(2 * np.pi * df_out["hour_of_day"] / 24.0)

            # High-risk nighttime transfer indicator (00:00 to 06:00)
            df_out["is_night_transaction"] = (df_out["hour_of_day"] < 6).astype(int)

        # 3. High-Signal PCA Composite & Interaction Features
        # V14, V17, V12, V10 have high negative correlation with fraud; V4, V11 have high positive correlation
        for col in ["V14", "V17", "V12", "V10", "V4", "V11"]:
            if col not in df_out.columns:
                # If lowercase exists, rename
                if col.lower() in df_out.columns:
                    df_out[col] = df_out[col.lower()]

        if all(col in df_out.columns for col in ["V14", "V17"]):
            df_out["v14_v17_ratio"] = df_out["V14"] / (df_out["V17"].abs() + 1e-5)

        if all(col in df_out.columns for col in ["V10", "V12"]):
            df_out["v10_v12_sum"] = df_out["V10"] + df_out["V12"]

        if all(col in df_out.columns for col in ["V4", "V11"]):
            df_out["v4_v11_sum"] = df_out["V4"] + df_out["V11"]

        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit feature generator on training dataset and return transformed features.
        """
        return self.fit(df).transform(df)
