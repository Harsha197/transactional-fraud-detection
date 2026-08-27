"""
Machine Learning Preprocessing Pipeline
Enforces strict stratified train/test partitioning, zero-leakage scaling,
and multiple class imbalance mitigation strategies (Baseline, Class-Weighting, SMOTE).
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from imblearn.over_sampling import SMOTE

from src.utils import get_logger

logger = get_logger("Preprocessing")


class DataPreprocessor:
    """
    Leakage-free preprocessing engine for financial fraud datasets.
    Scalers and oversamplers are strictly fit on the training partition only.
    """

    def __init__(self, scale_features: Optional[List[str]] = None):
        self.scale_features = scale_features or ["Amount", "log_amount", "amount_zscore", "Time"]
        self.scaler = RobustScaler()  # RobustScaler is robust to extreme financial outliers
        self.feature_columns: List[str] = []
        self.target_column: str = "Class"
        self.fitted: bool = False

    def split_data(
        self,
        df: pd.DataFrame,
        target_col: str = "Class",
        test_size: float = 0.20,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split transaction dataset into train and test sets using Stratified Splitting
        to preserve the rare fraud ratio (~0.17%) across both partitions.
        """
        self.target_column = target_col
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")

        X = df.drop(columns=[target_col])
        y = df[target_col]
        self.feature_columns = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
            shuffle=True,
        )

        logger.info(
            f"Stratified Train-Test Split (80/20): "
            f"Train={len(X_train):,} ({y_train.sum():,} frauds, {y_train.mean()*100:.3f}%), "
            f"Test={len(X_test):,} ({y_test.sum():,} frauds, {y_test.mean()*100:.3f}%)"
        )
        return X_train, X_test, y_train, y_test

    def fit(self, X_train: pd.DataFrame) -> "DataPreprocessor":
        """
        Fit scaler strictly on training features (avoids data snooping / leakage).
        """
        cols_to_scale = [col for col in self.scale_features if col in X_train.columns]
        if cols_to_scale:
            self.scaler.fit(X_train[cols_to_scale])
            logger.info(f"Fitted RobustScaler on training columns: {cols_to_scale}")
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform numerical features using the pre-fitted scaler.
        """
        if not self.fitted:
            raise RuntimeError("DataPreprocessor must be fit on training data before calling transform.")

        X_out = X.copy()
        cols_to_scale = [col for col in self.scale_features if col in X_out.columns]
        if cols_to_scale:
            X_out[cols_to_scale] = self.scaler.transform(X_out[cols_to_scale])
        return X_out

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Fit scaler and transform training features.
        """
        return self.fit(X_train).transform(X_train)

    def handle_imbalance(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        strategy: str = "smote",
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply class imbalance remediation strictly on the training partition.
        Strategies:
            - 'none': Unaltered baseline
            - 'smote': Synthetic Minority Over-sampling Technique
        """
        if strategy == "none":
            logger.info("Using baseline unweighted training distribution.")
            return X_train, y_train

        elif strategy == "smote":
            logger.info("Applying SMOTE oversampling to minority class on training partition...")
            # Use sampling_strategy=0.10 or 0.20 to avoid synthetic over-saturation while boosting signal
            smote = SMOTE(sampling_strategy=0.10, random_state=random_state)
            X_res, y_res = smote.fit_resample(X_train, y_train)
            logger.info(
                f"SMOTE complete: New train size={len(X_res):,}, "
                f"Frauds={int(y_res.sum()):,} ({y_res.mean()*100:.2f}%)"
            )
            return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res, name=y_train.name)

        else:
            raise ValueError(f"Unknown imbalance handling strategy: {strategy}")
