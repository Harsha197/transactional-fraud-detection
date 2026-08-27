"""
Machine Learning Model Training Pipeline
Implements modular training across multiple algorithmic families:
Logistic Regression, Random Forest, HistGradientBoosting, and XGBoost.
Tracks execution duration, supports class weighting and oversampled pipelines.
"""

import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import xgboost as xgb

from src.utils import get_logger, save_artifact, get_project_root

logger = get_logger("ModelTraining")


class ModelTrainer:
    """
    Modular training harness for financial fraud detection classifiers.
    """

    def __init__(self, models_dir: Optional[Path | str] = None):
        self.models_dir = Path(models_dir) if models_dir else get_project_root() / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.trained_models: Dict[str, Any] = {}
        self.training_times: Dict[str, float] = {}

    def get_model_instances(self, pos_weight_ratio: float = 577.0) -> Dict[str, Any]:
        """
        Define classifier architectures with appropriate cost-sensitive hyperparameters.
        """
        models = {
            "Logistic Regression (Balanced)": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
                solver="lbfgs"
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1
            ),
            "HistGradientBoosting": HistGradientBoostingClassifier(
                max_iter=100,
                class_weight="balanced",
                random_state=42
            ),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=120,
                max_depth=5,
                learning_rate=0.08,
                scale_pos_weight=10.0,  # Scaled weight for severe imbalance
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            )
        }
        return models

    def train_single_model(
        self,
        name: str,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Tuple[Any, float]:
        """
        Train a single classifier instance and track exact elapsed execution time.
        """
        logger.info(f"Training [{name}] on {len(X_train):,} samples...")
        start_time = time.time()
        model.fit(X_train, y_train)
        duration = round(time.time() - start_time, 2)
        logger.info(f"[{name}] training completed in {duration:.2f} seconds.")
        return model, duration

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, Any]:
        """
        Train the complete suite of benchmark classifiers.
        """
        pos_weight = float((len(y_train) - y_train.sum()) / (y_train.sum() + 1e-5))
        models_to_train = self.get_model_instances(pos_weight_ratio=pos_weight)

        for name, model in models_to_train.items():
            fitted_model, duration = self.train_single_model(name, model, X_train, y_train)
            self.trained_models[name] = fitted_model
            self.training_times[name] = duration

        return self.trained_models

    def save_best_model(self, model: Any, filename: str = "fraud_detection_model.pkl") -> Path:
        """
        Persist the designated best-performing model to disk.
        """
        save_path = self.models_dir / filename
        save_artifact(model, save_path)
        return save_path
