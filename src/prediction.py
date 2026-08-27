"""
Real-Time Style Fraud Risk Scoring and Prediction Engine
Converts raw transaction inputs into calibrated risk scores, operational tiers,
fraud flags, and actionable mitigation recommendations.
"""

from pathlib import Path
from typing import Dict, Any, Union, Optional
import pandas as pd
import numpy as np

from src.utils import get_logger, load_artifact, get_project_root

logger = get_logger("Prediction")


class FraudRiskScorer:
    """
    Inference and Risk Calibration Engine.
    Demonstrates real-time risk decisioning for incoming payment streams.
    """

    DEFAULT_MODEL_PATH = get_project_root() / "models" / "fraud_detection_model.pkl"
    DEFAULT_PREPROCESSOR_PATH = get_project_root() / "models" / "preprocessor.pkl"
    DEFAULT_FEATURE_ENGINEER_PATH = get_project_root() / "models" / "feature_engineer.pkl"

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        preprocessor_path: Optional[Path | str] = None,
        feature_engineer_path: Optional[Path | str] = None,
        threshold: float = 0.50,
    ):
        self.model_path = Path(model_path) if model_path else self.DEFAULT_MODEL_PATH
        self.preprocessor_path = Path(preprocessor_path) if preprocessor_path else self.DEFAULT_PREPROCESSOR_PATH
        self.feature_engineer_path = Path(feature_engineer_path) if feature_engineer_path else self.DEFAULT_FEATURE_ENGINEER_PATH
        self.threshold = threshold

        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """
        Load serialized ML artifacts if they exist on disk.
        """
        if self.model_path.exists():
            self.model = load_artifact(self.model_path)
            logger.info(f"Loaded ML model from: {self.model_path}")
        if self.preprocessor_path.exists():
            self.preprocessor = load_artifact(self.preprocessor_path)
        if self.feature_engineer_path.exists():
            self.feature_engineer = load_artifact(self.feature_engineer_path)

    def classify_risk_tier(self, probability: float) -> str:
        """
        Map probability to calibrated enterprise risk tiers:
        0.00–0.29 -> Low
        0.30–0.59 -> Medium
        0.60–0.79 -> High
        0.80–1.00 -> Critical
        """
        if probability < 0.30:
            return "Low"
        elif probability < 0.60:
            return "Medium"
        elif probability < 0.80:
            return "High"
        else:
            return "Critical"

    def get_recommended_action(self, risk_tier: str) -> str:
        """
        Generate operational policy recommendations based on risk classification.
        """
        actions = {
            "Low": "Auto-Approve transaction. Standard risk profile.",
            "Medium": "Approve with background logging; evaluate velocity triggers.",
            "High": "Step-up verification required (Prompt 2FA / OTP SMS challenge).",
            "Critical": "Immediate Transaction Decline & Trigger Real-Time Fraud Operations Alert.",
        }
        return actions.get(risk_tier, "Manual Review Required.")

    def score_transaction(self, transaction_data: Union[Dict[str, Any], pd.Series]) -> Dict[str, Any]:
        """
        Score a single transaction payload.
        Accepts a dictionary or Series with keys Time, Amount, V1-V28.
        """
        if self.model is None:
            raise RuntimeError("Model artifact is not loaded. Train the pipeline first.")

        # Convert to DataFrame
        if isinstance(transaction_data, dict):
            df_single = pd.DataFrame([transaction_data])
        else:
            df_single = pd.DataFrame([transaction_data.to_dict()])

        # Validate required numeric fields
        if "Amount" not in df_single.columns or "Time" not in df_single.columns:
            raise ValueError("Input transaction must contain 'Amount' and 'Time' fields.")

        # Apply feature engineering
        if self.feature_engineer:
            df_feat = self.feature_engineer.transform(df_single)
        else:
            df_feat = df_single.copy()
            df_feat["log_amount"] = np.log1p(df_feat["Amount"])
            df_feat["hour_of_day"] = ((df_feat["Time"] // 3600) % 24).astype(int)

        # Apply preprocessing
        if self.preprocessor:
            df_preprocessed = self.preprocessor.transform(df_feat)
        else:
            df_preprocessed = df_feat

        # Keep only model features
        if hasattr(self.model, "feature_names_in_"):
            expected_cols = list(self.model.feature_names_in_)
            # Handle missing engineered features if any
            for col in expected_cols:
                if col not in df_preprocessed.columns:
                    df_preprocessed[col] = 0.0
            df_input = df_preprocessed[expected_cols]
        else:
            df_input = df_preprocessed

        # Inference
        if hasattr(self.model, "predict_proba"):
            prob = float(self.model.predict_proba(df_input)[0, 1])
        elif hasattr(self.model, "decision_function"):
            dec = float(self.model.decision_function(df_input)[0])
            prob = float(1 / (1 + np.exp(-dec)))
        else:
            prob = float(self.model.predict(df_input)[0])

        risk_score = round(prob * 100, 1)
        risk_tier = self.classify_risk_tier(prob)
        is_fraud_flag = "SUSPICIOUS / FRAUD" if prob >= self.threshold else "LEGITIMATE"
        recommended_action = self.get_recommended_action(risk_tier)

        result = {
            "amount": float(df_single["Amount"].iloc[0]),
            "time_sec": float(df_single["Time"].iloc[0]),
            "fraud_probability": round(prob, 4),
            "fraud_probability_pct": f"{prob * 100:.2f}%",
            "risk_score": risk_score,
            "risk_tier": risk_tier,
            "fraud_flag": is_fraud_flag,
            "recommended_action": recommended_action,
            "decision_threshold": self.threshold,
        }
        return result

    def score_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a batch DataFrame and return enriched dataset for Power BI and Tableau.
        """
        if self.model is None:
            raise RuntimeError("Model artifact is not loaded.")

        df_out = df.copy()
        
        # Feature Engineering
        if self.feature_engineer:
            df_feat = self.feature_engineer.transform(df_out)
        else:
            df_feat = df_out.copy()

        # Preprocessing
        if self.preprocessor:
            df_prep = self.preprocessor.transform(df_feat)
        else:
            df_prep = df_feat

        if hasattr(self.model, "feature_names_in_"):
            expected_cols = list(self.model.feature_names_in_)
            for col in expected_cols:
                if col not in df_prep.columns:
                    df_prep[col] = 0.0
            df_input = df_prep[expected_cols]
        else:
            df_input = df_prep

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(df_input)[:, 1]
        else:
            probs = self.model.predict(df_input)

        df_out["fraud_probability"] = np.round(probs, 4)
        df_out["risk_score"] = np.round(probs * 100, 1)
        df_out["risk_tier"] = [self.classify_risk_tier(p) for p in probs]
        df_out["predicted_class"] = (probs >= self.threshold).astype(int)
        df_out["fraud_flag"] = np.where(probs >= self.threshold, "Suspicious", "Legitimate")

        return df_out
