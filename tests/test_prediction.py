"""
Unit Tests for Fraud Risk Scoring and Prediction Engine
"""

import pytest
import pandas as pd
import numpy as np
from src.prediction import FraudRiskScorer


def test_risk_tier_classification():
    scorer = FraudRiskScorer()
    assert scorer.classify_risk_tier(0.10) == "Low"
    assert scorer.classify_risk_tier(0.45) == "Medium"
    assert scorer.classify_risk_tier(0.72) == "High"
    assert scorer.classify_risk_tier(0.95) == "Critical"


def test_recommended_action_mapping():
    scorer = FraudRiskScorer()
    assert "Auto-Approve" in scorer.get_recommended_action("Low")
    assert "2FA" in scorer.get_recommended_action("High") or "Step-up" in scorer.get_recommended_action("High")
    assert "Decline" in scorer.get_recommended_action("Critical") or "Alert" in scorer.get_recommended_action("Critical")


def test_invalid_input_raises_error():
    scorer = FraudRiskScorer()
    # Missing required 'Amount' field
    invalid_tx = {"Time": 1000.0, "V1": 0.5}
    with pytest.raises(ValueError):
        scorer.score_transaction(invalid_tx)
