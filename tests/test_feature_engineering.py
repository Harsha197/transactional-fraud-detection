"""
Unit Tests for Feature Engineering Module
"""

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_fe_df():
    np.random.seed(42)
    n = 50
    data = {
        "Time": np.array([0, 3600, 7200, 10800, 43200, 86400] * 8 + [0, 3600]),
        "Amount": np.random.uniform(5, 500, n),
        "Class": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
    }
    for i in range(1, 29):
        data[f"V{i}"] = np.random.normal(0, 1, n)
    return pd.DataFrame(data)


def test_feature_engineering_creates_expected_columns(sample_fe_df):
    fe = FeatureEngineer()
    df_transformed = fe.fit_transform(sample_fe_df)

    expected_cols = [
        "log_amount",
        "amount_zscore",
        "hour_of_day",
        "day_number",
        "hour_sin",
        "hour_cos",
        "is_night_transaction",
        "v14_v17_ratio",
        "v10_v12_sum",
        "v4_v11_sum",
    ]
    for col in expected_cols:
        assert col in df_transformed.columns, f"Missing engineered column: {col}"


def test_zero_leakage_transform(sample_fe_df):
    fe = FeatureEngineer()
    train_df = sample_fe_df.iloc[:30]
    test_df = sample_fe_df.iloc[30:]

    fe.fit(train_df)
    train_mean = fe.amount_mean

    # Transform test set
    test_transformed = fe.transform(test_df)

    # Ensure test z-score was computed using training mean
    expected_zscore = (test_df["Amount"].iloc[0] - train_mean) / fe.amount_std
    assert np.isclose(test_transformed["amount_zscore"].iloc[0], expected_zscore, atol=1e-4)
