"""
Unit Tests for Data Cleaning Module
"""

import pytest
import pandas as pd
import numpy as np
from src.data_cleaning import DataCleaner


@pytest.fixture
def sample_raw_df():
    """Create synthetic test dataset with known duplicates, nulls, and outliers."""
    np.random.seed(42)
    n = 100
    data = {
        "Time": np.linspace(0, 1000, n),
        "Amount": np.random.exponential(scale=50, size=n),
        "Class": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
    }
    for i in range(1, 29):
        data[f"V{i}"] = np.random.normal(0, 1, n)

    df = pd.DataFrame(data)
    # Inject 5 duplicates
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    # Inject 3 nulls
    df.loc[10:12, "Amount"] = np.nan
    # Inject 2 extreme financial outliers
    df.loc[15, "Amount"] = 15000.0
    df.loc[16, "Amount"] = 25000.0
    return df


def test_clean_data_handles_nulls_and_duplicates(sample_raw_df, tmp_path):
    cleaner = DataCleaner(processed_dir=tmp_path)
    cleaned_df, audit = cleaner.clean_data(sample_raw_df)

    # Validate nulls are resolved
    assert cleaned_df.isnull().sum().sum() == 0
    assert audit["total_missing_imputed"] > 0

    # Validate exact duplicates were removed
    assert audit["duplicates_removed"] == 5
    assert len(cleaned_df) == len(sample_raw_df) - 5


def test_outliers_are_retained_with_justification(sample_raw_df, tmp_path):
    cleaner = DataCleaner(processed_dir=tmp_path)
    cleaned_df, audit = cleaner.clean_data(sample_raw_df)

    # Check outlier audit result
    outlier_audit = audit["amount_outlier_audit"]
    assert outlier_audit["outlier_count_iqr"] > 0
    assert "RETAINED" in outlier_audit["treatment_decision"]
    # Ensure extreme amounts are still present in cleaned dataframe
    assert cleaned_df["Amount"].max() >= 15000.0
