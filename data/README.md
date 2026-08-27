# Dataset Documentation: Credit Card Transaction Fraud Detection

## Overview
This repository uses the widely benchmarked **European Credit Card Fraud Detection Dataset** (originally curated by Machine Learning Group at ULB - Université Libre de Bruxelles).

## Dataset Characteristics
- **Total Transactions:** 284,807 transactions recorded in September 2013 by European cardholders.
- **Timeframe:** 2-day duration (48 hours / 172,800 seconds).
- **Target Variable (`Class`):**
  - `0`: Legitimate transaction (284,315 records, ~99.827%)
  - `1`: Fraudulent transaction (492 records, ~0.173%)
- **Class Imbalance Ratio:** 1:577 (extreme class imbalance)

## Feature Descriptions
- `Time` (Numeric): Seconds elapsed between each transaction and the first transaction in the dataset.
- `V1` to `V28` (Numeric): Principal Component Analysis (PCA) transformed numerical features to preserve cardholder confidentiality and sensitive financial information.
- `Amount` (Numeric): Transaction amount in Euros (€).
- `Class` (Categorical/Binary Integer): 1 for fraud, 0 for legitimate.

## Storage Hierarchy
- `data/raw/creditcard.csv`: Raw, unmodified transaction records.
- `data/processed/fraud_detection.db`: Cleaned and engineered SQLite database with relational indexing for performant SQL analytics.
- `data/processed/cleaned_transactions.csv`: Cleaned dataset with validated types and normalized values.
- `powerbi/dashboard_data.csv`: Analytical star-schema export enriched with model predictions, risk tiers, and time aggregations.
