-- ==============================================================================
-- TRANSACTIONAL FRAUD DETECTION DATABASE SCHEMA (SQLite)
-- Schema Definition for Financial Transaction Analytics & Model Monitoring
-- ==============================================================================

-- Drop tables if they exist for clean rebuilds
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS model_predictions;
DROP TABLE IF EXISTS fraud_risk_summary;

-- Primary Transaction Table
CREATE TABLE transactions (
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    time_sec REAL NOT NULL,
    hour_of_day INTEGER,
    day_number INTEGER,
    time_period TEXT,
    v1 REAL, v2 REAL, v3 REAL, v4 REAL, v5 REAL,
    v6 REAL, v7 REAL, v8 REAL, v9 REAL, v10 REAL,
    v11 REAL, v12 REAL, v13 REAL, v14 REAL, v15 REAL,
    v16 REAL, v17 REAL, v18 REAL, v19 REAL, v20 REAL,
    v21 REAL, v22 REAL, v23 REAL, v24 REAL, v25 REAL,
    v26 REAL, v27 REAL, v28 REAL,
    amount REAL NOT NULL,
    log_amount REAL,
    is_fraud INTEGER NOT NULL CHECK (is_fraud IN (0, 1)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexing for high-performance analytical queries
CREATE INDEX idx_transactions_is_fraud ON transactions (is_fraud);
CREATE INDEX idx_transactions_amount ON transactions (amount);
CREATE INDEX idx_transactions_hour ON transactions (hour_of_day);
CREATE INDEX idx_transactions_time_period ON transactions (time_period);

-- Model Predictions and Risk Scoring Table
CREATE TABLE model_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    fraud_probability REAL NOT NULL,
    risk_score REAL NOT NULL,
    risk_tier TEXT NOT NULL CHECK (risk_tier IN ('Low', 'Medium', 'High', 'Critical')),
    predicted_class INTEGER NOT NULL CHECK (predicted_class IN (0, 1)),
    actual_class INTEGER NOT NULL CHECK (actual_class IN (0, 1)),
    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (transaction_id) REFERENCES transactions (transaction_id)
);

CREATE INDEX idx_predictions_model ON model_predictions (model_name);
CREATE INDEX idx_predictions_risk_tier ON model_predictions (risk_tier);
