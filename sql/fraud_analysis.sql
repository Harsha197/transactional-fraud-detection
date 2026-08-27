-- ==============================================================================
-- FRAUD BEHAVIORAL & TEMPORAL ANALYSIS QUERIES
-- Analytical SQL queries extracting key patterns and statistical distributions
-- ==============================================================================

-- 1. Executive Summary: Core Fraud Metrics
SELECT 
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN is_fraud = 0 THEN 1 ELSE 0 END) AS legit_transactions,
    SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(AVG(amount), 2) AS overall_avg_amount,
    ROUND(AVG(CASE WHEN is_fraud = 1 THEN amount ELSE NULL END), 2) AS fraud_avg_amount,
    ROUND(AVG(CASE WHEN is_fraud = 0 THEN amount ELSE NULL END), 2) AS legit_avg_amount,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS total_fraud_exposure_loss
FROM transactions;

-- 2. Fraud Distribution by Transaction Amount Bracket
SELECT 
    CASE 
        WHEN amount = 0 THEN '01. Zero Amount (€0.00)'
        WHEN amount > 0 AND amount <= 10 THEN '02. Micro (€0.01 - €10.00)'
        WHEN amount > 10 AND amount <= 50 THEN '03. Small (€10.01 - €50.00)'
        WHEN amount > 50 AND amount <= 100 THEN '04. Medium (€50.01 - €100.00)'
        WHEN amount > 100 AND amount <= 500 THEN '05. High (€100.01 - €500.00)'
        WHEN amount > 500 AND amount <= 1000 THEN '06. Very High (€500.01 - €1000.00)'
        ELSE '07. Extreme Outlier (> €1000.00)'
    END AS amount_range,
    COUNT(*) AS total_volume,
    SUM(is_fraud) AS fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(SUM(amount), 2) AS total_amount,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_loss
FROM transactions
GROUP BY amount_range
ORDER BY amount_range;

-- 3. Temporal Analysis: Fraud Rate by Hour of Day
SELECT 
    hour_of_day,
    time_period,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(AVG(amount), 2) AS avg_transaction_amount,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS total_fraud_loss
FROM transactions
GROUP BY hour_of_day, time_period
ORDER BY hour_of_day;

-- 4. Fraud Concentration by Time Period (Dayparting)
SELECT 
    time_period,
    COUNT(*) AS transaction_count,
    SUM(is_fraud) AS fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(SUM(is_fraud) * 100.0 / (SELECT SUM(is_fraud) FROM transactions), 2) AS pct_of_all_frauds
FROM transactions
GROUP BY time_period
ORDER BY fraud_rate_pct DESC;

-- 5. Top 15 Highest Value Fraudulent Transactions (High-Risk Audit)
SELECT 
    transaction_id,
    time_sec,
    hour_of_day,
    time_period,
    amount,
    v1, v2, v3, v4, v14, v17
FROM transactions
WHERE is_fraud = 1
ORDER BY amount DESC
LIMIT 15;
