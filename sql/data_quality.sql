-- ==============================================================================
-- TRANSACTION DATA QUALITY & INTEGRITY VALIDATION QUERIES
-- Designed to identify data drift, missing values, anomalies, and duplicates
-- ==============================================================================

-- 1. Check for Missing Values across Critical Columns
SELECT
    COUNT(*) AS total_records,
    SUM(CASE WHEN time_sec IS NULL THEN 1 ELSE 0 END) AS missing_time,
    SUM(CASE WHEN amount IS NULL THEN 1 ELSE 0 END) AS missing_amount,
    SUM(CASE WHEN is_fraud IS NULL THEN 1 ELSE 0 END) AS missing_target,
    SUM(CASE WHEN v1 IS NULL THEN 1 ELSE 0 END) AS missing_v1
FROM transactions;

-- 2. Validate Amount Bounds & Negative Values (Financial Anomaly Check)
SELECT
    COUNT(*) AS invalid_amounts,
    MIN(amount) AS min_amount,
    MAX(amount) AS max_amount,
    AVG(amount) AS avg_amount
FROM transactions
WHERE amount < 0 OR amount IS NULL;

-- 3. Duplicate Transaction Identification (Exact feature matching)
SELECT
    amount,
    time_sec,
    v1, v2, v3,
    COUNT(*) AS duplicate_count
FROM transactions
GROUP BY amount, time_sec, v1, v2, v3
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC
LIMIT 10;

-- 4. Target Variable Validity Check (Ensure only binary 0 and 1 exist)
SELECT
    is_fraud,
    COUNT(*) AS record_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM transactions), 4) AS percentage
FROM transactions
GROUP BY is_fraud;

-- 5. Outlier Detection: Identify Extreme Outlier Transactions (> 99.9th Percentile)
SELECT
    transaction_id,
    time_sec,
    amount,
    is_fraud,
    CASE 
        WHEN amount > 1000 THEN 'Extreme High Value (>1000)'
        WHEN amount > 500 THEN 'High Value (500-1000)'
        WHEN amount <= 1 THEN 'Micro Transaction (<=1)'
        ELSE 'Standard Range (1-500)'
    END AS amount_bucket
FROM transactions
WHERE amount > 1000
ORDER BY amount DESC
LIMIT 20;
