-- ==============================================================================
-- BUSINESS INSIGHTS & RISK TIER AGGREGATION QUERIES
-- High-level operational queries for risk teams, management reporting & Power BI
-- ==============================================================================

-- 1. Model Risk Tier Distribution & Fraud Capture Effectiveness
SELECT 
    p.risk_tier,
    COUNT(*) AS transaction_volume,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM model_predictions), 2) AS volume_pct,
    SUM(p.actual_class) AS actual_frauds_detected,
    ROUND(SUM(p.actual_class) * 100.0 / COUNT(*), 2) AS tier_precision_pct,
    ROUND(SUM(p.actual_class) * 100.0 / (SELECT SUM(actual_class) FROM model_predictions), 2) AS fraud_capture_recall_pct,
    ROUND(AVG(t.amount), 2) AS avg_transaction_amount,
    ROUND(SUM(CASE WHEN p.actual_class = 1 THEN t.amount ELSE 0 END), 2) AS financial_loss_prevented
FROM model_predictions p
JOIN transactions t ON p.transaction_id = t.transaction_id
GROUP BY p.risk_tier
ORDER BY 
    CASE p.risk_tier
        WHEN 'Critical' THEN 1
        WHEN 'High' THEN 2
        WHEN 'Medium' THEN 3
        WHEN 'Low' THEN 4
    END;

-- 2. Business Impact: False Positive vs. False Negative Cost Analysis (Hypothetical Ops Cost)
-- Assumptions: Cost of Missed Fraud (FN) = 100% of Amount; Cost of False Alarm (FP) = €15 Investigation Cost
SELECT
    model_name,
    SUM(CASE WHEN actual_class = 1 AND predicted_class = 1 THEN 1 ELSE 0 END) AS true_positives,
    SUM(CASE WHEN actual_class = 0 AND predicted_class = 0 THEN 1 ELSE 0 END) AS true_negatives,
    SUM(CASE WHEN actual_class = 0 AND predicted_class = 1 THEN 1 ELSE 0 END) AS false_positives,
    SUM(CASE WHEN actual_class = 1 AND predicted_class = 0 THEN 1 ELSE 0 END) AS false_negatives,
    ROUND(SUM(CASE WHEN actual_class = 0 AND predicted_class = 1 THEN 15.0 ELSE 0 END), 2) AS total_false_alarm_cost_eur,
    ROUND(SUM(CASE WHEN actual_class = 1 AND predicted_class = 0 THEN t.amount ELSE 0 END), 2) AS total_missed_fraud_loss_eur
FROM model_predictions p
JOIN transactions t ON p.transaction_id = t.transaction_id
GROUP BY model_name;

-- 3. Top Suspicious PCA Anomaly Indicators in Critical Risk Bracket
SELECT 
    p.risk_tier,
    COUNT(*) AS count,
    ROUND(AVG(t.v14), 4) AS avg_v14_indicator,
    ROUND(AVG(t.v17), 4) AS avg_v17_indicator,
    ROUND(AVG(t.v12), 4) AS avg_v12_indicator,
    ROUND(AVG(t.v10), 4) AS avg_v10_indicator,
    ROUND(AVG(t.v4), 4) AS avg_v4_indicator
FROM model_predictions p
JOIN transactions t ON p.transaction_id = t.transaction_id
GROUP BY p.risk_tier;
