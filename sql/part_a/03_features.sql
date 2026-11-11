-- Part A (REAL) — feature mart + time-based split.
-- Profiling showed a usable time order (Year 1994/1995/1996, fraud rate 6.66% -> 5.79% -> 5.22%),
-- so Part A uses the same design as Part B: train 1994, calibrate 1995, test 1996.
-- Each row is a different policy (PolicyNumber unique), so no per-policyholder history exists:
-- the only "history" fields are the insurer's own pre-binned PastNumberOfClaims etc.

CREATE OR REPLACE TABLE `{project}.fraud_marts.a_claim_features` AS
SELECT
  c.*,
  -- accident month -> claim month (wraps over year end; profiling shows lags of 0-11 months)
  MOD(claim_month - accident_month + 12, 12) AS months_accident_to_claim,
  CASE year WHEN 1994 THEN 'train' WHEN 1995 THEN 'calibrate' ELSE 'test' END AS split
FROM `{project}.fraud_clean.a_claims` c;

-- Checkpoint [1]: split sizes and label rates
SELECT split, COUNT(*) AS claims, SUM(fraud_label) AS n_fraud, ROUND(AVG(fraud_label), 4) AS fraud_rate
FROM `{project}.fraud_marts.a_claim_features` GROUP BY split ORDER BY MIN(year);

-- Checkpoint [2]: one row per clean claim
SELECT (SELECT COUNT(*) FROM `{project}.fraud_clean.a_claims`) AS clean_rows,
       COUNT(*) AS feature_rows, COUNT(DISTINCT claim_id) AS distinct_claims
FROM `{project}.fraud_marts.a_claim_features`;
