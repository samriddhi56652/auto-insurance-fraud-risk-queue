-- Part B (SYNTHETIC) — raw -> clean. Each rule is justified by a count in
-- reports/profiling/part_b/01_profile_raw_*.csv (see README "Cleaning rules").
--   R1 exact duplicate claim rows (57)          -> keep one copy
--   R2 claim amount blank (26)                  -> exclude: amount drives the near-limit and history features
--   R3 report_date before incident_date (18)    -> treat as transposed fields: swap, flag dq_dates_swapped
--   R4 incident before policy start (13) or only
--      before the panel window (3 more)         -> exclude: no cover / no history to build features from
--   R5 region spelling variants (397 rows)      -> TRIM + INITCAP
--   Blank repair shop (Theft) and blank third party (Glass/Natural/Theft) are structural -> keep NULL

CREATE OR REPLACE TABLE `{project}.fraud_clean.b_policyholders` AS
SELECT
  policyholder_id,
  age_band,
  INITCAP(TRIM(region)) AS region,
  vehicle_segment,
  CAST(vehicle_age_years AS INT64) AS vehicle_age_years,
  contact_hash,
  PARSE_DATE('%Y-%m-%d', policy_start_date) AS policy_start_date,
  SAFE.PARSE_DATE('%Y-%m-%d', NULLIF(policy_end_date, '')) AS policy_end_date
FROM `{project}.fraud_raw.b_policyholders`;

CREATE OR REPLACE TABLE `{project}.fraud_clean.b_policy_months` AS
SELECT
  policyholder_id,
  PARSE_DATE('%Y-%m-%d', month) AS month,
  coverage_tier,
  CASE coverage_tier WHEN 'Basic' THEN 0 WHEN 'Standard' THEN 1 WHEN 'Comprehensive' THEN 2 END AS tier_rank,
  CAST(monthly_premium_inr AS NUMERIC) AS monthly_premium_inr
FROM `{project}.fraud_raw.b_policy_months`;

CREATE OR REPLACE TABLE `{project}.fraud_clean.b_claims_staged` AS
WITH d AS (SELECT DISTINCT * FROM `{project}.fraud_raw.b_claims`),          -- R1
t AS (
  SELECT d.*,
         PARSE_DATE('%Y-%m-%d', incident_date) AS inc_raw,
         PARSE_DATE('%Y-%m-%d', report_date) AS rep_raw,
         SAFE_CAST(NULLIF(claim_amount_inr, '') AS NUMERIC) AS amt
  FROM d
)
SELECT
  t.claim_id, t.policyholder_id,
  LEAST(inc_raw, rep_raw)    AS incident_date,                                 -- R3
  GREATEST(inc_raw, rep_raw) AS report_date,
  rep_raw < inc_raw          AS dq_dates_swapped,
  t.claim_type,
  amt                        AS claim_amount_inr,
  NULLIF(t.repair_shop_id, '')  AS repair_shop_id,
  NULLIF(t.third_party_ref, '') AS third_party_ref,
  t.police_report_filed = 'Yes' AS police_report_filed,
  t.witness_present = 'Yes'     AS witness_present,
  CAST(t.fraud_label AS INT64)  AS fraud_label,
  CASE
    WHEN amt IS NULL THEN 'R2_missing_amount'
    WHEN LEAST(inc_raw, rep_raw) < p.policy_start_date THEN 'R4_before_policy_start'
    WHEN LEAST(inc_raw, rep_raw) < DATE '2023-01-01' THEN 'R4_before_panel_window'
  END AS exclusion_reason
FROM t
JOIN `{project}.fraud_clean.b_policyholders` p USING (policyholder_id);

CREATE OR REPLACE TABLE `{project}.fraud_clean.b_claims_excluded` AS
SELECT claim_id, policyholder_id, exclusion_reason, fraud_label
FROM `{project}.fraud_clean.b_claims_staged`
WHERE exclusion_reason IS NOT NULL;

CREATE OR REPLACE TABLE `{project}.fraud_clean.b_claims` AS
SELECT * EXCEPT (exclusion_reason)
FROM `{project}.fraud_clean.b_claims_staged`
WHERE exclusion_reason IS NULL;

DROP TABLE `{project}.fraud_clean.b_claims_staged`;

-- Checkpoint [1]: rows accounted for (raw distinct = clean + excluded)
SELECT
  (SELECT COUNT(*) FROM (SELECT DISTINCT * FROM `{project}.fraud_raw.b_claims`)) AS raw_distinct,
  (SELECT COUNT(*) FROM `{project}.fraud_clean.b_claims`) AS clean_rows,
  (SELECT COUNT(*) FROM `{project}.fraud_clean.b_claims_excluded`) AS excluded_rows,
  (SELECT COUNTIF(dq_dates_swapped) FROM `{project}.fraud_clean.b_claims`) AS dates_swapped;

-- Checkpoint [2]: exclusions by reason
SELECT exclusion_reason, COUNT(*) AS n, SUM(fraud_label) AS fraud_labelled
FROM `{project}.fraud_clean.b_claims_excluded` GROUP BY 1 ORDER BY 1;

-- Checkpoint [3]: no impossible values left (every column should be 0)
SELECT
  COUNTIF(c.report_date < c.incident_date) AS report_before_incident,
  COUNTIF(c.incident_date < p.policy_start_date) AS before_policy_start,
  COUNTIF(c.claim_amount_inr IS NULL OR c.claim_amount_inr <= 0) AS bad_amount,
  COUNT(*) - COUNT(DISTINCT c.claim_id) AS duplicate_claim_ids,
  (SELECT COUNT(DISTINCT region) FROM `{project}.fraud_clean.b_policyholders`) - 5 AS extra_regions
FROM `{project}.fraud_clean.b_claims` c
JOIN `{project}.fraud_clean.b_policyholders` p USING (policyholder_id);

-- Checkpoint [4]: swapped dates now have plausible reporting delays
SELECT dq_dates_swapped, COUNT(*) AS n,
       APPROX_QUANTILES(DATE_DIFF(report_date, incident_date, DAY), 4) AS delay_quartiles
FROM `{project}.fraud_clean.b_claims` GROUP BY 1;
