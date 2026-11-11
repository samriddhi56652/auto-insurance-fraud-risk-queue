-- Part B (SYNTHETIC) — data-quality profile of the raw tables.
-- Every cleaning rule in 02_clean.sql is decided from these counts.

-- [1] Row counts
SELECT 'b_policyholders' AS tbl, COUNT(*) AS n_rows FROM `{project}.fraud_raw.b_policyholders`
UNION ALL SELECT 'b_policy_months', COUNT(*) FROM `{project}.fraud_raw.b_policy_months`
UNION ALL SELECT 'b_claims', COUNT(*) FROM `{project}.fraud_raw.b_claims`;

-- [2] Missing / blank values per claims column
SELECT
  COUNTIF(claim_id IS NULL OR TRIM(claim_id) = '')                  AS claim_id_missing,
  COUNTIF(policyholder_id IS NULL OR TRIM(policyholder_id) = '')    AS policyholder_id_missing,
  COUNTIF(incident_date IS NULL OR TRIM(incident_date) = '')        AS incident_date_missing,
  COUNTIF(report_date IS NULL OR TRIM(report_date) = '')            AS report_date_missing,
  COUNTIF(claim_amount_inr IS NULL OR TRIM(claim_amount_inr) = '')  AS claim_amount_missing,
  COUNTIF(repair_shop_id IS NULL OR TRIM(repair_shop_id) = '')      AS repair_shop_missing,
  COUNTIF(third_party_ref IS NULL OR TRIM(third_party_ref) = '')    AS third_party_missing,
  COUNTIF(fraud_label IS NULL OR fraud_label NOT IN ('0', '1'))     AS label_missing_or_invalid
FROM `{project}.fraud_raw.b_claims`;

-- [3] Blank shop / third party by claim type (structural, or a data problem?)
SELECT claim_type,
       COUNT(*) AS n,
       COUNTIF(repair_shop_id IS NULL OR repair_shop_id = '') AS no_shop,
       COUNTIF(third_party_ref IS NULL OR third_party_ref = '') AS no_third_party
FROM `{project}.fraud_raw.b_claims`
GROUP BY claim_type ORDER BY claim_type;

-- [4] Duplicates: exact duplicate rows vs duplicate claim_ids
SELECT
  (SELECT COUNT(*) FROM `{project}.fraud_raw.b_claims`) AS total_rows,
  (SELECT COUNT(*) FROM (SELECT DISTINCT * FROM `{project}.fraud_raw.b_claims`)) AS distinct_rows,
  (SELECT COUNT(DISTINCT claim_id) FROM `{project}.fraud_raw.b_claims`) AS distinct_claim_ids;

-- [5] Impossible or out-of-scope values
WITH c AS (
  SELECT c.*, SAFE_CAST(claim_amount_inr AS NUMERIC) AS amt,
         SAFE.PARSE_DATE('%Y-%m-%d', incident_date) AS inc,
         SAFE.PARSE_DATE('%Y-%m-%d', report_date) AS rep,
         SAFE.PARSE_DATE('%Y-%m-%d', p.policy_start_date) AS pstart,
         SAFE.PARSE_DATE('%Y-%m-%d', NULLIF(p.policy_end_date, '')) AS pend
  FROM (SELECT DISTINCT * FROM `{project}.fraud_raw.b_claims`) c
  LEFT JOIN `{project}.fraud_raw.b_policyholders` p USING (policyholder_id)
)
SELECT
  COUNT(*)                                             AS distinct_rows,
  COUNTIF(pstart IS NULL)                              AS unknown_policyholder,
  COUNTIF(amt IS NULL AND claim_amount_inr != '')      AS amount_unparseable,
  COUNTIF(amt <= 0)                                    AS amount_non_positive,
  COUNTIF(inc IS NULL OR rep IS NULL)                  AS date_unparseable,
  COUNTIF(rep < inc)                                   AS report_before_incident,
  COUNTIF(inc < pstart)                                AS incident_before_policy_start,
  COUNTIF(inc > pend)                                  AS incident_after_policy_end,
  COUNTIF(inc < DATE '2023-01-01')                     AS incident_before_panel,
  MIN(amt) AS min_amount, MAX(amt) AS max_amount,
  APPROX_QUANTILES(amt, 100)[OFFSET(50)] AS median_amount,
  APPROX_QUANTILES(DATE_DIFF(rep, inc, DAY), 100)[OFFSET(99)] AS p99_report_delay_days
FROM c;

-- [6] Region spelling variants (categorical consistency)
SELECT region AS raw_region, COUNT(*) AS n
FROM `{project}.fraud_raw.b_policyholders`
GROUP BY region ORDER BY region;

-- [7] Other categoricals
SELECT 'age_band' AS col, age_band AS value, COUNT(*) AS n FROM `{project}.fraud_raw.b_policyholders` GROUP BY 1, 2
UNION ALL SELECT 'vehicle_segment', vehicle_segment, COUNT(*) FROM `{project}.fraud_raw.b_policyholders` GROUP BY 1, 2
UNION ALL SELECT 'claim_type', claim_type, COUNT(*) FROM `{project}.fraud_raw.b_claims` GROUP BY 1, 2
UNION ALL SELECT 'coverage_tier', coverage_tier, COUNT(*) FROM `{project}.fraud_raw.b_policy_months` GROUP BY 1, 2
ORDER BY col, value;

-- [8] Panel structure: one row per policyholder-month, no gaps inside a policy
WITH pm AS (
  SELECT policyholder_id, PARSE_DATE('%Y-%m-%d', month) AS month
  FROM `{project}.fraud_raw.b_policy_months`
), g AS (
  SELECT policyholder_id, COUNT(*) AS n_rows, COUNT(DISTINCT month) AS n_months,
         DATE_DIFF(MAX(month), MIN(month), MONTH) + 1 AS span_months
  FROM pm GROUP BY policyholder_id
)
SELECT COUNT(*) AS policyholders_in_panel,
       COUNTIF(n_rows != n_months) AS with_duplicate_months,
       COUNTIF(n_months != span_months) AS with_gaps,
       MIN(n_months) AS min_months, MAX(n_months) AS max_months,
       ROUND(AVG(n_months), 1) AS avg_months
FROM g;

-- [9] Label rate and volume by report month (decides the time split)
SELECT FORMAT_DATE('%Y-%m', SAFE.PARSE_DATE('%Y-%m-%d', report_date)) AS report_month,
       COUNT(*) AS claims, SUM(CAST(fraud_label AS INT64)) AS fraud_labelled,
       ROUND(AVG(CAST(fraud_label AS INT64)), 4) AS label_rate
FROM (SELECT DISTINCT * FROM `{project}.fraud_raw.b_claims`)
GROUP BY report_month ORDER BY report_month;

-- [10] Overall label rate
SELECT COUNT(*) AS claims, SUM(CAST(fraud_label AS INT64)) AS fraud_labelled,
       ROUND(AVG(CAST(fraud_label AS INT64)), 4) AS label_rate
FROM (SELECT DISTINCT * FROM `{project}.fraud_raw.b_claims`);

-- [11] Claim types not covered by the policy tier in the incident month
--      (Basic = third-party liability only). Should be 0 in a consistent book.
SELECT pm.coverage_tier, c.claim_type, COUNT(*) AS n
FROM (SELECT DISTINCT * FROM `{project}.fraud_raw.b_claims`) c
JOIN `{project}.fraud_raw.b_policy_months` pm
  ON pm.policyholder_id = c.policyholder_id
 AND pm.month = FORMAT_DATE('%Y-%m-01', SAFE.PARSE_DATE('%Y-%m-%d', c.incident_date))
WHERE pm.coverage_tier = 'Basic' AND c.claim_type != 'ThirdParty'
GROUP BY 1, 2 ORDER BY 1, 2;
