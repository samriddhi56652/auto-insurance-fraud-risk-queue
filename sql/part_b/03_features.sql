-- Part B (SYNTHETIC) — feature marts. NO LOOK-AHEAD: every window frame ends at
-- "1 PRECEDING" or filters on report_date strictly before the current claim, so a
-- claim's features only use information available when it was reported.
-- tests/test_no_lookahead.py rebuilds this file on truncated data to prove it.

-- ---------------------------------------------------------------------------
-- 1) Policyholder-month panel (also used for the fixed/random-effects models)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE `{project}.fraud_marts.b_panel_months` AS
WITH claims_by_month AS (
  SELECT policyholder_id, DATE_TRUNC(report_date, MONTH) AS month,
         COUNT(*) AS n_claims_reported, SUM(fraud_label) AS n_fraud_labelled
  FROM `{project}.fraud_clean.b_claims` GROUP BY 1, 2
), incidents_by_month AS (
  SELECT policyholder_id, DATE_TRUNC(incident_date, MONTH) AS month,
         COUNT(*) AS n_claims_incident, SUM(fraud_label) AS n_fraud_incident
  FROM `{project}.fraud_clean.b_claims` GROUP BY 1, 2
), pm AS (
  SELECT m.*, p.policy_start_date, p.age_band, p.region, p.vehicle_segment,
         IFNULL(c.n_claims_reported, 0) AS n_claims_reported,
         IFNULL(i.n_claims_incident, 0) AS n_claims_incident,
         IFNULL(i.n_fraud_incident, 0) AS n_fraud_incident,
         m.tier_rank - LAG(m.tier_rank) OVER w AS tier_change
  FROM `{project}.fraud_clean.b_policy_months` m
  JOIN `{project}.fraud_clean.b_policyholders` p USING (policyholder_id)
  LEFT JOIN claims_by_month c USING (policyholder_id, month)
  LEFT JOIN incidents_by_month i USING (policyholder_id, month)
  WINDOW w AS (PARTITION BY m.policyholder_id ORDER BY m.month)
)
SELECT
  pm.*,
  DATE_DIFF(month, DATE_TRUNC(policy_start_date, MONTH), MONTH) AS tenure_months,
  -- rolling claim counts over the PREVIOUS 3/6/12 months (current month excluded)
  IFNULL(SUM(n_claims_reported) OVER (w ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING), 0)  AS claims_prev_3m,
  IFNULL(SUM(n_claims_reported) OVER (w ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING), 0)  AS claims_prev_6m,
  IFNULL(SUM(n_claims_reported) OVER (w ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING), 0) AS claims_prev_12m,
  -- last coverage change on or before this month (month start = effective date)
  MAX(IF(tier_change != 0, month, NULL)) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS last_change_month,
  MAX(IF(tier_change > 0, month, NULL)) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS last_upgrade_month,
  -- premium change vs 12 months earlier
  SAFE_DIVIDE(monthly_premium_inr, LAG(monthly_premium_inr, 12) OVER w) - 1 AS premium_change_12m
FROM pm
WINDOW w AS (PARTITION BY policyholder_id ORDER BY month);

-- ---------------------------------------------------------------------------
-- 2) Claim-level features (one row per clean claim)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE `{project}.fraud_marts.b_claim_features` AS
WITH c AS (
  SELECT c.*, p.contact_hash, p.policy_start_date, p.age_band, p.region,
         p.vehicle_segment, p.vehicle_age_years
  FROM `{project}.fraud_clean.b_claims` c
  JOIN `{project}.fraud_clean.b_policyholders` p USING (policyholder_id)
), hist AS (
  -- the policyholder's own claim history, ordered by when claims became known
  SELECT c.*,
    COUNT(*) OVER w_prev                         AS prior_claims_total,
    LAG(claim_amount_inr) OVER w                 AS prev_amount,
    LAG(claim_type) OVER w                       AS prev_type,
    LAG(repair_shop_id) OVER w                   AS prev_shop,
    LAG(report_date) OVER w                      AS prev_report_date,
    AVG(claim_amount_inr) OVER w_prev            AS hist_avg_amount,
    STDDEV_SAMP(claim_amount_inr) OVER w_prev    AS hist_std_amount
  FROM c
  WINDOW w AS (PARTITION BY policyholder_id ORDER BY report_date, claim_id),
         w_prev AS (w ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
), similar AS (
  -- earlier claims by the same policyholder: same type, amount within +-10%
  SELECT a.claim_id, COUNT(b.claim_id) AS prior_similar_claims
  FROM c a LEFT JOIN c b
    ON a.policyholder_id = b.policyholder_id AND b.claim_type = a.claim_type
   AND (b.report_date < a.report_date OR (b.report_date = a.report_date AND b.claim_id < a.claim_id))
   AND ABS(b.claim_amount_inr - a.claim_amount_inr) <= 0.10 * a.claim_amount_inr
  GROUP BY a.claim_id
), shop AS (
  -- claims at the same repair shop reported in the previous 90 / 365 days
  SELECT a.claim_id,
         COUNTIF(b.report_date >= DATE_SUB(a.report_date, INTERVAL 90 DAY)) AS shop_claims_prev_90d,
         COUNT(b.claim_id) AS shop_claims_prev_365d
  FROM c a LEFT JOIN c b
    ON a.repair_shop_id = b.repair_shop_id
   AND b.report_date < a.report_date AND b.report_date >= DATE_SUB(a.report_date, INTERVAL 365 DAY)
  GROUP BY a.claim_id
), contact_people AS (
  -- other policyholders at the same contact/address hash whose policy had started
  SELECT a.claim_id, COUNT(p.policyholder_id) AS shared_contact_policyholders
  FROM c a LEFT JOIN `{project}.fraud_clean.b_policyholders` p
    ON p.contact_hash = a.contact_hash AND p.policyholder_id != a.policyholder_id
   AND p.policy_start_date <= a.report_date
  GROUP BY a.claim_id
), contact AS (
  -- ... and their claims reported in the previous 180 days
  SELECT a.claim_id, ANY_VALUE(cp.shared_contact_policyholders) AS shared_contact_policyholders,
         COUNT(b.claim_id) AS shared_contact_claims_prev_180d
  FROM c a
  JOIN contact_people cp USING (claim_id)
  LEFT JOIN c b
    ON a.contact_hash = b.contact_hash AND b.policyholder_id != a.policyholder_id
   AND b.report_date < a.report_date AND b.report_date >= DATE_SUB(a.report_date, INTERVAL 180 DAY)
  GROUP BY a.claim_id
), third_party AS (
  -- is the named third party one of our customers, and did they claim recently?
  SELECT a.claim_id,
         LOGICAL_OR(p.policyholder_id IS NOT NULL) AS third_party_is_customer,
         COUNT(b.claim_id) AS third_party_claims_prev_180d
  FROM c a
  LEFT JOIN `{project}.fraud_clean.b_policyholders` p ON p.policyholder_id = a.third_party_ref
  LEFT JOIN c b ON b.policyholder_id = a.third_party_ref
   AND b.report_date < a.report_date AND b.report_date >= DATE_SUB(a.report_date, INTERVAL 180 DAY)
  GROUP BY a.claim_id
)
SELECT
  h.claim_id, h.policyholder_id, h.incident_date, h.report_date,
  CASE WHEN h.report_date <= DATE '2024-06-30' THEN 'train'
       WHEN h.report_date <= DATE '2024-12-31' THEN 'calibrate'
       ELSE 'test' END AS split,
  h.fraud_label,
  -- kept for fairness checks and the dashboard; NOT model features
  h.age_band, h.region,
  -- claim facts
  h.claim_type, h.claim_amount_inr,
  h.claim_amount_inr / 50000 AS amount_to_limit_ratio,
  h.claim_amount_inr >= 45000 AND h.claim_amount_inr < 50000 AS just_under_limit,
  DATE_DIFF(h.report_date, h.incident_date, DAY) AS report_delay_days,
  h.police_report_filed, h.witness_present,
  h.vehicle_segment, h.vehicle_age_years,
  pm.tier_rank, pm.monthly_premium_inr, pm.premium_change_12m,
  -- timing relative to the policy
  DATE_DIFF(h.incident_date, h.policy_start_date, DAY) AS days_since_policy_start,
  DATE_DIFF(h.incident_date, pm.last_change_month, DAY) AS days_since_coverage_change,
  DATE_DIFF(h.incident_date, pm.last_upgrade_month, DAY) AS days_since_upgrade,
  -- rolling frequency from the policyholder-month panel (previous months only)
  pm.claims_prev_3m, pm.claims_prev_6m, pm.claims_prev_12m,
  pm.claims_prev_3m - (pm.claims_prev_6m - pm.claims_prev_3m) AS claim_freq_change_3m,
  -- versus the policyholder's own history
  h.prior_claims_total,
  DATE_DIFF(h.report_date, h.prev_report_date, DAY) AS days_since_prev_claim,
  SAFE_DIVIDE(h.claim_amount_inr, h.hist_avg_amount) AS amount_vs_own_avg,
  SAFE_DIVIDE(h.claim_amount_inr - h.hist_avg_amount, NULLIF(h.hist_std_amount, 0)) AS amount_z_own_history,
  SAFE_DIVIDE(ABS(h.claim_amount_inr - h.prev_amount), h.prev_amount) AS amount_pct_diff_prev,
  h.claim_type = h.prev_type AS same_type_as_prev,
  h.repair_shop_id = h.prev_shop AS same_shop_as_prev,
  s.prior_similar_claims,
  -- network signals (staged-ring indicators)
  sh.shop_claims_prev_90d, sh.shop_claims_prev_365d,
  ct.shared_contact_policyholders, ct.shared_contact_claims_prev_180d,
  IFNULL(tp.third_party_is_customer, FALSE) AS third_party_is_customer,
  tp.third_party_claims_prev_180d
FROM hist h
JOIN `{project}.fraud_marts.b_panel_months` pm
  ON pm.policyholder_id = h.policyholder_id AND pm.month = DATE_TRUNC(h.incident_date, MONTH)
JOIN similar s USING (claim_id)
JOIN shop sh USING (claim_id)
JOIN contact ct USING (claim_id)
JOIN third_party tp USING (claim_id);

-- Checkpoint [1]: one feature row per clean claim, split sizes and label rates
SELECT split, COUNT(*) AS claims, SUM(fraud_label) AS fraud_labelled,
       ROUND(AVG(fraud_label), 4) AS label_rate, MIN(report_date) AS first_report, MAX(report_date) AS last_report
FROM `{project}.fraud_marts.b_claim_features` GROUP BY split ORDER BY first_report;

SELECT (SELECT COUNT(*) FROM `{project}.fraud_clean.b_claims`) AS clean_claims,
       (SELECT COUNT(*) FROM `{project}.fraud_marts.b_claim_features`) AS feature_rows,
       (SELECT COUNT(DISTINCT claim_id) FROM `{project}.fraud_marts.b_claim_features`) AS distinct_claims;
