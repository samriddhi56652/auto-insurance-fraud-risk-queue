-- Part B (SYNTHETIC) — base rates for reading the FE/RE coefficients (percentage points per month).

-- [1] Monthly probability of any claim / a fraud-labelled claim, overall and by recent upgrade
SELECT
  IFNULL(DATE_DIFF(month, last_upgrade_month, MONTH) BETWEEN 0 AND 2, FALSE) AS recent_upgrade_2m,
  COUNT(*) AS policy_months,
  COUNT(DISTINCT policyholder_id) AS policyholders,
  ROUND(100 * AVG(IF(n_claims_incident > 0, 1, 0)), 3) AS pct_months_with_claim,
  ROUND(100 * AVG(IF(n_fraud_incident > 0, 1, 0)), 3) AS pct_months_with_fraud_claim
FROM `{project}.fraud_marts.b_panel_months`
GROUP BY ROLLUP (1) ORDER BY 1;

-- [2] How much within-policyholder variation is there in the regressors? (FE needs it)
SELECT
  COUNTIF(n_upgrades > 0) AS policyholders_with_an_upgrade,
  COUNTIF(n_tiers > 1) AS policyholders_who_changed_tier,
  COUNTIF(min_tenure <= 2) AS policyholders_starting_in_panel
FROM (
  SELECT policyholder_id, COUNTIF(tier_change > 0) AS n_upgrades,
         COUNT(DISTINCT tier_rank) AS n_tiers, MIN(tenure_months) AS min_tenure
  FROM `{project}.fraud_marts.b_panel_months` GROUP BY 1);
