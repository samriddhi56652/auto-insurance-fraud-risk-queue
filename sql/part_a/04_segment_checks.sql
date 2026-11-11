-- Part A (REAL) — checks behind two modelling decisions (see README "Decisions").

-- [1] Fraud rate by accident month and year: whole blocks of months are ~0% and the
--     block moves every year -> labelling artefact, so accident month is NOT a feature.
SELECT year, accident_month, COUNT(*) AS n, SUM(fraud_label) AS n_fraud, ROUND(AVG(fraud_label), 3) AS fraud_rate
FROM `{project}.fraud_clean.a_claims` GROUP BY 1, 2 ORDER BY 1, 2;

-- [2] The '16 to 17' age band is one distinctive cluster: profile vs everyone else
SELECT age_band = '16 to 17' AS band_16_17, COUNT(*) AS n,
  ROUND(AVG(IF(vehicle_price_ord = 5, 1, 0)), 3) AS share_price_over_69k,
  ROUND(AVG(IF(vehicle_age_ord = 0, 1, 0)), 3) AS share_new_vehicle,
  ROUND(AVG(IF(supplements_ord = 0, 1, 0)), 3) AS share_no_supplements,
  ROUND(AVG(IF(marital_status = 'Single', 1, 0)), 3) AS share_single,
  ROUND(AVG(IF(fault_policyholder, 1, 0)), 3) AS share_policyholder_fault
FROM `{project}.fraud_clean.a_claims` GROUP BY 1 ORDER BY 1;

-- [3] ... and its fraud rate collapsed after 1994
SELECT year, age_band = '16 to 17' AS band_16_17, COUNT(*) AS n, SUM(fraud_label) AS n_fraud,
       ROUND(AVG(fraud_label), 3) AS fraud_rate
FROM `{project}.fraud_clean.a_claims` GROUP BY 1, 2 ORDER BY 1, 2;
