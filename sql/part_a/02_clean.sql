-- Part A (REAL) — raw -> clean. Rules decided from reports/profiling/part_a/01_profile_raw_*.csv
--   No exact duplicates (15,420 distinct rows) and no blank values in any column -> nothing to drop for those.
--   R1 Age = '0' (320 rows), ALL in AgeOfPolicyHolder '16 to 17'  -> placeholder, not an age: set NULL, keep the band
--   R2 DayOfWeekClaimed = '0' AND MonthClaimed = '0' (1 row)       -> claim date unknown: exclude
--   R3 banded text columns ('more than 30', '2 to 4' ...)          -> ordered integer codes (order kept, no invented midpoints)
--   R4 'none' in Days_Policy_Accident (55 rows)                     -> ambiguous, kept as its own code (0) plus a flag
--   PolicyNumber = unique row id 1..15420, sorted by Year           -> kept as an id only, never a feature

CREATE OR REPLACE TABLE `{project}.fraud_clean.a_claims` AS
WITH r AS (SELECT * FROM `{project}.fraud_raw.a_claims` WHERE NOT (DayOfWeekClaimed = '0' AND MonthClaimed = '0'))
SELECT
  CAST(PolicyNumber AS INT64) AS claim_id,
  CAST(Year AS INT64) AS year,
  DIV(STRPOS('JanFebMarAprMayJunJulAugSepOctNovDec', Month), 3) + 1 AS accident_month,
  DIV(STRPOS('JanFebMarAprMayJunJulAugSepOctNovDec', MonthClaimed), 3) + 1 AS claim_month,
  CAST(WeekOfMonth AS INT64) AS week_of_month,
  CAST(WeekOfMonthClaimed AS INT64) AS week_of_month_claimed,
  DayOfWeek AS day_of_week, DayOfWeekClaimed AS day_of_week_claimed,
  Make AS make,
  AccidentArea = 'Urban' AS accident_urban,
  Sex AS sex, MaritalStatus AS marital_status,
  NULLIF(CAST(Age AS INT64), 0) AS age,                                            -- R1
  AgeOfPolicyHolder AS age_band,
  CASE AgeOfPolicyHolder WHEN '16 to 17' THEN 0 WHEN '18 to 20' THEN 1 WHEN '21 to 25' THEN 2
    WHEN '26 to 30' THEN 3 WHEN '31 to 35' THEN 4 WHEN '36 to 40' THEN 5 WHEN '41 to 50' THEN 6
    WHEN '51 to 65' THEN 7 WHEN 'over 65' THEN 8 END AS age_band_ord,
  Fault = 'Policy Holder' AS fault_policyholder,
  PolicyType AS policy_type, VehicleCategory AS vehicle_category, BasePolicy AS base_policy,
  CASE VehiclePrice WHEN 'less than 20000' THEN 0 WHEN '20000 to 29000' THEN 1 WHEN '30000 to 39000' THEN 2
    WHEN '40000 to 59000' THEN 3 WHEN '60000 to 69000' THEN 4 WHEN 'more than 69000' THEN 5 END AS vehicle_price_ord,
  CAST(RepNumber AS INT64) AS rep_number,
  CAST(Deductible AS INT64) AS deductible,
  CAST(DriverRating AS INT64) AS driver_rating,
  CASE Days_Policy_Accident WHEN 'none' THEN 0 WHEN '1 to 7' THEN 1 WHEN '8 to 15' THEN 2
    WHEN '15 to 30' THEN 3 WHEN 'more than 30' THEN 4 END AS days_policy_accident_ord,  -- R4
  Days_Policy_Accident = 'none' AS days_policy_accident_none,
  CASE Days_Policy_Claim WHEN 'none' THEN 0 WHEN '8 to 15' THEN 2
    WHEN '15 to 30' THEN 3 WHEN 'more than 30' THEN 4 END AS days_policy_claim_ord,
  CASE PastNumberOfClaims WHEN 'none' THEN 0 WHEN '1' THEN 1 WHEN '2 to 4' THEN 2 WHEN 'more than 4' THEN 3 END AS past_claims_ord,
  CASE AgeOfVehicle WHEN 'new' THEN 0 WHEN '2 years' THEN 2 WHEN '3 years' THEN 3 WHEN '4 years' THEN 4
    WHEN '5 years' THEN 5 WHEN '6 years' THEN 6 WHEN '7 years' THEN 7 WHEN 'more than 7' THEN 8 END AS vehicle_age_ord,
  PoliceReportFiled = 'Yes' AS police_report_filed,
  WitnessPresent = 'Yes' AS witness_present,
  AgentType = 'External' AS agent_external,
  CASE NumberOfSuppliments WHEN 'none' THEN 0 WHEN '1 to 2' THEN 1 WHEN '3 to 5' THEN 2 WHEN 'more than 5' THEN 3 END AS supplements_ord,
  AddressChange_Claim AS address_change,
  CASE NumberOfCars WHEN '1 vehicle' THEN 1 WHEN '2 vehicles' THEN 2 WHEN '3 to 4' THEN 3
    WHEN '5 to 8' THEN 4 WHEN 'more than 8' THEN 5 END AS number_of_cars_ord,
  CAST(FraudFound_P AS INT64) AS fraud_label
FROM r;

-- Checkpoint [1]: rows accounted for, and no unmapped band (every *_ord must be non-NULL)
SELECT COUNT(*) AS clean_rows, 15420 - COUNT(*) AS excluded_rows, SUM(fraud_label) AS n_fraud,
  COUNTIF(age IS NULL) AS age_set_null,
  COUNTIF(age_band_ord IS NULL OR vehicle_price_ord IS NULL OR days_policy_accident_ord IS NULL
       OR days_policy_claim_ord IS NULL OR past_claims_ord IS NULL OR vehicle_age_ord IS NULL
       OR supplements_ord IS NULL OR number_of_cars_ord IS NULL
       OR accident_month NOT BETWEEN 1 AND 12 OR claim_month NOT BETWEEN 1 AND 12) AS unmapped_values
FROM `{project}.fraud_clean.a_claims`;
