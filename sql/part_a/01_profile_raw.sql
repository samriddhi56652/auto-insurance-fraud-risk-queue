-- Part A (REAL, Kaggle "Vehicle Insurance Claim Fraud Detection", CC0) — raw profile.
-- Every cleaning rule in 02_clean.sql is decided from these counts.

-- [1] Size, duplicates, is PolicyNumber a unique row id?
SELECT COUNT(*) AS n_rows,
       (SELECT COUNT(*) FROM (SELECT DISTINCT * FROM `{project}.fraud_raw.a_claims`)) AS distinct_rows,
       COUNT(DISTINCT PolicyNumber) AS distinct_policy_numbers,
       MIN(SAFE_CAST(PolicyNumber AS INT64)) AS min_policy_number,
       MAX(SAFE_CAST(PolicyNumber AS INT64)) AS max_policy_number,
       SUM(CAST(FraudFound_P AS INT64)) AS n_fraud,
       ROUND(AVG(CAST(FraudFound_P AS INT64)), 4) AS fraud_rate
FROM `{project}.fraud_raw.a_claims`;

-- [2] Missing / blank values and distinct counts per column
SELECT col, COUNTIF(val IS NULL OR TRIM(val) = '') AS n_missing, COUNT(DISTINCT val) AS n_distinct
FROM `{project}.fraud_raw.a_claims`
UNPIVOT INCLUDE NULLS (val FOR col IN (Month, WeekOfMonth, DayOfWeek, Make, AccidentArea, DayOfWeekClaimed,
  MonthClaimed, WeekOfMonthClaimed, Sex, MaritalStatus, Age, Fault, PolicyType, VehicleCategory, VehiclePrice,
  FraudFound_P, PolicyNumber, RepNumber, Deductible, DriverRating, Days_Policy_Accident, Days_Policy_Claim,
  PastNumberOfClaims, AgeOfVehicle, AgeOfPolicyHolder, PoliceReportFiled, WitnessPresent, AgentType,
  NumberOfSuppliments, AddressChange_Claim, NumberOfCars, Year, BasePolicy))
GROUP BY col ORDER BY col;

-- [3] Every value of every low-cardinality column, with its fraud rate
WITH u AS (
  SELECT col, val, CAST(FraudFound_P AS INT64) AS y
  FROM `{project}.fraud_raw.a_claims`
  UNPIVOT (val FOR col IN (Month, WeekOfMonth, DayOfWeek, Make, AccidentArea, DayOfWeekClaimed,
    MonthClaimed, WeekOfMonthClaimed, Sex, MaritalStatus, Fault, PolicyType, VehicleCategory, VehiclePrice,
    RepNumber, Deductible, DriverRating, Days_Policy_Accident, Days_Policy_Claim, PastNumberOfClaims,
    AgeOfVehicle, AgeOfPolicyHolder, PoliceReportFiled, WitnessPresent, AgentType, NumberOfSuppliments,
    AddressChange_Claim, NumberOfCars, Year, BasePolicy))
)
SELECT col, val, COUNT(*) AS n, SUM(y) AS n_fraud, ROUND(AVG(y), 4) AS fraud_rate
FROM u GROUP BY col, val ORDER BY col, val;

-- [4] Impossible / suspicious values
SELECT
  COUNTIF(SAFE_CAST(Age AS INT64) IS NULL)  AS age_not_numeric,
  COUNTIF(SAFE_CAST(Age AS INT64) = 0)      AS age_zero,
  COUNTIF(SAFE_CAST(Age AS INT64) BETWEEN 1 AND 15) AS age_1_to_15,
  MIN(NULLIF(SAFE_CAST(Age AS INT64), 0))   AS min_nonzero_age,
  MAX(SAFE_CAST(Age AS INT64))              AS max_age,
  COUNTIF(DayOfWeekClaimed = '0')           AS day_claimed_zero,
  COUNTIF(MonthClaimed = '0')               AS month_claimed_zero
FROM `{project}.fraud_raw.a_claims`;

-- [5] Where do the Age = 0 rows come from? (cross-check with the age band)
SELECT AgeOfPolicyHolder, COUNT(*) AS n, SUM(CAST(FraudFound_P AS INT64)) AS n_fraud
FROM `{project}.fraud_raw.a_claims`
WHERE SAFE_CAST(Age AS INT64) = 0
GROUP BY 1 ORDER BY 1;

-- [6] Rows with a '0' claimed day/month: what else is odd about them?
SELECT * FROM `{project}.fraud_raw.a_claims` WHERE DayOfWeekClaimed = '0' OR MonthClaimed = '0';

-- [7] Time order: rows and fraud rate by Year (decides time split vs stratified CV)
SELECT Year, COUNT(*) AS n, SUM(CAST(FraudFound_P AS INT64)) AS n_fraud,
       ROUND(AVG(CAST(FraudFound_P AS INT64)), 4) AS fraud_rate
FROM `{project}.fraud_raw.a_claims` GROUP BY Year ORDER BY Year;

-- [8] Is the file sorted by time? Year range over PolicyNumber deciles
SELECT decile, MIN(Year) AS min_year, MAX(Year) AS max_year, COUNT(*) AS n,
       ROUND(AVG(CAST(FraudFound_P AS INT64)), 4) AS fraud_rate
FROM (SELECT *, NTILE(10) OVER (ORDER BY SAFE_CAST(PolicyNumber AS INT64)) AS decile
      FROM `{project}.fraud_raw.a_claims`)
GROUP BY decile ORDER BY decile;

-- [9] Accident month vs claim month: implied lag, to spot impossible orderings
WITH m AS (
  SELECT *,
    DIV(STRPOS('JanFebMarAprMayJunJulAugSepOctNovDec', Month), 3) + 1 AS acc_m,
    DIV(STRPOS('JanFebMarAprMayJunJulAugSepOctNovDec', MonthClaimed), 3) + 1 AS clm_m
  FROM `{project}.fraud_raw.a_claims` WHERE MonthClaimed != '0'
)
SELECT MOD(clm_m - acc_m + 12, 12) AS months_accident_to_claim, COUNT(*) AS n,
       ROUND(AVG(CAST(FraudFound_P AS INT64)), 4) AS fraud_rate
FROM m GROUP BY 1 ORDER BY 1;
