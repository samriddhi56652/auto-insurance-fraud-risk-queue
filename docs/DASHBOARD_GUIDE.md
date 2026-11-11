# Tableau Public dashboard: build guide

Tableau Public can't connect to BigQuery, so the dashboard reads the CSVs in `data/exports/` (written by `python -m src.export`). Expect about 30–45 minutes in Tableau Public Desktop.

## 0. Connect
**Connect → Text file**, then add each CSV as its own data source:
`risk_queue.csv`, `kpis.csv`, `global_drivers.csv`, `cost_curve_by_queue_depth.csv`, `fairness_flag_rates.csv`, `pattern_recovery_synthetic.csv`.

Create one **parameter**, `Dataset` (string, list):
- `A - REAL Kaggle claims (test year 1996)`
- `B - SYNTHETIC panel (test year 2025)`

On every sheet, add a filter with the calculated field `[dataset] = [Dataset]`.

Create a calculated field `Data label`:
```
IF [is_synthetic] THEN "SYNTHETIC DATA: planted patterns, not real fraud rates"
ELSE "REAL (Kaggle, 1994-96, anonymised)" END
```
Put it in **every sheet title**. This is a hard rule from the README.

## Dashboard 1: SIU risk queue
**Parameter `Capacity %`:** float, range 0.01–0.10, step 0.01, default 0.05.

**Sheet "Queue"** (source `risk_queue.csv`)
- **Filter:** `[queue_pct] <= [Capacity %]`
- **Rows:** `rank`, `claim_id`, `segment`, `claim_amount_inr`, `top_reason_1`, `top_reason_2`, `top_reason_3`
- **Text:** `calibrated_p`, formatted as a %
- **Colour:** `calibrated_p` (sequential)
- **Tooltip:** "Score = reason to investigate, not proof of fraud."

**Sheet "KPI tiles"** (source `kpis.csv`), one tile each:
- `precision_top5`
- `recall_top5`
- `lift_top5`
- `pr_auc`, with `pr_auc_ci_low`–`pr_auc_ci_high` in the tooltip
- `base_rate`

**Sheet "Hit rate at capacity"** (source `risk_queue.csv`)
- `Hit rate = SUM([historical_fraud_label]) / COUNT([claim_id])`, filtered by `[queue_pct] <= [Capacity %]`
- This should reproduce `precision_top5` exactly when Capacity = 5%. Use it as the checkpoint below.

## Dashboard 2: red-flag drivers
**Sheet "Global drivers"** (source `global_drivers.csv`): horizontal bar of `mean_abs_shap` by `feature`, top 12 (Top N filter), sorted.

**Sheet "Most common reasons"** (source `risk_queue.csv`, flagged claims only, `flag_top5 = 1`)
- Pivot `top_reason_1..3` (Data Source tab → select the 3 columns → Pivot).
- Strip the numbers so reasons group together. Calculated field `Reason`:
  `REGEXP_REPLACE([Pivot Field Values], "[-+0-9.,%]+", "#")`
- Bar chart: count of claims by `Reason`.

**Part B only:** sheet "Planted pattern recovery" (source `pattern_recovery_synthetic.csv`), bars of `recovered_top5` by `pattern`. Title it **SYNTHETIC**.

## Dashboard 3: cost-based threshold
**Parameters:**
- `Investigation cost (Rs)`: 2,000–10,000, default 5,000
- `Recovery rate`: 0.4–0.8, default 0.6

**Calculated field** (source `cost_curve_by_queue_depth.csv`):
```
Net benefit = [fraud_amount_caught_inr] * [Recovery rate] - [Investigation cost (Rs)] * [n_reviewed]
```

**Sheet "Net benefit vs queue depth"**
- **Line:** `pct_reviewed` (x) against `Net benefit` (y)
- **Reference line:** `Capacity %`
- **Caption:** "Costs are stated assumptions (src/config.py). Part A uses an assumed Rs 60,000 average fraud loss."

## Dashboard 4 (optional): fairness
**Sheet "Flag rate by group"** (source `fairness_flag_rates.csv`)
- **Filter:** `operating_point = flag_top5`
- **Bars:** `flag_rate` by `group`, with error bars from `flag_rate_ci_low` to `flag_rate_ci_high`
- **Second mark:** `observed_fraud_rate`
- **Annotation (Part A):** call out the "16 to 17" band. See the README, "Explanations and fairness".

## Checkpoint before publishing
These three numbers on the dashboard must equal the CSVs exactly:
1. KPI tile `precision_top5` for each dataset equals `kpis.csv`.
2. "Hit rate at capacity" at 5% equals the same value.
3. Net benefit at 5,000 / 0.6, at the queue depth closest to the cost rule, matches `cost_curve_by_queue_depth.csv`.

Publish with **File → Save to Tableau Public**, then paste the link into the README's "Dashboard" section.
