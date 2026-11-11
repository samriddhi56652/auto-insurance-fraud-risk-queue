# Data

## Part A: REAL (`raw/fraud_oracle.csv`)

| | |
|---|---|
| Source | Kaggle, "Vehicle Insurance Claim Fraud Detection" by Shivam Bansal (`shivamb/vehicle-claim-fraud-detection`) |
| URL | https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection |
| Licence | **CC0: Public Domain** (per the Kaggle dataset metadata), so redistribution is allowed and the file is committed |
| Version | 1 ("Initial release", 2021-12-20) |
| File | `fraud_oracle.csv`, 3,618,564 bytes, starts with a UTF-8 byte-order mark |
| SHA-256 | `8b6aa59764ef4f8b058598d3e8f325ef3623f52f4b1cd4ef946baebb5ccfa9a6` |
| Shape | 15,420 rows × 33 columns; target `FraudFound_P` (923 = 1) |
| Period | `Year` 1994–1996 |

The Kaggle description does not name the original source of the data, so none is claimed here.

**To re-download instead of using the committed copy**, download the dataset zip from the URL above (or `kaggle datasets download shivamb/vehicle-claim-fraud-detection`), unzip `fraud_oracle.csv` into `data/raw/`, and check the SHA-256 above:

```bat
certutil -hashfile data\raw\fraud_oracle.csv SHA256
```

## Part B: SYNTHETIC (`synthetic/`)

The CSVs are **not committed**: run `python -m src.generate_panel`, which uses the fixed seed in `src/config.py`. The same seed gives byte-identical files (tested in `tests/test_generator.py`). The row counts and SHA-256 of each file are in `synthetic/generation_summary.json`, and every assumption is in `synthetic/ASSUMPTIONS.md`.

| File | Grain |
|---|---|
| `policyholders.csv` | 1 row per policyholder (age band, region, vehicle, contact hash, policy dates) |
| `policy_months.csv` | 1 row per policyholder × active month (coverage tier, premium) |
| `claims.csv` | 1 row per claim, **including planted data-quality problems** |
| `ground_truth_planted.csv` | 1 row per planted fraud claim (pattern, ring id). **Evaluation only, never loaded to BigQuery.** |

## `exports/`
These are the CSVs read by the Tableau Public dashboard, written by `python -m src.export`. Every row carries `dataset` and `is_synthetic` columns.
