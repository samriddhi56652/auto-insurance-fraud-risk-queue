"""Project-wide settings. Every assumed number used in the analysis lives here,
so README and dashboard figures can point to one place."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

GCP_PROJECT = os.environ.get("GCP_PROJECT", "")
BQ_LOCATION = os.environ.get("BQ_LOCATION", "asia-south1")
BQ_MAX_BYTES_BILLED = int(os.environ.get("BQ_MAX_BYTES_BILLED", 1_000_000_000))
DS_RAW, DS_CLEAN, DS_MARTS = "fraud_raw", "fraud_clean", "fraud_marts"

DATA_RAW = ROOT / "data" / "raw"
DATA_SYN = ROOT / "data" / "synthetic"
DATA_EXPORTS = ROOT / "data" / "exports"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
PROFILING = REPORTS / "profiling"
MODELS = ROOT / "models"

SEED = 20261111

# --- Part A source -----------------------------------------------------------
PART_A_FILE = DATA_RAW / "fraud_oracle.csv"
PART_A_URL = "https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection"

# --- Part B synthetic panel ----------------------------------------------------
PANEL_START = "2023-01-01"
PANEL_MONTHS = 36                      # Jan 2023 .. Dec 2025
APPROVAL_LIMIT_INR = 50_000            # claims below this are fast-tracked without senior review
# time-based split for Part B (by claim report date)
B_TRAIN_END = "2024-06-30"             # train:      2023-01 .. 2024-06
B_CALIB_END = "2024-12-31"             # calibrate:  2024-07 .. 2024-12 (calibration + threshold choice)
                                       # test:       2025-01 .. 2025-12

# --- SIU cost assumptions (STATED ASSUMPTIONS, not real figures) --------------
INVESTIGATION_COST_INR = 5_000         # analyst + field time per reviewed claim
RECOVERY_RATE = 0.60                   # share of a fraudulent payout prevented when investigated
PART_A_AVG_FRAUD_LOSS_INR = 60_000     # Part A has no claim amount, so one average loss is assumed
TOP_K = (0.01, 0.05, 0.10)             # SIU capacity scenarios: review top 1% / 5% / 10%

# Ranges for the sensitivity table
SENS_INVESTIGATION_COST = (2_000, 5_000, 10_000)
SENS_RECOVERY_RATE = (0.4, 0.6, 0.8)
