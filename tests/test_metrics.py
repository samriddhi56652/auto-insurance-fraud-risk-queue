"""Unit tests for the evaluation maths (no BigQuery needed)."""
import numpy as np

from src import modeling as M


def test_top_k_precision_recall_on_toy_queue():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1] * 10)       # 30 frauds in 100
    score = np.linspace(1, 0, 100)                          # queue order = row order
    m = M.top_k_metrics(y, score, ks=(0.10,))
    assert m["n_reviewed_top10"] == 10
    assert m["precision_top10"] == 0.3                      # 3 frauds in the first 10 rows
    assert m["recall_top10"] == 3 / 30


def test_wilson_interval_contains_point_estimate():
    lo, hi = M.wilson_ci(5, 100)
    assert lo < 0.05 < hi and 0 <= lo and hi <= 1


def test_platt_calibration_is_monotone():
    rng = np.random.default_rng(0)
    s = rng.random(2000)
    y = (rng.random(2000) < s ** 2).astype(int)
    p = M.PlattCalibrator().fit(s, y).predict(np.array([0.1, 0.5, 0.9]))
    assert p[0] < p[1] < p[2]
