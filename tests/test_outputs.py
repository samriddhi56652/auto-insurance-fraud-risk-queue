"""Checkpoint tests on the pipeline outputs (run after python -m src.export)."""
import json

import pandas as pd
import pytest

from src import bq, config

EXP = config.DATA_EXPORTS


@pytest.fixture(scope="module")
def queue():
    return pd.read_csv(EXP / "risk_queue.csv")


def test_ranks_unique_and_contiguous_per_dataset(queue):
    for ds, d in queue.groupby("dataset"):
        assert sorted(d["rank"]) == list(range(1, len(d) + 1)), ds


def test_synthetic_flag_matches_dataset(queue):
    assert (queue["is_synthetic"] == queue["dataset"].str.contains("SYNTHETIC")).all()
    assert queue.loc[~queue["is_synthetic"], "planted_pattern_synthetic"].isna().all()


def test_demographics_are_not_model_features():
    for part in ("part_a", "part_b"):
        feats = set(pd.read_csv(config.REPORTS / part / "global_drivers.csv")["feature"])
        assert not feats & {"age", "age_band", "age_band_ord", "sex", "marital_status", "region"}, part


def test_ids_and_split_variables_are_not_features():
    for part in ("part_a", "part_b"):
        feats = set(pd.read_csv(config.REPORTS / part / "global_drivers.csv")["feature"])
        assert not feats & {"claim_id", "policyholder_id", "rep_number", "year", "split", "fraud_label",
                            "pattern", "ring_id", "report_date", "incident_date"}, part


def test_kpis_match_headline_json():
    k = pd.read_csv(EXP / "kpis.csv").set_index("is_synthetic")
    for part, synth in (("part_a", False), ("part_b", True)):
        h = json.loads((config.REPORTS / part / "headline_metrics.json").read_text())
        assert abs(k.loc[synth, "pr_auc"] - h["pr_auc"]) < 1e-12
        assert abs(k.loc[synth, "precision_top5"] - h["precision_top5"]) < 1e-12


def test_queue_metrics_recomputed_from_queue_file(queue):
    """Top-5% precision recomputed independently from the exported queue."""
    k = pd.read_csv(EXP / "kpis.csv").set_index("dataset")
    for ds, d in queue.groupby("dataset"):
        n = int(round(0.05 * len(d)))
        prec = d.nsmallest(n, "rank")["historical_fraud_label"].mean()
        assert abs(prec - k.loc[ds, "precision_top5"]) < 1e-12, ds


@pytest.mark.bq
def test_bigquery_mart_matches_csv(queue):
    r = bq.query("SELECT dataset, COUNT(*) AS n, MIN(`rank`) AS lo, MAX(`rank`) AS hi "
                 "FROM `{project}.fraud_marts.risk_queue` GROUP BY dataset").set_index("dataset")
    for ds, d in queue.groupby("dataset"):
        assert r.loc[ds, "n"] == len(d) and r.loc[ds, "lo"] == 1 and r.loc[ds, "hi"] == len(d)
