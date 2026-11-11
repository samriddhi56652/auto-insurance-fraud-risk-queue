"""Build fraud_marts.risk_queue (both datasets, clearly labelled) and the CSVs
Tableau Public reads (it cannot connect to BigQuery).

Run:  python -m src.export      (after part_a_model, part_b_model, part_b_panel)
"""
import json

import pandas as pd

from src import bq, config

A, B = config.REPORTS / "part_a", config.REPORTS / "part_b"
DS_A = "A - REAL Kaggle claims (test year 1996)"
DS_B = "B - SYNTHETIC panel (test year 2025)"


def risk_queue() -> pd.DataFrame:
    a = pd.read_csv(A / "test_queue.csv")
    a = pd.DataFrame({
        "dataset": DS_A, "is_synthetic": False, "claim_id": "A-" + a["claim_id"].astype(str),
        "segment": a["base_policy"] + " / " + a["vehicle_category"], "claim_amount_inr": pd.NA,
        "age_band": a["age_band"], "group_2_name": "sex", "group_2": a["sex"],
        "score": a["score"], "calibrated_p": a["calibrated_p"], "rank": a["rank"],
        "top_reason_1": a["top_reason_1"], "top_reason_2": a["top_reason_2"], "top_reason_3": a["top_reason_3"],
        "flag_top5": a["flag_top5"], "flag_cost_rule": a["flag_cost_rule"],
        "historical_fraud_label": a["fraud_label"], "planted_pattern_synthetic": pd.NA})
    b = pd.read_csv(B / "test_queue.csv")
    b = pd.DataFrame({
        "dataset": DS_B, "is_synthetic": True, "claim_id": "B-" + b["claim_id"],
        "segment": b["claim_type"], "claim_amount_inr": b["claim_amount_inr"],
        "age_band": b["age_band"], "group_2_name": "region", "group_2": b["region"],
        "score": b["score"], "calibrated_p": b["calibrated_p"], "rank": b["rank"],
        "top_reason_1": b["top_reason_1"], "top_reason_2": b["top_reason_2"], "top_reason_3": b["top_reason_3"],
        "flag_top5": b["flag_top5"], "flag_cost_rule": b["flag_cost_rule"],
        "historical_fraud_label": b["fraud_label"], "planted_pattern_synthetic": b["pattern"]})
    q = pd.concat([a, b], ignore_index=True)
    q["queue_pct"] = q.groupby("dataset")["rank"].transform(lambda r: r / r.max())
    for c in ("top_reason_1", "top_reason_2", "top_reason_3"):
        q[c] = q[c].fillna("")
    return q.sort_values(["dataset", "rank"]).reset_index(drop=True)


def stacked(name: str, cols_a=None, cols_b=None) -> pd.DataFrame:
    parts = []
    for folder, ds, cols in ((A, DS_A, cols_a), (B, DS_B, cols_b)):
        p = folder / name
        if p.exists():
            d = pd.read_csv(p)
            parts.append((d[cols] if cols else d).assign(dataset=ds, is_synthetic=ds == DS_B))
    return pd.concat(parts, ignore_index=True)


def kpis() -> pd.DataFrame:
    rows = []
    for folder, ds in ((A, DS_A), (B, DS_B)):
        h = json.loads((folder / "headline_metrics.json").read_text())
        sens = pd.read_csv(folder / "cost_threshold_sensitivity.csv")
        base = sens[(sens["investigation_cost_inr"] == config.INVESTIGATION_COST_INR)
                    & (sens["recovery_rate"] == config.RECOVERY_RATE)].iloc[0]
        rows.append({"dataset": ds, "is_synthetic": ds == DS_B, "model": h["selected_model"],
                     "test_claims": h["n"], "test_fraud_labelled": h["n_fraud"], "base_rate": h["base_rate"],
                     "pr_auc": h["pr_auc"], "pr_auc_ci_low": h["pr_auc_95ci"][0], "pr_auc_ci_high": h["pr_auc_95ci"][1],
                     "roc_auc": h["roc_auc"], "precision_top1": h["precision_top1"], "recall_top1": h["recall_top1"],
                     "precision_top5": h["precision_top5"], "recall_top5": h["recall_top5"],
                     "precision_top10": h["precision_top10"], "recall_top10": h["recall_top10"],
                     "lift_top5": h["precision_top5"] / h["base_rate"], "brier": h["brier"],
                     "cost_rule_pct_flagged": base["pct_flagged"], "cost_rule_net_benefit_inr": base["net_benefit_inr"],
                     "assumed_investigation_cost_inr": config.INVESTIGATION_COST_INR,
                     "assumed_recovery_rate": config.RECOVERY_RATE,
                     "assumed_avg_fraud_loss_inr_part_a": config.PART_A_AVG_FRAUD_LOSS_INR})
    return pd.DataFrame(rows)


def main():
    config.DATA_EXPORTS.mkdir(parents=True, exist_ok=True)
    q = risk_queue()
    n_bq = bq.load_dataframe(q, bq.table_id(config.DS_MARTS, "risk_queue"))
    print(f"fraud_marts.risk_queue: {n_bq:,d} rows")
    outputs = {
        "risk_queue.csv": q,
        "kpis.csv": kpis(),
        "model_comparison.csv": stacked("model_comparison.csv"),
        "global_drivers.csv": stacked("global_drivers.csv"),
        "cost_curve_by_queue_depth.csv": stacked("cost_curve_by_queue_depth.csv"),
        "cost_threshold_sensitivity.csv": stacked("cost_threshold_sensitivity.csv"),
        "fairness_flag_rates.csv": stacked("fairness_flag_rates.csv"),
        "calibration_test.csv": stacked("calibration_test.csv"),
        "pattern_recovery_synthetic.csv": pd.read_csv(B / "pattern_recovery.csv").assign(is_synthetic=True),
        "panel_fe_re_synthetic.csv": pd.read_csv(B / "panel_fe_re.csv").assign(is_synthetic=True),
    }
    for name, df in outputs.items():
        df.to_csv(config.DATA_EXPORTS / name, index=False)
        print(f"  data/exports/{name:36s} {len(df):>6,d} rows")


if __name__ == "__main__":
    main()
