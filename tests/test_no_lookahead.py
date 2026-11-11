"""Look-ahead test for the Part B feature SQL (needs BigQuery).

Rebuild sql/part_b/03_features.sql on data truncated at a cut-off date. If any
feature used future information, claims reported before the cut-off would get
different values in the truncated build than in the full build.
The comparison runs in BigQuery with IS DISTINCT FROM, which treats NULL = NULL."""
import pytest

from src import bq, config

CUTOFF = "2024-06-30"
P = "{project}"
TMP = ("tmp_trunc_b_claims", "tmp_trunc_b_policy_months", "tmp_trunc_b_panel_months", "tmp_trunc_b_claim_features")
SWAPS = {
    f"{P}.fraud_clean.b_claims`": f"{P}.fraud_marts.tmp_trunc_b_claims`",
    f"{P}.fraud_clean.b_policy_months`": f"{P}.fraud_marts.tmp_trunc_b_policy_months`",
    f"{P}.fraud_marts.b_panel_months`": f"{P}.fraud_marts.tmp_trunc_b_panel_months`",
    f"{P}.fraud_marts.b_claim_features`": f"{P}.fraud_marts.tmp_trunc_b_claim_features`",
}


@pytest.mark.bq
def test_features_identical_on_truncated_history():
    c = bq.client()
    try:
        c.query(bq.render(f"""
            CREATE OR REPLACE TABLE `{P}.fraud_marts.tmp_trunc_b_claims` AS
            SELECT * FROM `{P}.fraud_clean.b_claims` WHERE report_date <= DATE '{CUTOFF}';
            CREATE OR REPLACE TABLE `{P}.fraud_marts.tmp_trunc_b_policy_months` AS
            SELECT * FROM `{P}.fraud_clean.b_policy_months` WHERE month <= DATE_TRUNC(DATE '{CUTOFF}', MONTH);
        """)).result()
        sql = (config.ROOT / "sql/part_b/03_features.sql").read_text(encoding="utf-8").split("-- Checkpoint")[0]
        for old, new in SWAPS.items():
            sql = sql.replace(old, new)
        for stmt in bq.split_statements(sql):
            c.query(bq.render(stmt)).result()

        schema = c.get_table(bq.table_id(config.DS_MARTS, "b_claim_features")).schema
        checks = []
        for f in schema:
            if f.name == "claim_id":
                continue
            if f.field_type in ("FLOAT", "FLOAT64"):
                cond = (f"(f.{f.name} IS NULL) != (t.{f.name} IS NULL) "
                        f"OR ABS(f.{f.name} - t.{f.name}) > 1e-9 * GREATEST(1, ABS(f.{f.name}))")
            else:
                cond = f"f.{f.name} IS DISTINCT FROM t.{f.name}"
            checks.append(f"COUNTIF({cond}) AS {f.name}")
        res = bq.query(f"""
            SELECT COUNT(*) AS n_full, COUNT(t.claim_id) AS n_matched, {", ".join(checks)}
            FROM (SELECT * FROM `{P}.fraud_marts.b_claim_features` WHERE report_date <= DATE '{CUTOFF}') f
            LEFT JOIN `{P}.fraud_marts.tmp_trunc_b_claim_features` t USING (claim_id)""").iloc[0]
        n_trunc = bq.query(f"SELECT COUNT(*) AS n FROM `{P}.fraud_marts.tmp_trunc_b_claim_features`").iloc[0]["n"]
    finally:
        for t in TMP:
            c.delete_table(bq.table_id(config.DS_MARTS, t), not_found_ok=True)

    assert res["n_full"] == res["n_matched"] == n_trunc > 1000
    changed = {k: int(v) for k, v in res.drop(["n_full", "n_matched"]).items() if v}
    assert not changed, f"features changed when future data was removed: {changed}"
