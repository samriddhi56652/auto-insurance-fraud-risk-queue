"""Checkpoint tests for the SYNTHETIC panel generator."""
import hashlib

import pytest

from src import config, generate_panel


@pytest.fixture(scope="module")
def tables():
    return generate_panel.generate()


def _digest(df):
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


def test_same_seed_gives_identical_output(tables):
    again = generate_panel.generate()
    for name in tables:
        assert _digest(tables[name]) == _digest(again[name]), name


def test_saved_files_match_generator(tables):
    for name, df in tables.items():
        path = config.DATA_SYN / f"{name}.csv"
        if not path.exists():
            pytest.skip("run python -m src.generate_panel first")
        assert path.read_text(encoding="utf-8") == df.to_csv(index=False, lineterminator="\n"), name


def test_ground_truth_not_in_claims_table(tables):
    claims = tables["claims"]
    assert not {"pattern", "ring_id", "is_fraud_true"} & set(claims.columns)
    gt = tables["ground_truth_planted"]
    assert set(gt["claim_id"]) <= set(claims["claim_id"])


def test_every_pattern_is_planted(tables):
    counts = tables["ground_truth_planted"]["pattern"].value_counts()
    for p in ["early_claim_new_policy", "early_claim_after_upgrade", "rising_frequency",
              "repeat_similar", "under_approval_limit", "staged_ring", "unpatterned"]:
        assert counts.get(p, 0) >= 50, p


def test_patterns_not_trivially_separable(tables):
    """Honest claims must also appear just under the approval limit (decoys)."""
    c = tables["claims"]
    gt_ids = set(tables["ground_truth_planted"]["claim_id"])
    amt = c["claim_amount_inr"].replace("", None).dropna().astype(float)
    near = c.loc[amt.index][(amt >= 45_000) & (amt < 50_000)]
    honest_near = (~near["claim_id"].isin(gt_ids)).sum()
    assert honest_near > 100
