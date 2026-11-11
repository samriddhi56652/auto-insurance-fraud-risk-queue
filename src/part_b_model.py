"""Part B (SYNTHETIC) — claim-level fraud models with a time-based split.

train      2023-01 .. 2024-06   fit models (hyper-parameters tuned on an inner time split)
calibrate  2024-07 .. 2024-12   choose the model, fit Platt calibration, nothing else
test       2025-01 .. 2025-12   touched once, for the reported numbers

Run:  python -m src.part_b_model
Outputs: reports/part_b/*.csv|json, reports/figures/b_*.png
"""
import hashlib

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve

from src import bq, config
from src import modeling as M

OUT = config.REPORTS / "part_b"
NUM = ["claim_amount_inr", "amount_to_limit_ratio", "just_under_limit", "report_delay_days",
       "police_report_filed", "witness_present", "vehicle_age_years", "tier_rank", "monthly_premium_inr",
       "premium_change_12m", "days_since_policy_start", "days_since_coverage_change", "days_since_upgrade",
       "claims_prev_3m", "claims_prev_6m", "claims_prev_12m", "claim_freq_change_3m", "prior_claims_total",
       "days_since_prev_claim", "amount_vs_own_avg", "amount_z_own_history", "amount_pct_diff_prev",
       "same_type_as_prev", "same_shop_as_prev", "prior_similar_claims", "shop_claims_prev_90d",
       "shop_claims_prev_365d", "shared_contact_policyholders", "shared_contact_claims_prev_180d",
       "third_party_is_customer", "third_party_claims_prev_180d"]
CAT = ["claim_type", "vehicle_segment"]
# Deliberately NOT features: age_band, region (kept only to audit flag rates), ids, dates.

DESCRIBE = {
    "days_since_policy_start": lambda v: f"claim {v:.0f} days after policy start",
    "days_since_upgrade": lambda v: f"incident {v:.0f} days after a coverage upgrade",
    "days_since_coverage_change": lambda v: f"incident {v:.0f} days after a coverage change",
    "amount_to_limit_ratio": lambda v: f"amount is {v:.0%} of the Rs 50k fast-track limit",
    "just_under_limit": lambda v: "amount just under the Rs 50k fast-track limit" if v else "amount not near the limit",
    "claim_amount_inr": lambda v: f"claim amount Rs {v:,.0f}",
    "claims_prev_3m": lambda v: f"{v:.0f} claims in the previous 3 months",
    "claims_prev_6m": lambda v: f"{v:.0f} claims in the previous 6 months",
    "claims_prev_12m": lambda v: f"{v:.0f} claims in the previous 12 months",
    "claim_freq_change_3m": lambda v: f"claim frequency changed by {v:+.0f} vs the 3 months before",
    "prior_claims_total": lambda v: f"{v:.0f} earlier claims on this policy",
    "prior_similar_claims": lambda v: f"{v:.0f} earlier claims of the same type and similar amount",
    "amount_pct_diff_prev": lambda v: f"amount within {v:.0%} of the previous claim",
    "amount_vs_own_avg": lambda v: f"amount {v:.1f}x the policyholder's own average",
    "amount_z_own_history": lambda v: f"amount {v:+.1f} SD vs own history",
    "same_type_as_prev": lambda v: "same claim type as previous claim" if v else "different type from previous claim",
    "same_shop_as_prev": lambda v: "same repair shop as previous claim" if v else "different shop from previous claim",
    "days_since_prev_claim": lambda v: f"{v:.0f} days since the previous claim",
    "shop_claims_prev_90d": lambda v: f"repair shop had {v:.0f} claims in the previous 90 days",
    "shop_claims_prev_365d": lambda v: f"repair shop had {v:.0f} claims in the previous year",
    "shared_contact_policyholders": lambda v: f"address/phone shared with {v:.0f} other policyholders",
    "shared_contact_claims_prev_180d": lambda v: f"{v:.0f} recent claims by others at the same address/phone",
    "third_party_is_customer": lambda v: "third party is another of our policyholders" if v else "third party is external",
    "third_party_claims_prev_180d": lambda v: f"third party filed {v:.0f} claims in the previous 180 days",
    "report_delay_days": lambda v: f"reported {v:.0f} days after the incident",
    "police_report_filed": lambda v: "police report filed" if v else "no police report",
    "witness_present": lambda v: "witness present" if v else "no witness",
    "tier_rank": lambda v: f"{['Basic', 'Standard', 'Comprehensive'][int(v)]} cover",
    "claim_type": lambda v: f"{v} claim",
    "vehicle_segment": lambda v: f"{v} vehicle",
    "vehicle_age_years": lambda v: f"vehicle {v:.0f} years old",
    "monthly_premium_inr": lambda v: f"premium Rs {v:,.0f}/month",
    "premium_change_12m": lambda v: f"premium {v:+.0%} vs a year ago",
}


def describe(feat, v):
    if pd.isna(v):
        return f"{feat} not available"
    f = DESCRIBE.get(feat)
    return f(v) if f else f"{feat} = {v}"


def load() -> pd.DataFrame:
    df = bq.query("SELECT * FROM `{project}.fraud_marts.b_claim_features`")
    for c in NUM:
        df[c] = pd.to_numeric(df[c].astype(object).where(df[c].notna(), np.nan)).astype(float)
    for c in CAT:
        df[c] = df[c].astype(str)
    df["fraud_label"] = df["fraud_label"].astype(int)
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df.sort_values(["report_date", "claim_id"]).reset_index(drop=True)


def tune(kind, weighted, tr):
    """Pick hyper-parameters on an inner time split INSIDE the training period."""
    inner_fit = tr[tr["report_date"] <= "2023-12-31"]
    inner_val = tr[tr["report_date"] > "2023-12-31"]
    best = None
    for params in M.PARAM_GRID[kind]:
        _, s = M.fit_predict(kind, weighted, params, inner_fit, inner_fit["fraud_label"], inner_val, NUM, CAT)
        ap = average_precision_score(inner_val["fraud_label"], s)
        if best is None or ap > best[1]:
            best = (params, ap)
    return best


def policyholder_bucket(ph: pd.Series) -> pd.Series:
    """Deterministic 0-99 bucket per policyholder (for the grouped hold-out)."""
    return ph.map(lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16) % 100)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    tr, ca, te = (df[df["split"] == s].copy() for s in ("train", "calibrate", "test"))
    assert tr["report_date"].max() < ca["report_date"].min() <= ca["report_date"].max() < te["report_date"].min()

    # ---------------- 4 models: {logreg, lgbm} x {unweighted, class-weighted}
    rows, fitted = [], {}
    for kind in ("logreg", "lgbm"):
        for weighted in (False, True):
            name = f"{kind}_{'weighted' if weighted else 'unweighted'}"
            params, inner_ap = tune(kind, weighted, tr)
            model, s = M.fit_predict(kind, weighted, params, tr, tr["fraud_label"], pd.concat([ca, te]), NUM, CAT)
            s_ca, s_te = s[: len(ca)], s[len(ca):]
            cal = M.PlattCalibrator().fit(s_ca, ca["fraud_label"])
            p_te = cal.predict(s_te)
            m_ca = M.evaluate(ca["fraud_label"], s_ca)
            m_te = M.evaluate(te["fraud_label"], s_te, prob=p_te)
            m_te["brier_uncalibrated"] = M.brier_score_loss(te["fraud_label"], s_te)
            fitted[name] = dict(model=model, kind=kind, weighted=weighted, params=params, cal=cal,
                                s_ca=s_ca, s_te=s_te, p_te=p_te)
            rows.append({"model": name, "params": params, "inner_val_pr_auc": inner_ap,
                         "calibrate_pr_auc": m_ca["pr_auc"],
                         **{f"test_{k}": v for k, v in m_te.items()}})
            print(f"{name:20s} calib PR-AUC {m_ca['pr_auc']:.3f} | test PR-AUC {m_te['pr_auc']:.3f} "
                  f"ROC {m_te['roc_auc']:.3f} P@5% {m_te['precision_top5']:.3f} R@5% {m_te['recall_top5']:.3f}")
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "model_comparison.csv", index=False)

    # model chosen on the CALIBRATION period, never on test
    best_name = comp.sort_values("calibrate_pr_auc", ascending=False)["model"].iloc[0]
    best = fitted[best_name]
    print(f"\nSelected on calibration PR-AUC: {best_name}")
    y_te = te["fraud_label"].to_numpy()
    s_te, p_te = best["s_te"], best["p_te"]

    headline = {"selected_model": best_name, "params": best["params"],
                **M.evaluate(y_te, s_te, prob=p_te),
                "pr_auc_95ci": M.bootstrap_ci(y_te, s_te, average_precision_score),
                "precision_top5_95ci": M.bootstrap_ci(
                    y_te, s_te, lambda y, s: M.top_k_metrics(y, s)["precision_top5"])}
    M.save_json(headline, OUT / "headline_metrics.json")

    # ---------------- calibration check
    rel = pd.concat([M.reliability_table(y_te, p_te).assign(scores="calibrated"),
                     M.reliability_table(y_te, s_te).assign(scores="raw_model_output")])
    rel.to_csv(OUT / "calibration_test.csv", index=False)

    # ---------------- grouped robustness: unseen policyholders only
    te_b, tr_b = policyholder_bucket(te["policyholder_id"]), policyholder_bucket(tr["policyholder_id"])
    hold = te_b < 30
    tr_g = tr[(tr_b >= 30).to_numpy()]
    _, s_g = M.fit_predict(best["kind"], best["weighted"], best["params"], tr_g, tr_g["fraud_label"],
                           te[hold.to_numpy()], NUM, CAT)
    grouped = pd.DataFrame([
        {"setup": "time split (all policyholders)", **M.evaluate(y_te[hold.to_numpy()], s_te[hold.to_numpy()])},
        {"setup": "time + grouped split (test policyholders never in training)",
         **M.evaluate(y_te[hold.to_numpy()], s_g)},
    ])
    grouped.to_csv(OUT / "grouped_split_check.csv", index=False)

    # ---------------- queue, explanations
    te = te.assign(score=s_te, calibrated_p=p_te)
    te["rank"] = te["score"].rank(ascending=False, method="first").astype(int)
    if best["kind"] == "lgbm":
        Xte = M.prep_lgbm(te[NUM + CAT], CAT)
        for c in CAT:
            Xte[c] = pd.Categorical(Xte[c].astype(object), categories=M.prep_lgbm(tr[[c]], [c])[c].cat.categories)
        contrib = M.lgbm_contributions(best["model"], Xte)
        raw = best["model"].predict_proba(Xte, raw_score=True)
        assert np.allclose(contrib.sum(axis=1), raw, atol=1e-6), "SHAP values do not add up to model output"
        reasons = M.top_reasons(contrib, Xte, describe)
        glob = contrib.drop(columns="_base_value").abs().mean().sort_values(ascending=False)
        signed = contrib.drop(columns="_base_value").mean()
    else:
        raise NotImplementedError("explanations implemented for the LightGBM model")
    te = pd.concat([te, reasons], axis=1)
    pd.DataFrame({"feature": glob.index, "mean_abs_shap": glob.values,
                  "mean_shap": signed[glob.index].values}).to_csv(OUT / "global_drivers.csv", index=False)

    # ---------------- planted-pattern recovery (ground truth never touched before this point)
    gt = pd.read_csv(config.DATA_SYN / "ground_truth_planted.csv")
    te = te.merge(gt[["claim_id", "pattern", "ring_id"]], on="claim_id", how="left")
    n = len(te)
    rec_rows = []
    for pat, d in te[te["pattern"].notna()].groupby("pattern"):
        r = {"pattern": pat, "planted_in_test": len(d), "labelled_by_siu": int(d["fraud_label"].sum())}
        for k in config.TOP_K:
            r[f"recovered_top{int(k * 100)}"] = (d["rank"] <= round(k * n)).mean()
        rec_rows.append(r)
    missed = te[te["pattern"].notna() & (te["fraud_label"] == 0)]
    rec_rows.append({"pattern": "ALL planted but NOT labelled by past SIU", "planted_in_test": len(missed),
                     "labelled_by_siu": 0,
                     **{f"recovered_top{int(k * 100)}": (missed["rank"] <= round(k * n)).mean() for k in config.TOP_K}})
    pd.DataFrame(rec_rows).to_csv(OUT / "pattern_recovery.csv", index=False)

    # ---------------- cost-based threshold (stated assumptions in src/config.py)
    amt = te["claim_amount_inr"].to_numpy()
    y = te["fraud_label"].to_numpy()

    def net_benefit(flag, C, r):
        return float((y[flag] * amt[flag] * r).sum() - C * flag.sum())

    order = np.argsort(-te["calibrated_p"].to_numpy(), kind="stable")
    curve = []
    for pct in np.arange(0.005, 0.305, 0.005):
        flag = np.zeros(n, bool)
        flag[order[: int(round(pct * n))]] = True
        curve.append({"pct_reviewed": pct, "n_reviewed": int(flag.sum()), "fraud_caught": int(y[flag].sum()),
                      "fraud_amount_caught_inr": float((y[flag] * amt[flag]).sum()),
                      "net_benefit_inr": net_benefit(flag, config.INVESTIGATION_COST_INR, config.RECOVERY_RATE)})
    pd.DataFrame(curve).to_csv(OUT / "cost_curve_by_queue_depth.csv", index=False)

    sens = []
    for C in config.SENS_INVESTIGATION_COST:
        for r in config.SENS_RECOVERY_RATE:
            # rule: investigate when expected prevented loss exceeds investigation cost
            flag = te["calibrated_p"].to_numpy() * amt * r > C
            sens.append({"investigation_cost_inr": C, "recovery_rate": r, "n_flagged": int(flag.sum()),
                         "pct_flagged": flag.mean(), "fraud_caught": int(y[flag].sum()),
                         "precision": y[flag].mean() if flag.any() else np.nan,
                         "net_benefit_inr": net_benefit(flag, C, r),
                         "net_benefit_review_all_inr": net_benefit(np.ones(n, bool), C, r)})
    pd.DataFrame(sens).to_csv(OUT / "cost_threshold_sensitivity.csv", index=False)
    te["flag_cost_rule"] = (te["calibrated_p"] * te["claim_amount_inr"] * config.RECOVERY_RATE
                            > config.INVESTIGATION_COST_INR).astype(int)
    te["flag_top5"] = (te["rank"] <= round(0.05 * n)).astype(int)

    # ---------------- fairness audit (age band, region are NOT model inputs)
    fair = pd.concat([M.group_flag_rates(te, g, f, "fraud_label").assign(operating_point=f)
                      for g in ("age_band", "region") for f in ("flag_top5", "flag_cost_rule")])
    fair.to_csv(OUT / "fairness_flag_rates.csv", index=False)

    keep = ["claim_id", "policyholder_id", "report_date", "claim_type", "claim_amount_inr", "age_band", "region",
            "score", "calibrated_p", "rank", "top_reason_1", "top_reason_2", "top_reason_3",
            "flag_top5", "flag_cost_rule", "fraud_label", "pattern", "ring_id"]
    te[keep].sort_values("rank").to_csv(OUT / "test_queue.csv", index=False)

    # ---------------- figures
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, f in fitted.items():
        pr, rc, _ = precision_recall_curve(y_te, f["s_te"])
        ax.plot(rc, pr, label=f"{name} (AP {average_precision_score(y_te, f['s_te']):.3f})")
    ax.axhline(y_te.mean(), ls="--", c="grey", label=f"base rate {y_te.mean():.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Part B (SYNTHETIC) - test 2025 PR curves")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(config.FIGURES / "b_pr_curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    for lab, d in rel.groupby("scores"):
        ax.plot(d["mean_predicted"], d["observed_rate"], "o-", label=lab)
    ax.plot([0, 1], [0, 1], "--", c="grey")
    lim = max(rel["mean_predicted"].max(), rel["observed_rate"].max()) * 1.05
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Mean predicted", ylabel="Observed fraud rate",
           title=f"Part B (SYNTHETIC) - calibration, {best_name}")
    ax.legend(); fig.tight_layout(); fig.savefig(config.FIGURES / "b_calibration.png", dpi=150); plt.close(fig)

    top = glob.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(top.index, top.values)
    ax.set(xlabel="mean |SHAP| (log-odds)", title="Part B (SYNTHETIC) - global fraud drivers")
    fig.tight_layout(); fig.savefig(config.FIGURES / "b_global_drivers.png", dpi=150); plt.close(fig)

    print(comp[["model", "calibrate_pr_auc", "test_pr_auc", "test_roc_auc", "test_precision_top5",
                "test_recall_top5", "test_brier"]].round(3).to_string(index=False))
    print(pd.read_csv(OUT / "pattern_recovery.csv").round(3).to_string(index=False))
    print(grouped[["setup", "n", "pr_auc", "precision_top5"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
