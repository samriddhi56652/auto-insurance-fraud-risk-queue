"""Part A (REAL, Kaggle CC0, 1994-1996, anonymised) — claim-level fraud models.

Time-based split (profiling showed a usable Year column):
  train 1994 | calibrate 1995 (model choice + Platt calibration) | test 1996 (reported numbers)
Also reported: 5-fold stratified CV on 1994-95, to show what a random split would have claimed.

Run:  python -m src.part_a_model
"""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold

from src import bq, config
from src import modeling as M

OUT = config.REPORTS / "part_a"
NUM = ["week_of_month", "week_of_month_claimed", "months_accident_to_claim", "accident_urban",
       "fault_policyholder", "vehicle_price_ord", "deductible", "driver_rating", "days_policy_accident_ord",
       "days_policy_accident_none", "days_policy_claim_ord", "past_claims_ord", "vehicle_age_ord",
       "police_report_filed", "witness_present", "agent_external", "supplements_ord", "number_of_cars_ord"]
CAT = ["make", "base_policy", "vehicle_category", "day_of_week", "day_of_week_claimed", "address_change"]
# accident month was dropped: its fraud rate is ~0 for whole, different blocks of months each year
# (a labelling artefact), and removing it raised CALIBRATION-year PR-AUC from 0.113 to 0.179.
DEMOGRAPHIC = ["age", "age_band_ord", "sex", "marital_status"]   # ablation only
# Never features: claim_id (row id sorted by year), rep_number (claim handler id),
# year (the split variable), policy_type (= vehicle_category x base_policy).

PRICE = ["< 20k", "20-29k", "30-39k", "40-59k", "60-69k", "> 69k"]
PAST = ["no", "1", "2-4", "> 4"]
SUPP = ["no", "1-2", "3-5", "> 5"]
DESCRIBE = {
    "fault_policyholder": lambda v: "policyholder at fault" if v else "third party at fault",
    "base_policy": lambda v: f"{v} cover",
    "vehicle_category": lambda v: f"{v} vehicle",
    "vehicle_price_ord": lambda v: f"vehicle price {PRICE[int(v)]}",
    "past_claims_ord": lambda v: f"{PAST[int(v)]} past claims",
    "supplements_ord": lambda v: f"{SUPP[int(v)]} supplements",
    "deductible": lambda v: f"deductible {v:.0f}",
    "driver_rating": lambda v: f"driver rating {v:.0f}",
    "vehicle_age_ord": lambda v: "new vehicle" if v == 0 else f"vehicle age band {v:.0f} yrs",
    "address_change": lambda v: f"address change: {v}",
    "months_accident_to_claim": lambda v: f"claimed {v:.0f} months after the accident",
    "police_report_filed": lambda v: "police report filed" if v else "no police report",
    "witness_present": lambda v: "witness present" if v else "no witness",
    "agent_external": lambda v: "external agent" if v else "internal agent",
    "accident_urban": lambda v: "urban accident" if v else "rural accident",
    "make": lambda v: f"make {v}",
    "days_policy_accident_none": lambda v: "policy-to-accident days 'none'" if v else "policy-to-accident days recorded",
    "number_of_cars_ord": lambda v: f"cars on policy band {v:.0f}",
    "accident_month_cat": lambda v: f"accident in month {v}",
    "day_of_week": lambda v: f"accident on {v}",
    "day_of_week_claimed": lambda v: f"claimed on {v}",
    "week_of_month": lambda v: f"accident in week {v:.0f} of month",
    "week_of_month_claimed": lambda v: f"claimed in week {v:.0f} of month",
}


def describe(feat, v):
    if pd.isna(v):
        return f"{feat} not available"
    f = DESCRIBE.get(feat)
    return f(v) if f else f"{feat} = {v}"


def load():
    df = bq.query("SELECT * FROM `{project}.fraud_marts.a_claim_features`")
    df["accident_month_cat"] = df["accident_month"].astype(str)
    for c in NUM + ["age", "age_band_ord"]:
        df[c] = pd.to_numeric(df[c].astype(object).where(df[c].notna(), np.nan)).astype(float)
    for c in CAT + ["sex", "marital_status"]:
        df[c] = df[c].astype(str)
    df["fraud_label"] = df["fraud_label"].astype(int)
    # BigQuery returns rows in no fixed order; sort so CV folds and LightGBM bagging are reproducible
    return df.sort_values("claim_id").reset_index(drop=True)


def tune(kind, weighted, tr, num, cat):
    """Hyper-parameters chosen with stratified 3-fold CV INSIDE the 1994 training year."""
    skf = StratifiedKFold(3, shuffle=True, random_state=config.SEED)
    best = None
    for params in M.PARAM_GRID[kind]:
        aps = []
        for a, b in skf.split(tr, tr["fraud_label"]):
            _, s = M.fit_predict(kind, weighted, params, tr.iloc[a], tr["fraud_label"].iloc[a], tr.iloc[b], num, cat)
            aps.append(average_precision_score(tr["fraud_label"].iloc[b], s))
        if best is None or np.mean(aps) > best[1]:
            best = (params, float(np.mean(aps)))
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    tr, ca, te = (df[df["split"] == s].copy() for s in ("train", "calibrate", "test"))

    rows, fitted = [], {}
    for kind in ("logreg", "lgbm"):
        for weighted in (False, True):
            name = f"{kind}_{'weighted' if weighted else 'unweighted'}"
            params, inner = tune(kind, weighted, tr, NUM, CAT)
            model, s = M.fit_predict(kind, weighted, params, tr, tr["fraud_label"], pd.concat([ca, te]), NUM, CAT)
            s_ca, s_te = s[: len(ca)], s[len(ca):]
            cal = M.PlattCalibrator().fit(s_ca, ca["fraud_label"])
            p_te = cal.predict(s_te)
            m_ca, m_te = M.evaluate(ca["fraud_label"], s_ca), M.evaluate(te["fraud_label"], s_te, prob=p_te)
            m_te["brier_uncalibrated"] = M.brier_score_loss(te["fraud_label"], s_te)
            fitted[name] = dict(model=model, kind=kind, weighted=weighted, params=params, s_te=s_te, p_te=p_te)
            rows.append({"model": name, "params": params, "inner_cv_pr_auc": inner, "calibrate_pr_auc": m_ca["pr_auc"],
                         **{f"test_{k}": v for k, v in m_te.items()}})
            print(f"{name:20s} calib PR-AUC {m_ca['pr_auc']:.3f} | test PR-AUC {m_te['pr_auc']:.3f} "
                  f"ROC {m_te['roc_auc']:.3f} P@5% {m_te['precision_top5']:.3f} R@5% {m_te['recall_top5']:.3f}")
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "model_comparison.csv", index=False)

    best_name = comp.sort_values("calibrate_pr_auc", ascending=False)["model"].iloc[0]
    best = fitted[best_name]
    print(f"\nSelected on calibration PR-AUC: {best_name}")
    y_te, s_te, p_te = te["fraud_label"].to_numpy(), best["s_te"], best["p_te"]
    M.save_json({"selected_model": best_name, "params": best["params"], **M.evaluate(y_te, s_te, prob=p_te),
                 "pr_auc_95ci": M.bootstrap_ci(y_te, s_te, average_precision_score),
                 "precision_top5_95ci": M.bootstrap_ci(y_te, s_te, lambda y, s: M.top_k_metrics(y, s)["precision_top5"])},
                OUT / "headline_metrics.json")
    rel = pd.concat([M.reliability_table(y_te, p_te).assign(scores="calibrated"),
                     M.reliability_table(y_te, s_te).assign(scores="raw_model_output")])
    rel.to_csv(OUT / "calibration_test.csv", index=False)

    # ---------------- what a random stratified CV would have reported (1994-95 pooled)
    pool = pd.concat([tr, ca])
    skf = StratifiedKFold(5, shuffle=True, random_state=config.SEED)
    cv_ap = []
    for a, b in skf.split(pool, pool["fraud_label"]):
        _, s = M.fit_predict(best["kind"], best["weighted"], best["params"], pool.iloc[a],
                             pool["fraud_label"].iloc[a], pool.iloc[b], NUM, CAT)
        cv_ap.append(average_precision_score(pool["fraud_label"].iloc[b], s))
    _, s_pool = M.fit_predict(best["kind"], best["weighted"], best["params"], pool, pool["fraud_label"], te, NUM, CAT)
    te_rank_pool = pd.Series(s_pool, index=te.index).rank(ascending=False, method="first")
    pd.DataFrame([
        {"validation": "stratified 5-fold CV on 1994-95 (mean of folds)", "pr_auc": np.mean(cv_ap), "pr_auc_sd": np.std(cv_ap)},
        {"validation": "time split: train 1994, test 1996", "pr_auc": average_precision_score(y_te, s_te), "pr_auc_sd": np.nan},
        {"validation": "time split: train 1994-95, test 1996", "pr_auc": average_precision_score(y_te, s_pool), "pr_auc_sd": np.nan},
    ]).to_csv(OUT / "cv_vs_time_split.csv", index=False)

    # ---------------- accident-month decision: evidence on the calibration year (test shown for honesty)
    abl = []
    for label, cat in (("without accident month (main model)", CAT), ("with accident month", CAT + ["accident_month_cat"])):
        _, s = M.fit_predict(best["kind"], best["weighted"], best["params"], tr, tr["fraud_label"],
                             pd.concat([ca, te]), NUM, cat)
        abl.append({"features": label, "calibrate_pr_auc": average_precision_score(ca["fraud_label"], s[: len(ca)]),
                    "test_pr_auc": average_precision_score(te["fraud_label"], s[len(ca):])})
    pd.DataFrame(abl).to_csv(OUT / "accident_month_ablation.csv", index=False)

    # ---------------- demographic ablation: does adding age/sex/marital status change ranking quality?
    num_d = NUM + ["age", "age_band_ord"]
    cat_d = CAT + ["sex", "marital_status"]
    _, s_demo = M.fit_predict(best["kind"], best["weighted"], best["params"], tr, tr["fraud_label"], te, num_d, cat_d)
    pd.DataFrame([{"features": "without demographics (main model)", **M.evaluate(y_te, s_te)},
                  {"features": "with age, age band, sex, marital status", **M.evaluate(y_te, s_demo)}]
                 ).to_csv(OUT / "demographic_ablation.csv", index=False)

    # ---------------- queue + SHAP reasons
    te = te.assign(score=s_te, calibrated_p=p_te)
    te["rank"] = te["score"].rank(ascending=False, method="first").astype(int)
    if best["kind"] != "lgbm":
        raise NotImplementedError("explanations implemented for the LightGBM model")
    Xte = M.prep_lgbm(te[NUM + CAT], CAT)
    for c in CAT:
        Xte[c] = pd.Categorical(Xte[c].astype(object), categories=M.prep_lgbm(tr[[c]], [c])[c].cat.categories)
    contrib = M.lgbm_contributions(best["model"], Xte)
    assert np.allclose(contrib.sum(axis=1), best["model"].predict_proba(Xte, raw_score=True), atol=1e-6)
    te = pd.concat([te, M.top_reasons(contrib, Xte, describe)], axis=1)
    glob = contrib.drop(columns="_base_value").abs().mean().sort_values(ascending=False)
    pd.DataFrame({"feature": glob.index, "mean_abs_shap": glob.values,
                  "mean_shap": contrib[glob.index].mean().values}).to_csv(OUT / "global_drivers.csv", index=False)

    # ---------------- cost-based threshold (Part A has no claim amount -> one assumed average loss)
    n, y = len(te), te["fraud_label"].to_numpy()
    p = te["calibrated_p"].to_numpy()
    L = config.PART_A_AVG_FRAUD_LOSS_INR
    sens = []
    for C in config.SENS_INVESTIGATION_COST:
        for r in config.SENS_RECOVERY_RATE:
            thr = C / (L * r)                      # investigate when p * L * r > C
            flag = p > thr
            sens.append({"investigation_cost_inr": C, "recovery_rate": r, "assumed_avg_fraud_loss_inr": L,
                         "probability_threshold": thr, "n_flagged": int(flag.sum()), "pct_flagged": flag.mean(),
                         "fraud_caught": int(y[flag].sum()), "precision": y[flag].mean() if flag.any() else np.nan,
                         "net_benefit_inr": float(y[flag].sum() * L * r - C * flag.sum()),
                         "net_benefit_review_all_inr": float(y.sum() * L * r - C * n)})
    pd.DataFrame(sens).to_csv(OUT / "cost_threshold_sensitivity.csv", index=False)
    order = np.argsort(-p, kind="stable")
    curve = []
    for pct in np.arange(0.005, 0.305, 0.005):
        k = int(round(pct * n))
        caught = int(y[order[:k]].sum())
        curve.append({"pct_reviewed": pct, "n_reviewed": k, "fraud_caught": caught,
                      "fraud_amount_caught_inr": caught * L,      # assumed average loss per fraud
                      "net_benefit_inr": caught * L * config.RECOVERY_RATE - config.INVESTIGATION_COST_INR * k})
    pd.DataFrame(curve).to_csv(OUT / "cost_curve_by_queue_depth.csv", index=False)
    te["flag_cost_rule"] = (p * L * config.RECOVERY_RATE > config.INVESTIGATION_COST_INR).astype(int)
    te["flag_top5"] = (te["rank"] <= round(0.05 * n)).astype(int)

    # ---------------- fairness audit (age band and sex are NOT model inputs)
    # mitigation check: same model trained on 1994-95 (more recent data) -> does the proxy effect shrink?
    te["flag_top5_trained_1994_95"] = (te_rank_pool <= round(0.05 * n)).astype(int)
    fair = pd.concat([M.group_flag_rates(te, g, f, "fraud_label").assign(operating_point=f)
                      for g in ("age_band", "sex") for f in ("flag_top5", "flag_cost_rule", "flag_top5_trained_1994_95")])
    fair.to_csv(OUT / "fairness_flag_rates.csv", index=False)

    keep = ["claim_id", "year", "make", "base_policy", "vehicle_category", "age_band", "sex", "score", "calibrated_p",
            "rank", "top_reason_1", "top_reason_2", "top_reason_3", "flag_top5", "flag_cost_rule", "fraud_label"]
    te[keep].sort_values("rank").to_csv(OUT / "test_queue.csv", index=False)

    # ---------------- figures
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, f in fitted.items():
        pr, rc, _ = precision_recall_curve(y_te, f["s_te"])
        ax.plot(rc, pr, label=f"{name} (AP {average_precision_score(y_te, f['s_te']):.3f})")
    ax.axhline(y_te.mean(), ls="--", c="grey", label=f"base rate {y_te.mean():.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Part A (REAL, Kaggle) - test 1996 PR curves")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(config.FIGURES / "a_pr_curves.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 5))
    for lab, d in rel.groupby("scores"):
        ax.plot(d["mean_predicted"], d["observed_rate"], "o-", label=lab)
    lim = max(rel["mean_predicted"].max(), rel["observed_rate"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", c="grey")
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Mean predicted", ylabel="Observed fraud rate",
           title=f"Part A (REAL) - calibration, {best_name}")
    ax.legend(); fig.tight_layout(); fig.savefig(config.FIGURES / "a_calibration.png", dpi=150); plt.close(fig)
    top = glob.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(top.index, top.values)
    ax.set(xlabel="mean |SHAP| (log-odds)", title="Part A (REAL) - global fraud drivers")
    fig.tight_layout(); fig.savefig(config.FIGURES / "a_global_drivers.png", dpi=150); plt.close(fig)

    print(comp[["model", "calibrate_pr_auc", "test_pr_auc", "test_roc_auc", "test_precision_top5",
                "test_recall_top5", "test_brier", "test_brier_uncalibrated"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
