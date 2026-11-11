"""Part B (SYNTHETIC) — panel econometrics on the policyholder x month table.

Question the ML model cannot answer: when a GIVEN policyholder upgrades cover,
does THEIR claim / fraud probability rise afterwards (within-person effect),
or do people who upgrade simply claim more in general (between-person)?

Linear probability models (coefficients x 100 = percentage points per month):
  pooled OLS     : ignores who the policyholder is
  random effects : policyholder effect assumed uncorrelated with the regressors
  fixed effects   : policyholder effect removed (within transformation) + month effects
Hausman test: are FE and RE significantly different? If yes, RE is biased.

Run:  python -m src.part_b_panel
"""
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects
from scipy import stats

from src import bq, config

OUT = config.REPORTS / "part_b"
X = ["recent_upgrade_2m", "new_policy_3m", "tier_standard", "tier_comprehensive"]


def load() -> pd.DataFrame:
    df = bq.query("""
        SELECT policyholder_id, month, tier_rank, tenure_months,
               n_claims_incident, n_fraud_incident,
               IFNULL(DATE_DIFF(month, last_upgrade_month, MONTH) BETWEEN 0 AND 2, FALSE) AS recent_upgrade_2m
        FROM `{project}.fraud_marts.b_panel_months`""")
    df["month"] = pd.to_datetime(df["month"])
    df["any_claim"] = (df["n_claims_incident"] > 0).astype(float)
    df["fraud_claim"] = (df["n_fraud_incident"] > 0).astype(float)
    df["recent_upgrade_2m"] = df["recent_upgrade_2m"].astype(float)
    df["new_policy_3m"] = (df["tenure_months"] <= 2).astype(float)
    df["tier_standard"] = (df["tier_rank"] == 1).astype(float)
    df["tier_comprehensive"] = (df["tier_rank"] == 2).astype(float)
    return df.set_index(["policyholder_id", "month"]).sort_index()


def hausman(fe, re, cols):
    b = fe.params[cols] - re.params[cols]
    v = fe.cov.loc[cols, cols] - re.cov.loc[cols, cols]
    stat = float(b @ np.linalg.pinv(v.to_numpy()) @ b)
    return stat, len(cols), float(stats.chi2.sf(stat, len(cols)))


def main():
    df = load()
    month_d = pd.get_dummies(df.index.get_level_values("month"), prefix="m", drop_first=True, dtype=float)
    month_d.index = df.index
    rows, summaries = [], []
    for y in ("any_claim", "fraud_claim"):
        exog = df[X].assign(const=1.0).join(month_d)
        pooled = PooledOLS(df[y], exog).fit(cov_type="clustered", cluster_entity=True)
        re = RandomEffects(df[y], exog).fit(cov_type="clustered", cluster_entity=True)
        fe = PanelOLS(df[y], df[X], entity_effects=True, time_effects=True).fit(cov_type="clustered",
                                                                                 cluster_entity=True)
        # classical (unadjusted) covariances for the Hausman statistic
        re_u = RandomEffects(df[y], exog).fit()
        fe_u = PanelOLS(df[y], df[X], entity_effects=True, time_effects=True).fit()
        h_stat, h_df, h_p = hausman(fe_u, re_u, X)
        for name, res in (("pooled_ols", pooled), ("random_effects", re), ("fixed_effects", fe)):
            ci = res.conf_int()
            for x in X:
                rows.append({"outcome": y, "model": name, "variable": x,
                             "coef_pp": 100 * res.params[x], "ci_low_pp": 100 * ci.loc[x, "lower"],
                             "ci_high_pp": 100 * ci.loc[x, "upper"], "p_value": res.pvalues[x],
                             "n_obs": res.nobs, "n_entities": df.index.get_level_values(0).nunique()})
        rows.append({"outcome": y, "model": "hausman_fe_vs_re", "variable": "chi2",
                     "coef_pp": h_stat, "p_value": h_p, "ci_low_pp": np.nan, "ci_high_pp": np.nan,
                     "n_obs": h_df, "n_entities": np.nan})
        summaries.append(f"===== outcome: {y} =====\n--- FIXED EFFECTS ---\n{fe.summary}\n"
                         f"--- RANDOM EFFECTS (theta mean {float(np.mean(re.theta)):.3f}) ---\n"
                         f"{re.params[X].to_string()}\nHausman chi2={h_stat:.2f} df={h_df} p={h_p:.4g}\n")
        print(f"{y}: Hausman chi2={h_stat:.2f}, p={h_p:.3g}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "panel_fe_re.csv", index=False)
    (OUT / "panel_fe_re_summary.txt").write_text("\n".join(summaries))
    print(res[res["model"] != "hausman_fe_vs_re"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
