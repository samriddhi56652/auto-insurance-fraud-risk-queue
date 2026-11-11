"""Shared model, metric and explanation code for Part A and Part B."""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

# ----------------------------------------------------------------- models


def make_model(kind: str, weighted: bool, num_cols, cat_cols, params: dict | None = None):
    """kind: 'logreg' or 'lgbm'. weighted: class weights 'balanced' vs none."""
    params = params or {}
    cw = "balanced" if weighted else None
    if kind == "logreg":
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                              ("sc", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat_cols),
        ])
        clf = LogisticRegression(C=params.get("C", 1.0), class_weight=cw, max_iter=5000)
        return Pipeline([("pre", pre), ("clf", clf)])
    if kind == "lgbm":
        return lgb.LGBMClassifier(
            n_estimators=params.get("n_estimators", 300), learning_rate=params.get("learning_rate", 0.03),
            num_leaves=params.get("num_leaves", 15), min_child_samples=params.get("min_child_samples", 30),
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            class_weight=cw, random_state=config.SEED, verbose=-1, n_jobs=4)
    raise ValueError(kind)


PARAM_GRID = {
    "logreg": [{"C": c} for c in (0.01, 0.1, 1.0)],
    "lgbm": [{"num_leaves": nl, "n_estimators": ne, "min_child_samples": mc}
             for nl in (7, 15, 31) for ne in (200, 500) for mc in (30,)],
}


def prep_lgbm(X: pd.DataFrame, cat_cols) -> pd.DataFrame:
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    return X


def fit_predict(kind, weighted, params, Xtr, ytr, Xte, num_cols, cat_cols):
    model = make_model(kind, weighted, num_cols, cat_cols, params)
    if kind == "lgbm":
        cols = num_cols + cat_cols
        Xtr_, Xte_ = prep_lgbm(Xtr[cols], cat_cols), prep_lgbm(Xte[cols], cat_cols)
        for c in cat_cols:   # same category levels in train and scoring data
            levels = Xtr_[c].cat.categories
            Xte_[c] = pd.Categorical(Xte_[c].astype(object), categories=levels)
        model.fit(Xtr_, ytr)
        return model, model.predict_proba(Xte_)[:, 1]
    model.fit(Xtr[num_cols + cat_cols], ytr)
    return model, model.predict_proba(Xte[num_cols + cat_cols])[:, 1]


# ----------------------------------------------------------------- metrics


def top_k_metrics(y, score, ks=config.TOP_K) -> dict:
    """Precision and recall when the SIU reviews the top k% of the queue."""
    y = np.asarray(y)
    order = np.argsort(-np.asarray(score), kind="stable")
    out = {}
    for k in ks:
        n = max(1, int(round(k * len(y))))
        hits = y[order[:n]].sum()
        out[f"precision_top{int(k * 100)}"] = hits / n
        out[f"recall_top{int(k * 100)}"] = hits / max(1, y.sum())
        out[f"n_reviewed_top{int(k * 100)}"] = n
    return out


def evaluate(y, score, prob=None) -> dict:
    y = np.asarray(y)
    m = {"n": len(y), "n_fraud": int(y.sum()), "base_rate": y.mean(),
         "pr_auc": average_precision_score(y, score), "roc_auc": roc_auc_score(y, score)}
    m.update(top_k_metrics(y, score))
    if prob is not None:
        m["brier"] = brier_score_loss(y, prob)
    return m


def bootstrap_ci(y, score, fn, n_boot=1000, seed=config.SEED):
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        vals.append(fn(y[i], score[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def reliability_table(y, prob, bins=10) -> pd.DataFrame:
    """Calibration check: mean predicted vs observed fraud rate per score decile."""
    d = pd.DataFrame({"y": y, "p": prob})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), bins, labels=False)
    return d.groupby("bin").agg(n=("y", "size"), mean_predicted=("p", "mean"),
                                observed_rate=("y", "mean")).reset_index()


class PlattCalibrator:
    """Logistic calibration of a score on a held-out set (logit of score -> probability)."""

    def fit(self, score, y):
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(self._x(score), y)
        return self

    @staticmethod
    def _x(score):
        s = np.clip(np.asarray(score, float), 1e-6, 1 - 1e-6)
        return np.log(s / (1 - s)).reshape(-1, 1)

    def predict(self, score):
        return self.lr.predict_proba(self._x(score))[:, 1]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return centre - half, centre + half


def group_flag_rates(df: pd.DataFrame, group_col: str, flag_col: str, label_col: str) -> pd.DataFrame:
    rows = []
    for g, d in df.groupby(group_col, observed=True):
        n, k = len(d), int(d[flag_col].sum())
        lo, hi = wilson_ci(k, n)
        rows.append({"group_col": group_col, "group": g, "n_claims": n, "n_flagged": k,
                     "flag_rate": k / n, "flag_rate_ci_low": lo, "flag_rate_ci_high": hi,
                     "observed_fraud_rate": d[label_col].mean(),
                     "precision_in_group": d.loc[d[flag_col] == 1, label_col].mean() if k else np.nan})
    out = pd.DataFrame(rows)
    out["flag_rate_vs_overall"] = out["flag_rate"] / df[flag_col].mean()
    return out


# ----------------------------------------------------------------- explanations


def lgbm_contributions(model, X: pd.DataFrame) -> pd.DataFrame:
    """Exact TreeSHAP values (log-odds) from LightGBM itself, plus the base value."""
    contrib = model.booster_.predict(X, pred_contrib=True)
    cols = list(X.columns) + ["_base_value"]
    return pd.DataFrame(contrib, columns=cols, index=X.index)


def top_reasons(contrib: pd.DataFrame, X: pd.DataFrame, describe, n=3) -> pd.DataFrame:
    """For each row, the n features that pushed the score UP the most, as text."""
    c = contrib.drop(columns="_base_value")
    vals = c.to_numpy()
    order = np.argsort(-vals, axis=1)[:, :n]
    out = {f"top_reason_{j + 1}": [] for j in range(n)}
    for i in range(len(c)):
        for j in range(n):
            col_i = order[i, j]
            feat, v = c.columns[col_i], vals[i, col_i]
            out[f"top_reason_{j + 1}"].append(
                f"{describe(feat, X.iloc[i][feat])} (+{v:.2f})" if v > 0 else "")
    return pd.DataFrame(out, index=c.index)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
