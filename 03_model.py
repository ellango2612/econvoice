"""
EconVoice Survey Pipeline — Step 3: Predictive Modeling
========================================================
Trains two models to predict individual policy support from demographics
and attitudinal survey responses.

Models
------
  1. Logistic Regression  — interpretable coefficients for stakeholders
  2. Random Forest        — captures income × stress interactions

Also produces region-level weighted estimates (MRP-style) that let us
reliably report results for demographic cells too small to report directly.

Outputs
-------
data/predictions.csv      per-respondent predicted probabilities
data/region_estimates.csv region × income-group estimates with CIs
data/model_results.json   metrics + coefficients + feature importance
"""

import json
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")
df = pd.read_csv("data/weighted_survey.csv")
N  = len(df)
print(f"Loaded {N:,} respondents\n")

# ── Feature matrix ───────────────────────────────────────────────────────

def build_X(df):
    X = pd.DataFrame(index=df.index)
    # Demographics (reference: 18-34, Woman, No college, Northeast, Under $40k)
    X["age_35_49"]      = (df["age"] == "35-49").astype(float)
    X["age_50_64"]      = (df["age"] == "50-64").astype(float)
    X["age_65plus"]     = (df["age"] == "65+").astype(float)
    X["gender_man"]     = (df["gender"] == "Man").astype(float)
    X["educ_some_coll"] = (df["educ"] == "Some college").astype(float)
    X["educ_bach_plus"] = (df["educ"] == "Bachelor+").astype(float)
    X["region_midwest"] = (df["region"] == "Midwest").astype(float)
    X["region_south"]   = (df["region"] == "South").astype(float)
    X["region_west"]    = (df["region"] == "West").astype(float)
    # Income — NaN where missing (imputed in pipeline)
    X["income_mid"]     = (df["income"] == "$40k–$80k").astype(float)
    X["income_high"]    = (df["income"] == "Over $80k").astype(float)
    X.loc[df["income"].isna(), ["income_mid","income_high"]] = np.nan
    # Attitudinal
    X["financial_stress"]  = df["q3_financial_stress"].astype(float)
    X["econ_confidence"]   = df["q4_econ_confidence"].astype(float)
    X["worry_wages"]       = (df["q2_worry"] == "Wages not keeping up").astype(float)
    X["worry_jobs"]        = (df["q2_worry"] == "Job loss / automation").astype(float)
    return X

X = build_X(df)
y = df["q1_support_binary"].values
w = df["weight"].values
feat_names = X.columns.tolist()

# ── Pipelines ────────────────────────────────────────────────────────────

logit = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scl", StandardScaler()),
    ("mdl", LogisticRegression(C=1.0, max_iter=500, random_state=42)),
])

rf = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("mdl", RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=25,
        random_state=42, n_jobs=-1)),
])

# ── Cross-validated AUC ──────────────────────────────────────────────────

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("5-fold cross-validated AUC:")
for name, pipe in [("Logistic Regression", logit), ("Random Forest", rf)]:
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    print(f"  {name:22s}  {scores.mean():.4f} ± {scores.std():.4f}")

# ── Fit final models ─────────────────────────────────────────────────────

print("\nFitting on full sample (with survey weights)…")
logit.fit(X, y, mdl__sample_weight=w)
rf.fit(X, y,    mdl__sample_weight=w)

df["prob_logit"]    = logit.predict_proba(X)[:, 1]
df["prob_rf"]       = rf.predict_proba(X)[:, 1]
df["prob_ensemble"] = (df["prob_logit"] + df["prob_rf"]) / 2

# ── Evaluation ───────────────────────────────────────────────────────────

print("\nFull-sample metrics:")
results = {}
for name, col in [("logit","prob_logit"),("rf","prob_rf"),("ensemble","prob_ensemble")]:
    auc   = roc_auc_score(y, df[col])
    brier = brier_score_loss(y, df[col])
    results[name] = {"auc": round(auc,4), "brier": round(brier,4)}
    print(f"  {name:12s}  AUC={auc:.4f}  Brier={brier:.4f}")

# ── Logit coefficients ───────────────────────────────────────────────────

coef_df = pd.DataFrame({
    "feature": feat_names,
    "coef":    logit["mdl"].coef_[0],
    "odds_ratio": np.exp(logit["mdl"].coef_[0]),
}).sort_values("coef", ascending=False)

print("\nLogistic regression coefficients (log-odds):")
for _, r in coef_df.iterrows():
    direction = "▲" if r["coef"] > 0 else "▼"
    print(f"  {direction} {r['feature']:22s}  {r['coef']:+.3f}  (OR={r['odds_ratio']:.2f})")

# ── Feature importance ───────────────────────────────────────────────────

fi_df = pd.DataFrame({
    "feature":    feat_names,
    "importance": rf["mdl"].feature_importances_,
}).sort_values("importance", ascending=False)

print("\nRandom Forest feature importance (top 8):")
for _, r in fi_df.head(8).iterrows():
    bar = "█" * int(r["importance"] * 150)
    print(f"  {r['feature']:22s}  {r['importance']:.4f}  {bar}")

# ── Region × income subgroup estimates (MRP-style) ───────────────────────

print("\nRegion × income weighted estimates:")
deff = json.load(open("data/weight_stats.json"))["deff"]

rows = []
for region in df["region"].unique():
    for income in ["Under $40k", "$40k–$80k", "Over $80k"]:
        mask = (df["region"] == region) & (df["income"] == income)
        sub  = df[mask]
        if len(sub) < 10:
            continue
        wt_obs  = np.average(sub["q1_support_binary"], weights=sub["weight"])
        wt_pred = np.average(sub["prob_ensemble"],      weights=sub["weight"])
        n_eff   = len(sub) / deff
        se      = np.sqrt(wt_pred * (1 - wt_pred) / max(n_eff, 1))
        rows.append({
            "region": region, "income": income, "n": len(sub),
            "observed":  round(wt_obs, 4),
            "predicted": round(wt_pred, 4),
            "ci_lo":     round(max(0, wt_pred - 1.96*se), 4),
            "ci_hi":     round(min(1, wt_pred + 1.96*se), 4),
        })
        print(f"  {region:12s} × {income:14s}  "
              f"obs={wt_obs:.1%}  pred={wt_pred:.1%}  n={len(sub)}")

# ── Save ─────────────────────────────────────────────────────────────────

df.to_csv("data/predictions.csv", index=False)
pd.DataFrame(rows).to_csv("data/region_estimates.csv", index=False)

with open("data/model_results.json","w") as f:
    json.dump({
        "metrics":             results,
        "feature_importance":  fi_df.to_dict(orient="records"),
        "logit_coefs":         coef_df.to_dict(orient="records"),
    }, f, indent=2)

print("\n✓ data/predictions.csv")
print("✓ data/region_estimates.csv")
print("✓ data/model_results.json")
