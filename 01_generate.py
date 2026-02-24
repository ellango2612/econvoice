"""
EconVoice Survey Pipeline — Step 1: Data Generation
=====================================================
Simulates a 1,200-person online survey measuring economic anxiety and
policy preferences across the U.S. adult population.

Design choices for clarity:
  - 5 clean demographic variables (age, gender, education, region, income)
  - 4 survey questions with clear social-outcomes framing
  - Intentional online-panel bias: younger, higher-income, more educated
    respondents are over-represented (standard in opt-in panels)
  - Missingness only on income (sensitive question — realistic ~8%)

True support rates are derived from a simple logistic model with
interpretable coefficients, so weighting and modeling results are
easy to explain and validate.
"""

import json
import numpy as np
import pandas as pd
from scipy.special import expit

SEED = 42
rng  = np.random.default_rng(SEED)

N_POP    = 30_000   # simulated population
N_SAMPLE = 1_200    # target completed interviews

# ── Census-approximate population marginals ─────────────────────────────

POP = {
    "age":    {"18-34": 0.28, "35-49": 0.27, "50-64": 0.25, "65+": 0.20},
    "gender": {"Woman": 0.51, "Man": 0.49},
    "educ":   {"No college": 0.38, "Some college": 0.28, "Bachelor+": 0.34},
    "region": {"Northeast": 0.18, "Midwest": 0.21, "South": 0.38, "West": 0.23},
    "income": {"Under $40k": 0.30, "$40k–$80k": 0.35, "Over $80k": 0.35},
}

# ── True log-odds model (interpretable ground truth) ────────────────────
# Outcome: supports federal job guarantee / wage support policy

TRUE_COEF = {
    "intercept":        0.20,
    "age_35_49":       -0.15,
    "age_50_64":       -0.35,
    "age_65plus":      -0.55,
    "gender_man":      -0.30,
    "educ_some_coll":   0.15,
    "educ_bach_plus":  -0.25,   # higher earners less supportive
    "region_south":    -0.20,
    "region_midwest":  -0.05,
    "income_mid":      -0.10,
    "income_high":     -0.60,   # key driver: income → less support
}

def draw(mapping, size):
    return rng.choice(list(mapping), size=size, p=list(mapping.values()))

# ── Simulate population ─────────────────────────────────────────────────

pop = pd.DataFrame({
    "age":    draw(POP["age"],    N_POP),
    "gender": draw(POP["gender"], N_POP),
    "educ":   draw(POP["educ"],   N_POP),
    "region": draw(POP["region"], N_POP),
    "income": draw(POP["income"], N_POP),
})

# Build feature matrix
def features(df):
    f = pd.DataFrame(index=df.index)
    f["intercept"]      = 1.0
    f["age_35_49"]      = (df["age"] == "35-49").astype(float)
    f["age_50_64"]      = (df["age"] == "50-64").astype(float)
    f["age_65plus"]     = (df["age"] == "65+").astype(float)
    f["gender_man"]     = (df["gender"] == "Man").astype(float)
    f["educ_some_coll"] = (df["educ"] == "Some college").astype(float)
    f["educ_bach_plus"] = (df["educ"] == "Bachelor+").astype(float)
    f["region_south"]   = (df["region"] == "South").astype(float)
    f["region_midwest"] = (df["region"] == "Midwest").astype(float)
    f["income_mid"]     = (df["income"] == "$40k–$80k").astype(float)
    f["income_high"]    = (df["income"] == "Over $80k").astype(float)
    return f

coefs = np.array([TRUE_COEF[c] for c in features(pop).columns])
pop["support_prob"]   = expit(features(pop).values @ coefs)
pop["supports_policy"] = (rng.random(N_POP) < pop["support_prob"]).astype(int)

print(f"True population support rate: {pop['supports_policy'].mean():.1%}")

# ── Biased online-panel sampling ────────────────────────────────────────
# Young, high-income, educated respondents more likely to be sampled

def sel_prob(row):
    p = 0.04
    if row["age"]    == "18-34":      p *= 3.5
    elif row["age"]  == "35-49":      p *= 1.8
    elif row["age"]  == "65+":        p *= 0.5
    if row["educ"]   == "Bachelor+":  p *= 2.0
    elif row["educ"] == "No college": p *= 0.6
    if row["income"] == "Over $80k":  p *= 2.2
    elif row["income"]=="Under $40k": p *= 0.5
    return float(np.clip(p, 0.001, 0.99))

pop["sel_prob"] = pop.apply(sel_prob, axis=1)
pop["selected"] = rng.random(N_POP) < pop["sel_prob"]

sample = pop[pop["selected"]].sample(N_SAMPLE, random_state=SEED).copy().reset_index(drop=True)
print(f"Unweighted sample support rate: {sample['supports_policy'].mean():.1%}  (biased high)")

# ── Survey questions ─────────────────────────────────────────────────────

# Q1: Policy support (main outcome)
likert = ["Strongly oppose","Somewhat oppose","Neither","Somewhat support","Strongly support"]
def to_likert(probs):
    cuts = [0.18, 0.38, 0.58, 0.78]
    noisy = np.clip(probs + rng.normal(0, 0.07, len(probs)), 0, 1)
    return [likert[np.searchsorted(cuts, v)] for v in noisy]

sample["q1_policy_likert"]  = to_likert(sample["support_prob"].values)
sample["q1_support_binary"] = sample["supports_policy"]

# Q2: Biggest economic worry (multinomial)
worries = ["Job loss / automation", "Wages not keeping up",
           "Healthcare costs", "Housing costs", "Retirement security"]
sample["q2_worry"] = rng.choice(worries, size=N_SAMPLE,
    p=[0.22, 0.28, 0.20, 0.18, 0.12])

# Q3: Personal financial stress (1–5 scale)
stress = np.clip(
    (1 - sample["support_prob"].values) * 3 + 2 + rng.normal(0, 0.6, N_SAMPLE),
    1, 5).round().astype(int)
sample["q3_financial_stress"] = stress

# Q4: Confidence economy will improve (1–5 scale)
confidence = np.clip(
    sample["support_prob"].values * 3 + 1 + rng.normal(0, 0.7, N_SAMPLE),
    1, 5).round().astype(int)
sample["q4_econ_confidence"] = confidence

# ── Missingness: income is sensitive, ~8% skip ─────────────────────────

miss_income = rng.random(N_SAMPLE) < 0.08
sample.loc[miss_income, "income"] = np.nan
print(f"Missing income: {miss_income.sum()} ({miss_income.mean():.1%})")

# ── Respondent IDs ───────────────────────────────────────────────────────

sample.insert(0, "respondent_id", [f"R{i:04d}" for i in range(N_SAMPLE)])

out_cols = [
    "respondent_id", "age", "gender", "educ", "region", "income",
    "q1_policy_likert", "q1_support_binary",
    "q2_worry", "q3_financial_stress", "q4_econ_confidence",
    "support_prob",
]
sample[out_cols].to_csv("data/raw_survey.csv", index=False)

# Save population targets
with open("data/pop_targets.json", "w") as f:
    json.dump({k: v for k, v in POP.items() if k != "income"}, f, indent=2)
# Include income targets separately (for imputation-aware weighting)
with open("data/pop_targets_full.json", "w") as f:
    json.dump(POP, f, indent=2)

print(f"\n✓ Saved {N_SAMPLE} respondents → data/raw_survey.csv")
print("\nSampling bias summary (sample vs. population):")
for var in ["age", "educ", "income"]:
    print(f"\n  {var}:")
    for cat, pop_pct in POP[var].items():
        samp_pct = (sample[var] == cat).mean()
        flag = " ⚠" if abs(samp_pct - pop_pct) > 0.05 else ""
        print(f"    {cat:20s}  pop={pop_pct:.0%}  sample={samp_pct:.0%}{flag}")
