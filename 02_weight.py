"""
EconVoice Survey Pipeline — Step 2: Post-Stratification Weighting
=================================================================
Corrects for sampling bias using iterative proportional fitting (raking).

What raking does in plain English:
  Online opt-in panels over-represent certain groups. Raking assigns each
  respondent a weight so that the weighted sample matches known population
  proportions (from the Census). A 45-year-old low-income woman who is
  under-represented in the sample gets a weight > 1; an over-represented
  25-year-old college grad gets a weight < 1.

We weight on 4 variables: age, gender, education, region.
We exclude income from weighting targets because ~8% is missing —
instead income is imputed before modeling.

Outputs
-------
data/weighted_survey.csv    original data + 'weight' column
data/weight_stats.json      diagnostics: DEFF, effective N, convergence
"""

import json
import numpy as np
import pandas as pd

df     = pd.read_csv("data/raw_survey.csv")
target = json.load(open("data/pop_targets.json"))   # age, gender, educ, region
N      = len(df)

# ── Raking (iterative proportional fitting) ─────────────────────────────

def rake(df, targets, max_iter=100, tol=1e-8):
    w = np.ones(N)
    for i in range(max_iter):
        prev = w.copy()
        for var, cats in targets.items():
            for cat, pop_share in cats.items():
                mask = df[var] == cat
                if mask.sum() == 0:
                    continue
                target_sum  = pop_share * N
                current_sum = w[mask].sum()
                if current_sum > 0:
                    w[mask] *= target_sum / current_sum
        delta = np.abs(w - prev).max()
        if delta < tol:
            print(f"  Converged in {i+1} iterations (max Δw = {delta:.1e})")
            return w, i + 1
    print(f"  Did not converge after {max_iter} iterations")
    return w, max_iter

print("Raking on: age, gender, education, region")
raw_w, iters = rake(df, target)

# Normalize (mean = 1) then trim at 5th/95th percentile
raw_w /= raw_w.mean()
lo, hi = np.percentile(raw_w, 5), np.percentile(raw_w, 95)
w = np.clip(raw_w, lo, hi)
w /= w.mean()
df["weight"] = w

n_trimmed = int(((raw_w < lo) | (raw_w > hi)).sum())
print(f"  Trimmed {n_trimmed} weights to [{lo:.3f}, {hi:.3f}]")

# ── Design effect & effective N ─────────────────────────────────────────
# Kish (1965): DEFF = 1 + CV²(w)  where CV = std/mean
# Effective N = N / DEFF

deff    = 1 + (w.std() / w.mean()) ** 2
eff_n   = round(N / deff, 1)

print(f"\nWeight diagnostics:")
print(f"  Kish DEFF:     {deff:.3f}")
print(f"  Effective N:   {eff_n:,.0f}  (from N={N:,})")
print(f"  Weight range:  [{w.min():.3f}, {w.max():.3f}]")

# ── Calibration check ────────────────────────────────────────────────────

print("\nCalibration (weighted % vs. target):")
all_ok = True
for var, cats in target.items():
    for cat, tgt in cats.items():
        wt_pct  = df.loc[df[var] == cat, "weight"].sum() / w.sum()
        raw_pct = (df[var] == cat).mean()
        ok      = abs(wt_pct - tgt) < 0.005
        all_ok &= ok
        print(f"  {var:8s} {cat:20s}  raw={raw_pct:.3f}  "
              f"wtd={wt_pct:.3f}  tgt={tgt:.3f}  {'✓' if ok else '✗'}")

# ── Headline result ──────────────────────────────────────────────────────

raw_supp  = df["q1_support_binary"].mean()
wtd_supp  = np.average(df["q1_support_binary"], weights=w)
true_supp = df["support_prob"].mean()

se  = np.sqrt(wtd_supp * (1 - wtd_supp) / eff_n)
moe = 1.96 * se

print(f"\nPolicy support rates:")
print(f"  Unweighted:      {raw_supp:.1%}")
print(f"  Weighted:        {wtd_supp:.1%}  ± {moe:.1%}  (95% CI)")
print(f"  True population: {true_supp:.1%}  ← ground truth")
print(f"  Bias corrected:  {wtd_supp - raw_supp:+.1%}")

# ── Subgroup estimates ────────────────────────────────────────────────────

print("\nWeighted support by subgroup:")
for var in ["age", "educ", "income", "region"]:
    print(f"\n  {var}:")
    for cat in df[var].dropna().unique():
        mask    = df[var] == cat
        wt_sup  = np.average(df.loc[mask, "q1_support_binary"],
                             weights=df.loc[mask, "weight"])
        n_sub   = mask.sum()
        n_eff   = n_sub / deff
        se_sub  = np.sqrt(wt_sup * (1 - wt_sup) / max(n_eff, 1))
        print(f"    {cat:22s}  {wt_sup:.1%} ± {1.96*se_sub:.1%}  (n={n_sub})")

# ── Save ─────────────────────────────────────────────────────────────────

df.to_csv("data/weighted_survey.csv", index=False)

stats = {
    "n": N, "iters": iters, "n_trimmed": n_trimmed,
    "trim_range": [round(lo,4), round(hi,4)],
    "deff": round(deff,4), "eff_n": eff_n,
    "weight_min": round(float(w.min()),4),
    "weight_max": round(float(w.max()),4),
    "weight_std": round(float(w.std()),4),
    "support": {
        "unweighted": round(float(raw_supp),4),
        "weighted":   round(float(wtd_supp),4),
        "true_pop":   round(float(true_supp),4),
        "moe_95":     round(float(moe),4),
    }
}
with open("data/weight_stats.json","w") as f:
    json.dump(stats, f, indent=2)

print("\n✓ data/weighted_survey.csv")
print("✓ data/weight_stats.json")
