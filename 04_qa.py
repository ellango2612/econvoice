"""
EconVoice Survey Pipeline — Step 4: Quality Assurance
======================================================
Automated checks that would run after every data delivery in production.
Outputs a structured QA report that could trigger alerts or block release.

Checks
------
  1. Response completeness — missingness by variable
  2. Outlier / impossible values
  3. Inter-item consistency (stress ↔ confidence should be negatively correlated)
  4. Weight health — DEFF, effective N, weight range
  5. Cell size minimums — flag sub-groups too small to report
  6. Margin of error vs. target (±4pp at 95%)
  7. Calibration verification — weighted % vs. population targets

Output: data/qa_report.json
"""

import json
import numpy as np
import pandas as pd
from scipy import stats

df    = pd.read_csv("data/weighted_survey.csv")
ws    = json.load(open("data/weight_stats.json"))
tgts  = json.load(open("data/pop_targets.json"))

N   = len(df)
w   = df["weight"].values
passed, warnings, errors = [], [], []

print("=" * 55)
print("EconVoice QA Report")
print("=" * 55)

# ── 1. Missingness ───────────────────────────────────────────────────────

print("\n[1] Missingness")
for col in ["age","gender","educ","region","income",
            "q1_support_binary","q3_financial_stress","q4_econ_confidence"]:
    rate = df[col].isna().mean()
    if rate > 0.15:
        errors.append(f"Critical missingness: {col} = {rate:.1%}")
        print(f"  ✗ {col:25s} {rate:.1%}  CRITICAL")
    elif rate > 0.05:
        warnings.append(f"Elevated missingness: {col} = {rate:.1%}")
        print(f"  ⚠ {col:25s} {rate:.1%}")
    elif rate > 0:
        print(f"  ~ {col:25s} {rate:.1%}")
    else:
        print(f"  ✓ {col:25s} 0%")

# ── 2. Range checks ──────────────────────────────────────────────────────

print("\n[2] Value range checks")
checks = [
    ("q3_financial_stress", 1, 5),
    ("q4_econ_confidence",  1, 5),
    ("q1_support_binary",   0, 1),
]
for col, lo, hi in checks:
    bad = ((df[col] < lo) | (df[col] > hi)).sum()
    if bad:
        errors.append(f"Out-of-range values: {col} n={bad}")
        print(f"  ✗ {col:30s} {bad} out-of-range values")
    else:
        passed.append(f"Range check: {col}")
        print(f"  ✓ {col:30s} all in [{lo},{hi}]")

# ── 3. Inter-item consistency ────────────────────────────────────────────

print("\n[3] Inter-item consistency")

r_stress_supp, p1 = stats.pearsonr(df["q3_financial_stress"], df["q1_support_binary"])
r_conf_stress, p2 = stats.spearmanr(df["q4_econ_confidence"], df["q3_financial_stress"])

print(f"  Financial stress × policy support:  r={r_stress_supp:.3f}  p={p1:.1e}")
print(f"  Econ confidence × financial stress: ρ={r_conf_stress:.3f}  p={p2:.1e}")

if r_stress_supp > 0.05:
    passed.append(f"Stress-support correlation positive (r={r_stress_supp:.3f})")
    print(f"  ✓ Direction expected (higher stress → more support for policy)")
else:
    warnings.append(f"Stress-support correlation unexpectedly low (r={r_stress_supp:.3f})")
    print(f"  ⚠ Stress-support correlation weaker than expected")

if r_conf_stress < -0.05:
    passed.append(f"Confidence-stress correlation negative (ρ={r_conf_stress:.3f})")
    print(f"  ✓ Direction expected (higher confidence → lower stress)")
else:
    warnings.append(f"Confidence-stress correlation unexpected (ρ={r_conf_stress:.3f})")
    print(f"  ⚠ Confidence-stress correlation unexpected")

# ── 4. Weight health ─────────────────────────────────────────────────────

print("\n[4] Weight health")
deff  = ws["deff"]
eff_n = ws["eff_n"]
print(f"  Kish DEFF:     {deff:.3f}  (threshold: 1.5 warn, 2.0 error)")
print(f"  Effective N:   {eff_n:,.0f} / {N}  ({eff_n/N:.0%} efficiency)")
print(f"  Weight range:  [{ws['weight_min']:.3f}, {ws['weight_max']:.3f}]")

if deff > 2.0:
    errors.append(f"DEFF={deff:.2f} exceeds 2.0 — severe weighting inflation")
    print(f"  ✗ DEFF critical — consider quota sampling for next wave")
elif deff > 1.5:
    warnings.append(f"DEFF={deff:.2f} elevated")
    print(f"  ⚠ DEFF elevated — monitor next wave")
else:
    passed.append(f"DEFF={deff:.2f} within acceptable range")
    print(f"  ✓ DEFF acceptable")

# ── 5. Cell size minimums ────────────────────────────────────────────────

print("\n[5] Subgroup cell sizes  (min reportable: n=50)")
MIN_N = 50
for var in ["age","gender","educ","region","income"]:
    for cat, cnt in df[var].value_counts().items():
        if cnt < MIN_N:
            warnings.append(f"Small cell: {var}={cat} n={cnt}")
            print(f"  ⚠ {var}={cat}: n={cnt} — suppress from reporting")

# ── 6. Margin of error ───────────────────────────────────────────────────

print("\n[6] Margin of error")
TARGET_MOE = 0.04
supp  = ws["support"]["weighted"]
moe   = ws["support"]["moe_95"]

print(f"  Target MoE:   ±{TARGET_MOE:.0%}")
print(f"  Achieved MoE: ±{moe:.1%}")
if moe <= TARGET_MOE:
    passed.append(f"MoE ±{moe:.1%} meets target ±{TARGET_MOE:.0%}")
    print(f"  ✓ Target met")
else:
    warnings.append(f"MoE ±{moe:.1%} exceeds target ±{TARGET_MOE:.0%}")
    print(f"  ⚠ MoE exceeds target — consider n=1,500 for wave 2")

# ── 7. Calibration verification ──────────────────────────────────────────

print("\n[7] Calibration verification")
for var, cats in tgts.items():
    for cat, tgt_pct in cats.items():
        wtd = df.loc[df[var]==cat, "weight"].sum() / w.sum()
        err = abs(wtd - tgt_pct)
        ok  = err < 0.005
        if not ok:
            warnings.append(f"Calibration gap: {var}={cat} ({wtd:.3f} vs {tgt_pct:.3f})")
        mark = "✓" if ok else "⚠"
        print(f"  {mark} {var:8s} {cat:22s}  wtd={wtd:.3f}  tgt={tgt_pct:.3f}")

# ── Summary ───────────────────────────────────────────────────────────────

status = "FAIL" if errors else "PASS"
print(f"\n{'='*55}")
print(f"STATUS: {status}")
print(f"  ✓ Passed:   {len(passed)}")
print(f"  ⚠ Warnings: {len(warnings)}")
print(f"  ✗ Errors:   {len(errors)}")

for e in errors:
    print(f"    ✗ {e}")
for w_item in warnings:
    print(f"    ⚠ {w_item}")

report = {
    "status": status,
    "n": N,
    "passed": passed,
    "warnings": warnings,
    "errors": errors,
    "metrics": {
        "deff": deff,
        "eff_n": eff_n,
        "moe_95": round(moe, 4),
        "target_moe": TARGET_MOE,
        "stress_support_r": round(float(r_stress_supp), 4),
    }
}
with open("data/qa_report.json","w") as f:
    json.dump(report, f, indent=2)

print("\n✓ data/qa_report.json")
