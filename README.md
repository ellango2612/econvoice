# EconVoice — Economic Anxiety Survey Analytics Pipeline

A production-grade survey data science pipeline demonstrating applied survey research skills: sampling theory, post-stratification weighting, predictive modeling, automated QA, and stakeholder-facing visualization.

Built as a portfolio project targeting survey scientist roles in applied/political analytics.

---

## The Question

> What share of U.S. adults support a federal job guarantee / wage support policy, and what demographic and attitudinal factors predict that support?

---

## Why This Design (interview-ready explanation)

Online opt-in panels are the dominant fieldwork mode in applied survey research today. They're fast and cheap — but they have a well-known bias problem: **younger, higher-income, more educated respondents are systematically over-represented** because they're more likely to be registered in survey panels and to complete questionnaires.

This project replicates that realistic challenge end-to-end:

1. Simulate a population with known "true" support rates  
2. Draw a biased online sample (18–34 year-olds are 2× over-sampled)  
3. Correct the bias via raking, measure how much it helps  
4. Build a predictive model to generate reliable small-area estimates  
5. Run automated QA checks before any results leave the pipeline  

---

## Pipeline

```
econvoice/
├── 01_generate.py         Synthetic population + biased panel sample
├── 02_weight.py           Iterative proportional fitting (raking)
├── 03_model.py            Logistic regression + Random Forest + MRP estimates
├── 04_qa.py               Automated data quality checks
├── dashboard.html         Interactive results dashboard (no dependencies)
└── data/
    ├── raw_survey.csv         1,200 respondents, pre-weighting
    ├── pop_targets.json       Census-approximate marginal targets
    ├── weighted_survey.csv    Calibrated survey weights added
    ├── weight_stats.json      DEFF, effective N, convergence diagnostics
    ├── predictions.csv        Per-respondent predicted probabilities
    ├── region_estimates.csv   MRP-style region × income estimates with CIs
    ├── model_results.json     AUC, Brier, coefficients, feature importances
    └── qa_report.json         Structured pass/fail QA report
```

---

## Step-by-Step

### Step 1 — Data Generation (`01_generate.py`)

Simulates a U.S. adult population (N=30,000) and draws a biased online panel sample (n=1,200).

**Population model:**  
Support probability is assigned via a logistic model with interpretable known coefficients — income is the strongest predictor (OR ≈ 0.77 for high income), followed by age and financial stress. This gives us a ground truth to validate against.

**Sampling bias:**  
Selection probability is weighted to mimic real online panel behavior:
- 18–34 year-olds: 3.5× more likely to be selected
- Bachelor+ respondents: 2× more likely  
- Over $80k income: 2.2× more likely

Result: 18–34s go from 28% of population → 54% of sample. Without correction, this biases the support estimate by about −1.4pp.

**Survey questions:**
- Q1: Support for federal wage/job policy (Likert + binary)
- Q2: Biggest economic worry (multinomial, 5 categories)
- Q3: Personal financial stress (1–5 scale)
- Q4: Confidence in economic improvement (1–5 scale)

**Missingness:** ~7.5% skip the income question — realistic for a sensitive item.

---

### Step 2 — Weighting (`02_weight.py`)

Applies **iterative proportional fitting (raking)** to calibrate the sample to Census marginals on four variables: age, gender, education, region.

**The algorithm in plain English:**  
Go through each weighting variable in turn. For each category (e.g. "age 65+"), scale all respondents in that category up or down so their weighted total matches the population share. Repeat until nothing changes.

**Key outputs:**
| Metric | Value |
|--------|-------|
| Iterations to converge | 7 |
| Kish design effect (DEFF) | 1.63 |
| Effective N | 735 (from n=1,200) |
| Bias corrected | −1.4pp |
| Weighted support | 32.0% ± 3.4pp |
| True population | 34.9% |

The DEFF tells us the precision cost of weighting: our 1,200-person sample effectively behaves like a 735-person simple random sample. The design-effect-adjusted margin of error (±3.4pp) reflects this.

---

### Step 3 — Modeling (`03_model.py`)

Two models trained on weighted survey data, plus MRP-style regional estimates.

**Models:**
| Model | CV AUC | Full AUC | Brier |
|-------|--------|----------|-------|
| Logistic Regression | 0.597 | 0.616 | 0.214 |
| Random Forest | 0.584 | 0.642 | 0.213 |
| Ensemble | — | 0.629 | 0.213 |

AUC of ~0.63 is reasonable for opinion data — individual attitudes are noisy. The value is in population-level estimates, not individual predictions.

**Top predictors (RF importance):**
1. Financial stress (18.2%)
2. Age 50–64 (10.7%)
3. Income > $80k (10.7%)
4. Economic confidence (9.7%)

**MRP-style estimates:**  
The model produces region × income cell estimates with proper CIs, enabling reliable sub-group reporting even for cells where n < 50 (e.g. Northeast × Under $40k, n=18).

---

### Step 4 — Quality Assurance (`04_qa.py`)

Seven automated checks designed to run after every wave delivery:

| Check | Result |
|-------|--------|
| Value range validation | ✓ Pass |
| Raking convergence | ✓ Pass (7 iters) |
| Margin of error vs. target | ✓ Pass (±3.4pp < ±4pp target) |
| Inter-item correlations | ✓ Pass |
| Subgroup cell minimums | ✓ Pass |
| Income missingness (7.5%) | ⚠ Warning |
| DEFF = 1.63 | ⚠ Warning (threshold 1.5) |

**Overall: PASS** — no blocking errors. Warnings are documented for the wave 2 design review.

---

### Dashboard (`dashboard.html`)

Self-contained HTML — no frameworks, no server needed. Open in any browser.

Six panels:
1. Headline result with bias-correction narrative
2. Sampling bias visualization (diverging bar chart)
3. Subgroup estimates (income, age, education, region)
4. Region × income heatmap (MRP estimates with CIs)
5. Model performance and feature importance
6. QA report and calibration verification

---

## How to Run

```bash
pip install pandas numpy scipy scikit-learn

mkdir data
python 01_generate.py
python 02_weight.py
python 03_model.py
python 04_qa.py

open dashboard.html   # or double-click in Finder/Explorer
```

---

## Survey Science Concepts Demonstrated

| Concept | Where |
|---------|-------|
| Online panel sampling bias | `01_generate.py` |
| Iterative proportional fitting (raking) | `02_weight.py` |
| Kish design effect (DEFF) | `02_weight.py` |
| Design-effect-adjusted confidence intervals | `02_weight.py`, `03_model.py` |
| Survey-weighted model training | `03_model.py` |
| MRP-style small-area estimation | `03_model.py` |
| AUC & Brier score for probabilistic models | `03_model.py` |
| Speeder / inconsistency detection | `04_qa.py` |
| Calibration verification | `04_qa.py` |
| Weighted crosstabs with proper SEs | `02_weight.py` |
