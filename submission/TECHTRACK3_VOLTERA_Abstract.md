# TECHTRACK 3.0 — Competition Abstract

## EV Range Intelligence: Specification-Based Driving Range Prediction

**Team Voltra** | MANIT Bhopal EV Day 2026

---

### Problem

Predicting electric vehicle driving range from static manufacturer specifications without using efficiency data (which would allow trivial algebraic target reconstruction).

### Dataset

478 EV models from 59 brands (ev-database.org, 2025). 22 features covering battery, performance, dimensional, and categorical specifications. Target: `range_km` (official driving range).

### Approach

1. **Data Cleaning:** Resolved 3 text values ("Banana Boxes"), 42% missing `number_of_cells`, zero-variance columns, and mixed-type columns. All 478 rows preserved.

2. **Feature Engineering:** 11 domain-inspired features with physical interpretations (battery energy density per vehicle volume, height-to-length ratio as aerodynamic proxy, C-rate proxy, etc.). Every feature audited for leakage risk.

3. **Model Selection:** 12+ models compared via 10-fold cross-validation (Model Arena). Top 3 tuned via RandomizedSearchCV. Voting Ensemble selected as final model.

4. **Leakage Prevention:** Multi-layered defence — automated feature audit, forbidden-feature registry, source-code-level scanning, and 9 dedicated leakage tests.

### Results

| Metric | Holdout Test (n=96) | 10-Fold CV (n=382) |
|---|---|---|
| MAE | 9.96 km | 11.68 ± 2.47 km |
| RMSE | 13.49 km | — |
| R² | 0.9834 | 0.9672 ± 0.0195 |
| MAPE | 2.53% | — |
| Physics plausibility | 100% | — |

### Key Strengths

- **No leakage** — 9 automated tests + source code scanning confirm no forbidden features
- **Reproducible** — single script, fixed seeds, 38 automated tests
- **Explainable** — permutation importance, SHAP values, live local explanations
- **Deployable** — Streamlit demo + FastAPI backend + saved pipeline artifact
- **Honest** — worst predictions documented, limitations clearly stated, conservative CV estimate preferred

### Deliverables

- `run_pipeline.py` — full reproducible pipeline
- `app/app.py` — interactive Streamlit demo with live SHAP
- `app/api.py` — FastAPI REST endpoint
- `tests/` — 38 automated tests (contract, leakage, data quality, consistency)
- `reports/technical_report.md` — complete technical documentation
- `JUDGE_QA.md` — 20 anticipated judge questions with data-backed answers
