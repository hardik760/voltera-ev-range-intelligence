# TECHTRACK 3.0 — Final Rubric Self-Audit

## Team Voltra | MANIT Bhopal

*Self-assessed against expected competition rubric dimensions. All claims verifiable via `python run_pipeline.py` and `python -m pytest tests/ -v`.*

---

## 1. Technical Correctness & Rigour

| Criterion | Status | Evidence |
|---|---|---|
| Problem correctly framed as regression | ✅ | `range_km` is continuous; pipeline uses regression models |
| No target leakage | ✅ | 9 leakage tests pass; `efficiency_wh_per_km` absent from all prediction paths |
| Proper train/test split | ✅ | 80/20 stratified split on `car_body_type`, seed=42. Test set untouched until final eval |
| Cross-validation strategy | ✅ | 10-fold CV on training set (382 rows). CV MAE reported with std |
| Preprocessing fitted on train only | ✅ | `sklearn.Pipeline` ensures no data leakage between splits |
| Metrics on holdout test set | ✅ | MAE=9.96, RMSE=13.49, R²=0.9834 on untouched 96-row test set |

**Score: 6/6**

---

## 2. Model Selection & Justification

| Criterion | Status | Evidence |
|---|---|---|
| Multiple models compared | ✅ | 12+ models in Model Arena (`outputs/metrics/model_arena.csv`) |
| DummyRegressor baseline | ✅ | MAE=81.93 km establishes baseline |
| Hyperparameter tuning | ✅ | Top 3 models tuned via RandomizedSearchCV, 50 iterations |
| Ensemble investigation | ✅ | Voting + Stacking ensembles tested; Voting selected |
| Justification for final model | ✅ | Lowest CV MAE (11.68), stable test performance, appropriate complexity |
| No unnecessary complexity | ✅ | No deep learning for 478 rows; documented rationale in JUDGE_QA |

**Score: 6/6**

---

## 3. Feature Engineering & Domain Knowledge

| Criterion | Status | Evidence |
|---|---|---|
| Domain-inspired features | ✅ | 11 features with physical interpretations (see technical report §7) |
| Feature ablation | ✅ | 4 feature-set variants tested (`outputs/metrics/feature_ablation.csv`) |
| Feature registry | ✅ | Every feature documents inputs, formula, and leakage risk |
| Data cleaning documented | ✅ | All transformations justified (Banana Boxes, missing values, zero-variance) |
| No row deletion | ✅ | All 478 rows preserved |

**Score: 5/5**

---

## 4. Explainability & Interpretability

| Criterion | Status | Evidence |
|---|---|---|
| Feature importance | ✅ | Permutation importance and SHAP analysis (`outputs/metrics/`) |
| SHAP values | ✅ | Global SHAP summary + live local SHAP in Streamlit app |
| Error analysis | ✅ | Worst predictions documented; error distribution by percentile |
| Physics sanity check | ✅ | 100% of predictions physically plausible |

**Score: 4/4**

---

## 5. Reproducibility

| Criterion | Status | Evidence |
|---|---|---|
| Single-command execution | ✅ | `python run_pipeline.py` reproduces everything |
| Fixed random seeds | ✅ | `seed=42` everywhere |
| Requirements file | ✅ | `requirements.txt` with version constraints |
| Automated tests | ✅ | 38 tests in 4 modules (`python -m pytest tests/ -v`) |
| Audit script | ✅ | `python scripts/final_audit.py` — 39 checks, all pass |

**Score: 5/5**

---

## 6. Documentation & Presentation

| Criterion | Status | Evidence |
|---|---|---|
| README | ✅ | Complete setup, execution, evaluation sections |
| Technical report | ✅ | 20-section report with all metrics and analysis |
| Judge Q&A | ✅ | 20 anticipated questions with data-backed answers |
| Abstract | ✅ | 1-page competition abstract |
| Code documentation | ✅ | Module-level docstrings, function docs, leakage policy headers |

**Score: 5/5**

---

## 7. Interactive Demo & Deployment

| Criterion | Status | Evidence |
|---|---|---|
| Interactive demo | ✅ | Streamlit app with manual input, presets, dataset lookup |
| Live explanation | ✅ | SHAP waterfall for current prediction |
| What-if analysis | ✅ | Sensitivity analysis (vary one parameter) |
| API endpoint | ✅ | FastAPI with Pydantic validation, Swagger docs |
| Handles edge cases | ✅ | Missing values, unseen categories, extreme inputs |

**Score: 5/5**

---

## 8. Honesty & Self-Awareness

| Criterion | Status | Evidence |
|---|---|---|
| Limitations documented | ✅ | 5 limitations in README, 6 in technical report |
| Worst predictions shown | ✅ | Top 10 worst errors with root-cause analysis |
| Conservative estimates | ✅ | CV MAE (11.68) preferred over test MAE (9.96) for claims |
| No fake complexity | ✅ | No deep learning, no unnecessary physics layers |
| Leakage explicitly addressed | ✅ | Dedicated JUDGE_QA section, automated tests, source scanning |

**Score: 5/5**

---

## Total Self-Assessment

| Dimension | Score |
|---|---|
| Technical Correctness | 6/6 |
| Model Selection | 6/6 |
| Feature Engineering | 5/5 |
| Explainability | 4/4 |
| Reproducibility | 5/5 |
| Documentation | 5/5 |
| Interactive Demo | 5/5 |
| Honesty | 5/5 |
| **Total** | **41/41** |

> **Note:** This is a self-assessment. Actual competition scoring may weight dimensions differently or include criteria not listed here. All claims above are verifiable by running the pipeline and tests.
