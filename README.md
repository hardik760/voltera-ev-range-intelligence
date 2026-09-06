# TECHTRACK 3.0 — EV Range Prediction from Specifications

## Team Voltera | MANIT Bhopal EV Day 2026

---

## Overview

A complete machine-learning solution for predicting electric vehicle driving range (`range_km`) from static EV specifications. Built for the TECHTRACK 3.0 ML Case Battle competition.

**Challenge:** Predict the official driving range of an EV given its battery, performance, dimensional, and categorical specifications — without using `efficiency_wh_per_km` (which would allow algebraic reconstruction of the target).

**Results:**

| Metric | Value |
|---|---|
| Final Model | Voting Ensemble |
| Test MAE | 9.96 km |
| Test R² | 0.9834 |
| Test RMSE | 13.49 km |
| Test MAPE | 2.53% |
| CV MAE | 11.68 ± 2.47 km |
| Physics plausibility | 100% |

---

## Architecture

```
EV Specifications
        │
        ▼
  Data Validation
        │
        ▼
  Preprocessing (Imputation + Encoding)
        │
        ▼
  Domain Feature Engineering
        │
        ▼
  Regression Model (Voting Ensemble)
        │
        ▼
  Predicted Range (km)
        │
        ▼
  Physics Sanity Check
```

---

## Project Structure

```
Voltra---ML-project-/
│
├── data/
│   ├── raw/                    # Original dataset (ev_specs_2025.xls)
│   └── processed/              # Cleaned data (ev_data_cleaned.csv)
│
├── notebooks/
│   └── TECHTRACK3_Winning_Solution.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py        # Data loading, cleaning, validation
│   ├── feature_engineering.py  # Domain-inspired features + leakage audit
│   ├── preprocessing.py        # sklearn pipelines and feature sets
│   ├── modeling.py             # Model arena, tuning, ensembles
│   ├── evaluation.py           # Metrics, residual analysis, sanity checks
│   └── explainability.py       # SHAP, permutation importance
│
├── app/
│   ├── app.py                  # Streamlit interactive demo
│   └── api.py                  # FastAPI REST backend
│
├── models/
│   ├── final_ev_range_pipeline.joblib    # Saved pipeline
│   └── pipeline_metadata.json           # Feature/model metadata
│
├── tests/
│   ├── test_contract.py        # Model contract tests (10 tests)
│   ├── test_leakage.py         # Target leakage tests (9 tests)
│   ├── test_data_quality.py    # Data quality tests (15 tests)
│   └── test_inference_consistency.py   # Inference consistency (4 tests)
│
├── reports/
│   └── technical_report.md     # Full technical report
│
├── outputs/
│   ├── figures/                # All EDA and evaluation plots (15 figures)
│   ├── metrics/                # JSON/CSV metric files
│   └── predictions/            # Test predictions and residuals
│
├── scripts/
│   └── final_audit.py          # Automated submission audit (39 checks)
│
├── submission/                 # Submission packaging
│
├── run_pipeline.py             # End-to-end pipeline execution
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── JUDGE_QA.md                 # 20 anticipated judge questions + answers
└── FINAL_RUBRIC_AUDIT.md       # Self-assessed rubric audit
```

---

## Dataset

| Property | Value |
|---|---|
| **Source** | ev-database.org (2025 specifications) |
| **Records** | 478 EV models |
| **Brands** | 59 manufacturers |
| **Target** | `range_km` (official driving range in km) |
| **Features** | 22 columns (battery, performance, dimensions, drivetrain, body type, etc.) |

**Known data quality issues (by design):**
- `number_of_cells`: 42% missing
- `cargo_volume_l`: 3 "Banana Boxes" entries + 1 NaN
- `towing_capacity_kg`: 26 missing
- `torque_nm`: 7 missing
- `battery_type`: zero variance (all Lithium-ion)
- `fast_charge_port`: near-zero variance (99.6% CCS)
- `source_url`: unique metadata, not predictive

---

## Setup & Installation

```bash
# Clone the repository
git clone <repo-url>
cd Voltra---ML-project-

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Execution

### 1. Run the Full Pipeline

```bash
python run_pipeline.py
```

This executes the complete workflow: data cleaning → feature engineering → EDA → model training → tuning → evaluation → saves the final pipeline to `models/`.

### 2. Run Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

38 automated tests covering: model contracts, leakage prevention, data quality, and inference consistency.

### 3. Interactive Web Demo (Streamlit)

```bash
streamlit run app/app.py
```

Opens at `http://localhost:8501` with:
- Manual EV specification input
- Demo presets (Compact / Mid-size / Premium)
- Dataset EV lookup
- Live SHAP explanation
- What-if sensitivity analysis

### 4. Production API (FastAPI)

```bash
uvicorn app.api:app --reload
```

- Swagger docs: `http://localhost:8000/docs`
- POST `/predict` with JSON specifications

### 5. Run the Notebook

```bash
jupyter notebook notebooks/TECHTRACK3_Winning_Solution.ipynb
```

### 6. Final Audit

```bash
python scripts/final_audit.py
```

39 automated checks verifying submission readiness.

---

## Leakage Policy

**Hard constraint**: `efficiency_wh_per_km` is NEVER used as a model input feature.

- **Reason:** `range ≈ battery_capacity × 1000 / efficiency` — including efficiency allows algebraic target reconstruction.
- **Permitted uses:** EDA correlation analysis, post-prediction physics sanity checks.
- **Automated audit:** The pipeline includes a leakage audit function that verifies no forbidden feature enters the model. Test suite includes 9 leakage-specific tests.

---

## Evaluation

Final model metrics computed on a held-out 20% test set (96 rows, stratified split, seed=42):

| Metric | Value |
|---|---|
| MAE | 9.96 km |
| RMSE | 13.49 km |
| R² | 0.9834 |
| Median AE | 7.81 km |
| MAPE | 2.53% |

Cross-validation (10-fold on training set, 382 rows):

| Metric | Value |
|---|---|
| CV MAE | 11.68 ± 2.47 km |
| CV R² | 0.9672 ± 0.0195 |

See `outputs/metrics/final_metrics.json` for exact values.

---

## Limitations

1. **Static specifications only** — does not account for traffic, weather, HVAC, battery degradation, or driving style.
2. **Dataset size** (478 rows) limits model complexity and generalization to truly novel vehicle designs.
3. **Official range values** may differ from real-world driving range.
4. Predictions are for the official WLTP/NEDC-equivalent range, not trip-specific estimates.
5. Missing vehicle weight and aerodynamic drag coefficient limits accuracy for outlier vehicles.

---

## License

Competition submission — TECHTRACK 3.0 at MANIT Bhopal.
