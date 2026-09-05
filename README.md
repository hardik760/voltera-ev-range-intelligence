# TECHTRACK 3.0 — EV Range Prediction from Specifications

## Team Voltra | MANIT Bhopal EV Day 2026

---

## Overview

A complete machine-learning solution for predicting electric vehicle driving range (`range_km`) from static EV specifications. Built for the TECHTRACK 3.0 ML Case Battle competition.

**Challenge:** Predict the official driving range of an EV given its battery, performance, dimensional, and categorical specifications — without using `efficiency_wh_per_km` (which would allow algebraic reconstruction of the target).

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
  Regression Model (Ensemble / Tree-Based)
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
techtrack3_ev_range/
│
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned data
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
│   └── app.py                  # Streamlit interactive demo
│
├── models/
│   ├── final_ev_range_pipeline.joblib    # Saved pipeline
│   └── pipeline_metadata.json           # Feature/model metadata
│
├── reports/
│   └── technical_report.md
│
├── outputs/
│   ├── figures/                # All EDA and evaluation plots
│   ├── metrics/                # JSON/CSV metric files
│   └── predictions/            # Test predictions and residuals
│
├── run_pipeline.py             # End-to-end pipeline execution
├── requirements.txt
├── README.md
└── JUDGE_QA.md
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
- `source_url`: unique metadata, not predictive

---

## Setup & Installation

```bash
# Clone the repository
git clone <repo-url>
cd techtrack3_ev_range

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

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

### 3. Interactive Web Demo (Streamlit)
To launch the interactive Streamlit application for manual testing and demonstration:

```bash
streamlit run app/app.py
```
This will open a web interface at `http://localhost:8501`.

### 4. Production API (FastAPI)
To demonstrate production-readiness, we provide a REST API backend:

```bash
# Start the API server
uvicorn app.api:app --reload
```
- Interactive Swagger documentation: `http://localhost:8000/docs`
- Send POST requests to `http://localhost:8000/predict` with JSON specifications to receive range predictions and physics sanity checks.

### 3. Run the Notebook

```bash
jupyter notebook notebooks/TECHTRACK3_Winning_Solution.ipynb
```

---

## Leakage Policy

**Hard constraint**: `efficiency_wh_per_km` is NEVER used as a model input feature.

- **Reason:** `range ≈ battery_capacity × 1000 / efficiency` — including efficiency allows algebraic target reconstruction.
- **Permitted uses:** EDA correlation analysis, post-prediction physics sanity checks.
- **Automated audit:** The pipeline includes a leakage audit function that verifies no forbidden feature enters the model.

---

## Evaluation

Final model metrics are computed on a held-out 20% test set (stratified split, seed=42). Cross-validation uses 10-fold on the training set.

Metrics reported: MAE, RMSE, R², Median AE, MAPE.

See `outputs/metrics/final_metrics.json` for exact values.

---

## Interactive Demo

The Streamlit app (`app/app.py`) provides:
- **Manual input mode**: Enter any EV specifications
- **Demo preset mode**: Select from real dataset entries
- **Instant prediction** with implied energy consumption sanity check
- **Robust to edge cases**: handles missing values, unseen categories, extreme inputs

---

## Limitations

1. **Static specifications only** — does not account for traffic, weather, HVAC, battery degradation, or driving style.
2. **Dataset size** (478 rows) limits model complexity and generalization to truly novel vehicle designs.
3. **Official range values** may differ from real-world driving range.
4. Predictions are for the official WLTP/NEDC-equivalent range, not trip-specific estimates.

---

## License

Competition submission — TECHTRACK 3.0 at MANIT Bhopal.
