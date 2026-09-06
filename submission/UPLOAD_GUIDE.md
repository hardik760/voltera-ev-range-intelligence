# Submission Upload Guide

## TECHTRACK 3.0 — Team Voltra

---

## Quick Start for Judges

```bash
# 1. Extract the submission
unzip TECHTRACK3_VOLTERA_FINAL.zip
cd Voltra---ML-project-

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline (reproduces all results)
python run_pipeline.py

# 5. Run automated tests (38 tests)
pip install pytest
python -m pytest tests/ -v

# 6. Run submission audit (39 checks)
python scripts/final_audit.py

# 7. Launch interactive demo
streamlit run app/app.py

# 8. Launch API (optional)
uvicorn app.api:app --reload
```

---

## File Overview

| File/Directory | Purpose |
|---|---|
| `run_pipeline.py` | Complete ML pipeline — single command reproduces everything |
| `app/app.py` | Streamlit interactive demo |
| `app/api.py` | FastAPI REST backend |
| `tests/` | 38 automated tests (contract, leakage, data quality, consistency) |
| `scripts/final_audit.py` | 39-check submission audit |
| `notebooks/` | Jupyter notebook companion |
| `reports/technical_report.md` | Complete technical documentation |
| `JUDGE_QA.md` | 20 anticipated judge questions |
| `FINAL_RUBRIC_AUDIT.md` | Self-assessed rubric compliance |
| `README.md` | Project overview and setup instructions |

---

## Expected Outputs After Pipeline Run

- `models/final_ev_range_pipeline.joblib` — saved pipeline artifact
- `models/pipeline_metadata.json` — feature/model metadata
- `outputs/figures/` — 15 EDA and evaluation plots
- `outputs/metrics/` — all metric files (JSON/CSV)
- `outputs/predictions/` — test predictions and residuals
- `data/processed/ev_data_cleaned.csv` — cleaned dataset

---

## System Requirements

- Python 3.10+
- ~2 GB RAM
- Pipeline execution time: ~3 minutes
- No GPU required
