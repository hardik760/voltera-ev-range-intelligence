# TECHTRACK 3.0 — Submission Checklist

## Team Voltra | MANIT Bhopal EV Day 2026

---

### Pre-Submission Verification

- [x] Pipeline runs end-to-end: `python run_pipeline.py`
- [x] No target leakage: `efficiency_wh_per_km` excluded from all model inputs
- [x] Leakage audit passes: `outputs/metrics/leakage_audit.json`
- [x] Model artifact saved: `models/final_ev_range_pipeline.joblib`
- [x] Pipeline metadata saved: `models/pipeline_metadata.json`
- [x] All metrics reproducible with fixed `random_state=42`
- [x] Stratified 80/20 train/test split

### Deliverables

| Item | Status | Location |
|------|--------|----------|
| Full pipeline script | ✅ | `run_pipeline.py` |
| Cleaned dataset | ✅ | `data/processed/ev_data_cleaned.csv` |
| Model artifact | ✅ | `models/final_ev_range_pipeline.joblib` |
| Pipeline metadata | ✅ | `models/pipeline_metadata.json` |
| All EDA figures (15+) | ✅ | `outputs/figures/` |
| Model arena comparison | ✅ | `outputs/metrics/model_arena.csv` |
| Feature ablation results | ✅ | `outputs/metrics/feature_ablation.csv` |
| Tuning results | ✅ | `outputs/metrics/tuning_results.csv` |
| Final metrics | ✅ | `outputs/metrics/final_metrics.json` |
| Error analysis | ✅ | `outputs/predictions/test_residuals.csv` |
| Physics sanity check | ✅ | `outputs/predictions/physics_sanity_check.csv` |
| Feature importance (SHAP) | ✅ | `outputs/metrics/shap_importance.csv` |
| Permutation importance | ✅ | `outputs/metrics/permutation_importance.csv` |
| Feature registry | ✅ | `outputs/metrics/feature_registry.csv` |
| Leakage audit | ✅ | `outputs/metrics/leakage_audit.json` |
| Streamlit demo app | ✅ | `app/app.py` |
| FastAPI backend | ✅ | `app/api.py` |
| Jupyter notebook | ✅ | `notebooks/TECHTRACK3_Winning_Solution.ipynb` |
| README | ✅ | `README.md` |
| Judge Q&A preparation | ✅ | `JUDGE_QA.md` |
| Requirements | ✅ | `requirements.txt` |
| Final audit script | ✅ | `scripts/final_audit.py` |

### Rubric Alignment

| Criterion | Weight | What We Did |
|-----------|--------|-------------|
| Data Cleaning (15%) | ✅ | Banana Box extraction, brand normalisation, towing 0 vs NaN distinction, justified drops |
| EDA (10%) | ✅ | 15+ figures, efficiency leakage analysis with empirical justification |
| Feature Engineering (15%) | ✅ | 11 physics-inspired features, registry, ablation experiment |
| Model Performance (25%) | ✅ | 13 models compared, 3 tuned, 2 ensembles tested, honest CV + holdout reporting |
| Pipeline/Reproducibility (10%) | ✅ | Single-script, fixed seeds, sklearn Pipeline artifact |
| Code Quality/Docs (10%) | ✅ | Modular src/, descriptive naming, inline documentation |
| Technical Report (10%) | ✅ | Honest limitations, documented decisions |
| Interactive Demo (5%) | ✅ | Streamlit + FastAPI, input validation, physics sanity check |
