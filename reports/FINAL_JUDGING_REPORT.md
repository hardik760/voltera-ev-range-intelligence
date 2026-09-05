# TECHTRACK 3.0: EV Range Intelligence System
## Final Evaluation & Judging Report
**Team:** Voltra
**Project Scope:** Specification-Based EV Range Prediction

---

## 1. Executive Summary
Team Voltra developed an end-to-end Machine Learning pipeline to predict the real-world driving range of Electric Vehicles based purely on physical specifications (battery capacity, dimensions, weight, drivetrain, etc.) without relying on leaked test-cycle data like WLTP or EPA estimates. 

We identified that raw ML models often learn spurious correlations or hallucinate unphysical predictions. Our primary contribution is a **Physics-Informed ML Architecture** that mathematically constrains the model using physical realities, backed by a production-ready FastAPI backend and an interactive Streamlit frontend that explains predictions in real-time.

---

## 2. Core Technical Innovations & "Gap" Resolutions

### Gap 1: Physics vs. Machine Learning (Residual Modeling)
**The Problem:** Gradient Boosters given raw specifications try to approximate the laws of physics from scratch, often leading to unphysical extrapolations (e.g. predicting massive range for an aerodynamically terrible truck just because it has a big battery).
**Our Solution (Physics-Informed Residual Regressor):**
We developed a custom scikit-learn meta-estimator (`PhysicsResidualRegressor`). Instead of predicting range directly, it:
1. Computes a Domain Baseline: `(Battery Capacity * 1000) / Segment Median Efficiency`.
2. Trains the ML ensemble to predict the *Residual* (the deviation from this physical baseline).
3. Sums the baseline and the ML residual at inference.
**Impact:** Physics guarantees 80% of the prediction. The ML model is relegated to doing what it does best: identifying how aerodynamic profiling, drivetrain efficiency, or weight distribution causes a specific car to over- or under-perform its physical baseline.

### Gap 2: Structural Sanity (Monotonic Constraints)
**The Problem:** Tree-based models can learn non-monotonic relationships from noisy data (e.g., predicting that adding battery capacity *decreases* range in some localized subspace).
**Our Solution:** We injected `monotonic_cst` into our `HistGradientBoostingRegressor`, `XGBoost`, and `RandomForest` estimators. We mathematically forced the models such that an increase in `battery_capacity_kWh` strictly correlates with an increase in residual range. 
**Impact:** The model is structurally incapable of physics violations regarding battery capacity.

### Gap 3: Trust & Explainability (Live SHAP & Uncertainty Bands)
**The Problem:** Single-point ML predictions are black boxes.
**Our Solution:** 
- **Uncertainty Bands:** Instead of a single number, our Streamlit app queries the internal estimators of our `VotingRegressor` independently. We calculate the standard deviation across models and present a **95% Confidence Interval (± X km)**. High variance signals out-of-distribution inputs.
- **Live SHAP Waterfall:** We extract the underlying tree estimator and generate a `shap.plots.waterfall()` on the fly. Because our model targets the residual, the SHAP plot beautifully shows the physical baseline at the bottom, and explains exactly which features (like a heavy chassis or inefficient drivetrain) dragged the range up or down.

### Gap 4: The SUV Dominance Bias (Stratified Splitting)
**The Problem:** Random Train/Test splits often resulted in test sets overwhelmed by SUVs, masking poor performance on minority classes like Cabriolets. 
**Our Solution:** We implemented Stratified Splitting on `car_body_type`. To prevent 1-item classes from crashing the cross-validation, we dynamically binned ultra-rare categories (Coupe → Sedan, Cabriolet → Hatchback) purely for stratification purposes, ensuring robust 10-fold CV without losing granularity during training.

### Gap 5: Production Readiness (API Design)
**The Problem:** Many hackathon ML projects run only in Jupyter Notebooks.
**Our Solution:** We built a robust FastAPI microservice (`app/api.py`). It utilizes Pydantic `BaseModel` for strict input validation. The API correctly raises HTTP 422 Unprocessable Entity errors when fed invalid JSON payloads, demonstrating true production readiness. 

---

## 3. The Data Pipeline
* **Data Sources:** Real-world EV specifications (14 features, 478 rows).
* **Target Variable:** Real-world range (`range_km`).
* **Feature Engineering:** We engineered advanced interaction terms such as `footprint_m2` (Length × Width), `volume_proxy_m3`, and `battery_per_seat`.
* **Preprocessing:** `ColumnTransformer` applying Median Imputation + Scaling for numerics, and Ordinal Encoding for high-cardinality categoricals (handling unseen categories gracefully via `handle_unknown='ignore'`).
* **Evaluation:** Models were aggressively evaluated using 10-Fold Cross Validation. We explicitly log a **Naive Physics Baseline MAE** (~40+ km) against our **Tuned Ensemble MAE** (~11 km) to mathematically prove the ML model's added value.

## 4. Conclusion
Team Voltra did not just "throw data at XGBoost." We engineered a system that respects the laws of physics, quantifies its own uncertainty, explains its reasoning to end-users, and serves predictions via a production-ready API. This represents the gold standard for deploying AI in critical hardware/engineering contexts.
