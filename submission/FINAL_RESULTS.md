# TECHTRACK 3.0 — Final Results
**Team:** Voltra
**Event:** MANIT Bhopal EV Day 2026

## 1. Goal
Predict the official driving range (`range_km`) of 2025 electric vehicles based solely on static specifications. 

## 2. Integrity Guarantee
We completely eliminated the algebraic target leakage present in naive models by strictly dropping `efficiency_wh_per_km`. We proved our predictions are valid by using efficiency *only* as a post-prediction physics sanity check.

## 3. Results
- **Test Set (n=96, 20% holdout):**
  - **Mean Absolute Error:** 9.96 km
  - **R-squared:** 0.9834
  - **Mean Absolute Percentage Error:** 2.53%
- **Physical Plausibility:** 100% of our predictions imply a realistic energy efficiency (post-prediction check).

## 4. Winning Features
Our domain-engineered features out-predicted raw specifications. The top 3 most important drivers of EV range according to our model's SHAP values:
1. `battery_per_volume`: Energy density proxy.
2. `battery_per_seat`: Utility/passenger load proxy.
3. `height_ratio`: Aerodynamic profile proxy.

## 5. Model Architecture
- **Preprocessing:** Median imputation for numerics, Frequent imputation + One-Hot for categoricals.
- **Model:** A robust **Voting Ensemble** of Gradient Boosting (`GradientBoostingRegressor`), Histogram-based Gradient Boosting (`HistGradientBoostingRegressor`), and LightGBM (`LGBMRegressor`).
- **Tuning:** Extensive hyperparameter search using 10-fold cross-validation.
