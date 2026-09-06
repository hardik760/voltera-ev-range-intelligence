# TECHTRACK 3.0 — Final Results

**Team:** Voltera  
**Event:** MANIT Bhopal EV Day 2026

## 1. Goal

Predict the official driving range (`range_km`) of 2025 electric vehicles using only legitimate static vehicle specifications available before prediction.

## 2. Integrity Guarantee

We completely eliminated algebraic target leakage by strictly excluding `efficiency_wh_per_km` from the predictive pipeline.

The feature is used only for exploratory analysis and as a **post-prediction physics sanity check**. It is never used to generate, train, or reconstruct the predicted range.

## 3. Results

- **Test Set:** 96 vehicles (20% holdout)
  - **Mean Absolute Error (MAE):** 9.96 km
  - **R-squared (R²):** 0.9834
  - **Mean Absolute Percentage Error (MAPE):** 2.53%

- **Physical Plausibility:** 100% of predictions passed the post-prediction energy-efficiency sanity check.

## 4. Winning Features

Domain-engineered features improved predictive performance over raw specifications. According to the model's SHAP analysis, the top three feature drivers of predicted EV range were:

1. `battery_per_volume` — Energy-density / packaging proxy.
2. `battery_per_seat` — Battery capacity relative to passenger capacity.
3. `height_ratio` — Vehicle-proportion proxy associated with vehicle form factor.

These features are derived exclusively from legitimate vehicle specifications and do not use `range_km` or `efficiency_wh_per_km`.

## 5. Model Architecture

- **Preprocessing:** Median imputation for numeric features and `OrdinalEncoder` for categorical features, with unseen categories encoded as `-1`.
- **Model:** A robust **Voting Ensemble** combining:
  - `GradientBoostingRegressor`
  - `HistGradientBoostingRegressor`
  - `LGBMRegressor`
- **Tuning:** Hyperparameter optimization using cross-validation, with the final evaluation performed on an untouched 20% holdout test set.
- **Final Feature Set:** `specs_engineered_cells`

## 6. Cross-Validation

- **CV:** 10-fold cross-validation
- **CV MAE:** 11.68 ± 2.47 km
- **CV RMSE:** 17.49 ± 4.57 km
- **CV R²:** 0.9672 ± 0.0195

The final test set was kept separate from model selection and tuning to provide an unbiased final performance estimate.
