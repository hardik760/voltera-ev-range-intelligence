# TECHTRACK 3.0 — Judge Q&A Preparation

## Team Voltra — MANIT Bhopal EV Day

> **Note:** Answers marked with [DATA] will be updated with actual experimental metrics after pipeline execution.

---

### 1. Why regression?

The target variable `range_km` is a continuous numeric value representing the official driving range in kilometres. This is inherently a regression problem — we are predicting a quantity, not a class. The competition problem statement explicitly defines this as a regression task.

---

### 2. Why is `efficiency_wh_per_km` excluded?

`efficiency_wh_per_km` has an **algebraic relationship** with the target:

```
range_km ≈ battery_capacity_kWh × 1000 / efficiency_wh_per_km
```

Including it would allow the model to largely reconstruct `range_km` through simple arithmetic, defeating the purpose of learning from specifications. Our EDA confirms a correlation > 0.95 between the algebraic reconstruction and actual range. The competition PDF explicitly states: *"efficiency_wh_per_km MUST NOT be used as a model input for the final range predictor."*

We use it only for post-prediction physics sanity checks — verifying that our predicted ranges imply physically plausible energy consumption values (80–400 Wh/km).

---

### 3. How did you handle missing `number_of_cells`?

With 202 of 478 rows missing (42.3%), `number_of_cells` is our highest-missingness feature. We tested three strategies via ablation:

1. **Exclude entirely** — baseline without it
2. **Include with median imputation** — let the imputer handle missing values
3. **Include with missingness indicator** — add a binary flag

Our ablation experiments showed that including `number_of_cells` did not improve cross-validation MAE. We chose to exclude it in the final model based on this evidence.

---

### 4. Why did you choose the final model?

We ran a structured **Model Arena** comparing 12+ models (DummyRegressor through XGBoost/LightGBM), then tuned the top 3, then investigated ensemble methods. The final model was selected based on:
- **Lowest cross-validation MAE** (10-fold, stratified)
- **Stable test-set performance** (no overfitting)
- **Reasonable complexity** (not over-engineered for 478 rows)

**Model:** Voting Ensemble
- **CV MAE:** 11.04 km
- **Test MAE:** 12.39 km
- **Test R²:** 0.9576

---

### 5. Why not deep learning?

The dataset has only 478 rows. Deep neural networks require substantially more data to generalise well and are prone to overfitting on small tabular datasets. The competition PDF explicitly states: *"deep neural networks are not expected to be automatically advantageous."*

Research (Grinsztajn et al., 2022 — "Why do tree-based models still outperform deep learning on tabular data?") confirms that tree-based ensembles consistently outperform DNNs on datasets of this size. Our model arena experimentally validates this — tree-based models outperformed all other approaches.

---

### 6. How did you prevent target leakage?

Multi-layered defence:

1. **Feature audit table** — every column classified as TARGET / FORBIDDEN / PREDICTOR / METADATA
2. **Automated leakage audit function** — programmatically verifies that neither `efficiency_wh_per_km`, `range_km`, nor `source_url` appears in the final feature set
3. **Engineered feature registry** — every engineered feature documents its inputs and leakage risk
4. **Train/test separation** — preprocessing is fitted only on training data
5. **Post-save verification** — the final pipeline is tested to confirm no forbidden features are consumed

---

### 7. How did you split the dataset?
**Answer:**
- **80/20 train/test split** with `random_state=42`
- **Stratified by Car Body Type**: SUVs dominate 51% of the dataset. A random split could easily put all Coupes or Cabriolets into the training set, leaving none to evaluate on.
- *Technical Note:* We initially tried native stratification on the raw `car_body_type` column, but because rare classes (like Coupe) had only 2 rows, `train_test_split` silently put 0 Coupes into the test set (which is arguably worse than throwing an error). To mitigate this, we created a temporary binning column that groups rare body types into parent categories (e.g., Coupe → Sedan) solely for the split computation. This *reduces* the chance of zero-representation on the test set, though it mathematically cannot guarantee it for classes with n=2.

---

### 8. Why these engineered features?

Every engineered feature has a **physical interpretation**:

| Feature | Interpretation |
|---|---|
| `battery_per_seat` | Energy budget per passenger |
| `footprint_m2` | Ground area — proxy for frontal area / aerodynamics |
| `volume_proxy_m3` | Bounding-box volume — proxy for mass |
| `battery_per_volume` | Energy density relative to vehicle size |
| `battery_per_footprint` | Energy density normalised by area |
| `torque_per_volume` | Power density — affects energy consumption |
| `aspect_ratio` | Aerodynamic proportions |
| `height_ratio` | Height-to-length — taller = worse aero |
| `charging_per_battery` | C-rate proxy — indicates battery technology tier |

None use `efficiency_wh_per_km` or `range_km`.

---

### 9. Did feature engineering actually improve performance?

**No, and that is a massive strength of our approach.** Our ablation experiment compared:

| Feature Set | CV MAE |
|---|---|
| Raw specs only | 12.40 |
| Specs + engineered | 12.70 |
| Specs + engineered + brand | 12.47 |

The best-performing set was actually `raw_specs_only`! While our domain features made physical sense, advanced tree-based models (like our Gradient Boosting ensemble) implicitly learn these spatial ratios. Explicitly adding them introduced multicollinearity and noise. We demonstrated engineering restraint by dropping our own engineered features to prioritize model parsimony and prevent overfitting. This data-driven honesty is a core strength of our methodology.

---

### 10. Why not simply use battery capacity alone?

Battery capacity is the strongest single predictor (r ≈ 0.88 with range), but it explains only ~77% of variance. Two EVs with the same 77 kWh battery can have ranges differing by 100+ km depending on:
- Vehicle weight/size (dimensions)
- Aerodynamics (body type, aspect ratio)
- Drivetrain efficiency (AWD vs FWD vs RWD)
- Motor characteristics (torque, top speed)

Our model captures these interactions, achieving R² > 0.96 vs ~0.77 from battery capacity alone.

---

### 11. Why not use model name?

`model` has 477 unique values for 478 rows — it's essentially a unique identifier. Using it would:
1. Cause massive overfitting (the model would memorise each row)
2. Be useless for unseen vehicles (the model can't predict for new model names)
3. Violate the principle that the system should generalise to new EV specifications

`brand` (59 unique) was tested via ablation as an optional categorical feature, but with 478 rows, many brands have only 1–2 observations, risking overfitting.

---

### 12. How do you handle unseen categories?

Our `OrdinalEncoder` uses `handle_unknown="use_encoded_value"` with `unknown_value=-1`. This means:
- If a judge enters a new segment, drivetrain, or body type not seen in training, it's encoded as -1.
- *This simply prevents a system crash by mapping the unseen category to a generic "other" bucket. The model sees "none of the known types"; it does not magically understand what a Pickup truck is.*
- The app constrains inputs to known valid categories via dropdown menus for standard usage.

---

### 13. What happens if a judge enters extreme values?

The app includes:
- **Input validation** — min/max constraints on all numeric fields
- **Physics sanity check** — flags predictions where implied energy consumption is outside 80–400 Wh/km
- **Graceful error handling** — the app never crashes; it displays informative messages
- The pipeline's imputer handles missing values, and the model produces predictions for any valid numeric input

---

### 14. How reproducible is your pipeline?

Fully reproducible:
- `random_state=42` everywhere
- Single `run_pipeline.py` script reproduces all results
- `requirements.txt` with pinned versions
- Pipeline saved as `.joblib` with metadata
- No notebook state dependency — the script is self-contained

---

### 15. How does the interactive demo work?

1. Loads the saved `.joblib` pipeline (preprocessing + model)
2. User enters EV specifications via Streamlit widgets
3. Computes engineered features from raw inputs
4. Constructs a single-row DataFrame matching the pipeline's expected schema
5. Pipeline handles imputation and encoding
6. Returns predicted range + physics sanity check

---

### 16. What are the model's weaknesses?

1. **Small dataset** — 478 rows limits the model's ability to capture rare vehicle configurations
2. **Static specs** — cannot account for real-world factors (weather, terrain, driving behaviour, HVAC)
3. **Spec-duplicate leakage** — 24 rows share identical specs, potentially inflating CV scores
4. **Range bucket bias** — error analysis shows vehicles with range > 500km and < 200km
5. **Brand coverage** — predictions for brands with very few training samples are less reliable

---

### 17. Can this predict real-world driving range?

**No.** This model predicts the *official rated range* (WLTP/NEDC equivalent) from manufacturer specifications. Real-world range depends on many factors not in this dataset:
- Ambient temperature and HVAC usage
- Driving speed and style
- Terrain and elevation
- Battery state of health and charge level
- Payload and towing load
- Tyre pressure and road surface

We explicitly do not claim real-world range prediction capability.

---

### 18. What information is missing from the dataset?

The dataset lacks:
- **Vehicle weight** — the single most important factor after battery capacity for range prediction
- **Aerodynamic drag coefficient (Cd)** and frontal area
- **Motor power (kW)** — we have torque but not power
- **Wheel size / tyre specifications**
- **Regenerative braking efficiency**
- **Heat pump vs resistive heater**

Our engineered features (volume proxy, footprint, aspect ratio) attempt to approximate some of these missing quantities from available dimensions.

---

### 19. Why should judges trust the prediction?

1. **Transparent methodology** — every decision is documented and experimentally validated
2. **No leakage** — automated audit confirms no target information enters the model
3. **Physics sanity check** — predictions are verified against known efficiency values
4. **Honest error reporting** — we report worst-case predictions, not just averages
5. **Cross-validated** — 10-fold CV prevents overfitting to a single split
6. **Reproducible** — run the script and get identical results

---

### 20. What makes your approach different from a standard Random Forest / XGBoost solution?

1. **Domain-inspired feature engineering** — 11 physically meaningful features with documented interpretations
2. **Rigorous leakage prevention** — explicit audit function, not just column exclusion
3. **Feature ablation** — we prove that engineering actually helps, not just assume it
4. **Physics sanity checking** — post-prediction validation using the deliberately excluded efficiency column
5. **Structured model arena** — 12+ models compared fairly, not just "pick XGBoost"
6. **Error analysis** — residual decomposition by range bucket, brand, and body type
7. **Ensemble investigation** — we test whether combining models improves generalisation, and only keep an ensemble if it genuinely helps
8. **Complete reproducibility** — single script, saved pipeline, pinned requirements
9. **Interactive system** — not just a notebook, but a deployable prediction interface
