# EV Range Prediction — Edge Case Checklist & 1st Runner-Up Strategy

*Based on direct inspection of the actual dataset (478 rows × 22 cols) + the problem statement PDF*

---

## 0. What I actually found in your dataset (ground truth, not just the spec sheet)

| Column | Issue found | Count |
|---|---|---|
| `number_of_cells` | Missing | 202 / 478 (42.3%) |
| `torque_nm` | Missing | 7 / 478 |
| `fast_charging_power_kw_dc` | Missing | 1 / 478 |
| `fast_charge_port` | Missing | 1 / 478 |
| `towing_capacity_kg` | Missing | 26 / 478 (also **106 rows have a legitimate `0`**, not missing — don't conflate the two) |
| `cargo_volume_l` | Missing | 1 / 478 |
| `cargo_volume_l` | Non-numeric `"10 Banana Boxes"`, `"31 Banana Boxes"`, `"13 Banana Boxes"` | Audi Q6 e-tron, Maxus MIFA 9, Mercedes EQS SUV |
| `model` | Missing | 1 row (`firefly`, brand lowercase — the only brand not Title Case) |
| `battery_type` | Zero variance (100% "Lithium-ion") | confirmed — drop it |
| `fast_charge_port` | Near-zero variance (476 CCS, 1 CHAdeMO) | confirmed — drop or bin as rare |
| `number_of_cells` | Extreme outlier: 7920 cells (Tesla, 95 kWh pack) | plausible (small-cell chemistry), not an error — don't blindly cap it |
| `torque_nm` | Max 1350 Nm (Maserati GranTurismo/GranCabrio Folgore) | real, verify not a data-entry slip vs. 2nd-highest (1340 Nm Porsche Taycan) |
| Duplicates | Full-row or brand+model duplicates | **0** — no dedup needed |
| `efficiency_wh_per_km` correlation to `range_km` | **Only 0.02** | Good — confirms it's genuinely NOT a trivial algebraic leak the way `battery_capacity_kWh` is (0.88 correlation). Still exclude it per the rules, but you now have data to *justify* why in your report. |

**This is real signal for your technical report** — most teams will just say "I dropped columns per instructions." You can say "I verified `efficiency_wh_per_km` has only r=0.02 correlation with range, empirically confirming it doesn't trivially leak the target — the leakage risk is instead through the `battery_capacity_kWh × efficiency = range` algebraic identity, which we tested for and confirmed drives the exclusion." That's the kind of reasoning judges want to hear at EV Day.

---

## 1. Edge Cases Your Pipeline Must Survive

### A. Schema / ingestion edge cases
- [ ] Pipeline doesn't crash if `number_of_cells` is entirely missing for a new input (it's missing 42% of the time — your final feature set probably shouldn't *require* it, or must have a documented imputation default)
- [ ] Handles the 1 row with missing `model` name without erroring the whole load
- [ ] Handles inconsistent brand casing (`firefly` vs `Tesla`) — normalize with `.str.title()` or similar before any brand-based feature/grouping
- [ ] `cargo_volume_l` is read as **object/string dtype**, not auto-cast to numeric (pandas will NOT auto-convert this column because of the "Banana Boxes" strings) — your cleaning step must explicitly extract the leading number via regex (`str.extract(r'(\d+)')`) and cast
- [ ] Pipeline explicitly drops `source_url`, `battery_type`, and either drops or properly encodes `fast_charge_port` (near-constant) — and you can articulate *why* each was dropped

### B. Missing-value edge cases
- [ ] `number_of_cells`: since almost half the rows are missing this, test what happens if a judge's live-demo input **doesn't include it at all** — your pipeline should not fail, and ideally should perform nearly as well without it (this is also a strong argument for making it an *optional* engineered feature, exactly as the brief hints)
- [ ] `torque_nm` (7 missing), `fast_charging_power_kw_dc` (1 missing), `towing_capacity_kg` (26 missing): imputer must be **fit on train only** and applied to test/live input — test that a single-row live prediction with a missing field doesn't throw a shape/NaN error
- [ ] Distinguish **true missing** (`NaN`) in `towing_capacity_kg` from **legitimate zero** (106 rows genuinely have 0 kg towing capacity — e.g. small hatchbacks). Do not impute `0` as "missing" or you'll bias the model.

### C. Out-of-distribution / extrapolation edge cases (critical for live judge testing)
Judges *will* try to break your demo on EV Day. Test these yourself first:
- [ ] A **tiny battery** input (e.g. 15 kWh, below your training min of 21.3 kWh) — does the model produce a sane (not negative, not absurd) range?
- [ ] A **huge battery** input (e.g. 150 kWh, above your training max of 118 kWh) — tree-based models (RF/XGBoost/LightGBM) **cannot extrapolate** past training range and will flatline at the max leaf value. Know this limitation and be ready to explain it — or clip/flag out-of-range inputs with a warning message in the UI rather than silently returning a nonsense number.
- [ ] Extreme acceleration values (near your min 2.2s or above max 19.1s)
- [ ] A body type / segment / drivetrain value that **never appeared in training** (e.g. a judge types "Pickup" when your `car_body_type` categories are only the 8 seen: SUV, Sedan, Hatchback, Small Passenger Van, Liftback Sedan, Station/Estate, Cabriolet, Coupe) — your `OneHotEncoder` **must** be set with `handle_unknown='ignore'` or the pipeline will crash on an unseen category
- [ ] Rare categories with very few samples: `segment` has categories with only 1–3 rows (`I - Luxury`: 1, `G - Sports`: 2, `JA - Mini`: 2, `A - Mini`: 3) — these will barely be learned; consider whether to keep, merge, or bucket rare segments into an "Other" category
- [ ] Negative or zero inputs where physically impossible (negative battery kWh, 0 seats, negative dimensions) — add basic input validation in the interactive tester, not just trust the model
- [ ] A prediction that comes out **negative** — since you're not constraining output range, a linear/Lasso model *can* mathematically return a negative range_km for a pathological input combination. Add a floor (e.g. `max(0, prediction)`) as a safety net in the pipeline.

### D. Feature engineering / correctness edge cases
- [ ] Confirm your engineered "footprint" feature (e.g. `length_mm × width_mm`) uses the same units/scale consistently between train and the live single-row input
- [ ] If you engineer a `power-to-weight`-style or `torque/acceleration` ratio feature, guard against **divide-by-zero** (e.g. acceleration extremely close to 0 or unseen combos)
- [ ] If you build a "brand tier" or "brand average range" feature (target encoding by brand), a **brand not seen in training** (a judge inventing a fictional spec) will break target encoding — you need a global-mean fallback
- [ ] Verify no leakage: rerun the correlation check on your *final* feature set, not just visually inspect — confirm `efficiency_wh_per_km` and `source_url` are truly excluded from `X`, not just dropped from a display dataframe while still lingering in the actual training matrix (a classic bug)

### E. Train/test split edge cases
- [ ] With only 478 rows, a plain random 80/20 split leaves ~96 test rows — check that rare `segment`/`car_body_type` categories aren't **entirely absent from either split** (stratify by `segment` or `car_body_type` bucket if possible, or at minimum verify post-split)
- [ ] Fix `random_state` everywhere (split, model, any resampling) so Round 2 demo reproduces Round 1 submission numbers exactly — judges will notice if your live numbers don't match your report

### F. Interactive demo edge cases (5% weight but outsized reputational risk on EV Day)
- [ ] Empty/blank field submitted by a judge
- [ ] Text typed into a numeric field ("fast" instead of a number)
- [ ] Extremely long-tail brand-model text input if it's free text anywhere
- [ ] Demo runs with **no internet / no external files** available at presentation time — bundle the trained model + a small reference CSV so it's fully offline-capable
- [ ] Fresh Python environment: does `pip install -r requirements.txt` + running the app actually work on a machine that isn't yours? (test this on a clean venv before EV Day)

---

## 2. What Separates 1st Runner-Up from "Also Submitted" (mapped to the actual rubric)

| Criterion | Weight | What most teams do | What gets you to the podium |
|---|---|---|---|
| Data Cleaning | 15% | Drop NaNs, done | Document *why* each decision was made (zero towing ≠ missing towing; extract numeric from "Banana Boxes" instead of dropping those 3 rows; justify keeping vs. dropping the 42%-missing `number_of_cells`) |
| EDA | 10% | Decorative histograms | EDA that **directly drives** a later decision — e.g. show the efficiency-vs-range scatter to *justify* the leakage exclusion; show segment-wise range distributions to justify a segment-based feature or stratified split |
| Feature Engineering | 15% | Raw columns + one-hot | Physically-motivated features: power-to-weight proxy, frontal-area proxy (width×height), battery energy density (kWh/weight or kWh/footprint), a `has_cell_count_data` missingness indicator flag (turns missingness itself into signal rather than just imputing it away) |
| Model Performance | **25% (biggest single weight)** | One model, default params | Compare ≥4 models (Linear/Ridge baseline → RF → XGBoost/LightGBM → maybe a stacked ensemble), tune the winner with `RandomizedSearchCV`/`Optuna`, report MAE/RMSE/**R²** with cross-validation (not just one train/test split — with only 478 rows, k-fold CV gives more trustworthy numbers than a single split and judges will ask about this) |
| Pipeline/Reproducibility | 10% | Saved `.pkl` of just the model | Full `sklearn.Pipeline` (imputer → encoder → scaler → model) saved as one artifact so a single `pipeline.predict(new_row)` call does everything — this also directly de-risks the live-demo edge cases above |
| Code Quality/Docs | 10% | Comments repeating code | Descriptive names (the brief literally names this), a short markdown cell before each major step explaining the *why*, fixed seeds everywhere |
| Technical Report | 10% | Summary of what was done | Explicitly state **limitations honestly** (small dataset, extrapolation limits, `number_of_cells` missingness, static-spec vs. real-world caveat already flagged in the brief) — judges are told to reward honesty about limitations, so hiding weaknesses actively costs points |
| Interactive Demo | 5% | Static notebook widget | A Gradio/Streamlit app that **doesn't break** on the edge cases in section C/F above, with basic input validation and a sensible message ("input outside training range — prediction may be less reliable") rather than crashing or returning nonsense |

### The single highest-leverage move
Since **Model Performance is 25%** but the dataset is small (478 rows), the biggest risk for every team is overfitting and reporting an inflated single-split R². Use **k-fold cross-validation** (5-fold, stratified by segment if feasible) as your primary reported metric, and show the train-vs-CV gap explicitly — this single choice signals rigor to judges more than squeezing out an extra 0.01 R².

### A believable outcome
With good preprocessing, honest CV, and gradient boosting (XGBoost/LightGBM) tuned reasonably, expect roughly **R² in the 0.85–0.93 range, MAE roughly 20–35 km** on held-out data given `battery_capacity_kWh` alone already correlates 0.88 with `range_km`. Don't chase R²>0.97 — on a 478-row dataset that's a near-guaranteed sign of leakage or overfitting, and a sharp judge will ask about it.

---

## 3. Concrete test rows to run through your final interactive demo before EV Day

| Test case | Example input | Expected robust behavior |
|---|---|---|
| Normal case | Battery 60 kWh, RWD, SUV, JC-Medium | Sane prediction, roughly 350–420 km |
| Missing optional field | Same as above but no `number_of_cells` | Still predicts, doesn't crash |
| Unseen category | `car_body_type = "Pickup"` | Handled gracefully (ignored/one-hot zero row), not a crash |
| Below-range battery | Battery = 15 kWh | No crash; ideally a flag that this is below training range |
| Above-range battery | Battery = 150 kWh | No silent absurd extrapolation; flag or clip |
| Zero/blank field | Acceleration left blank | Graceful error message, not a stack trace |
| Rare segment | `segment = "I - Luxury"` (1 training row) | Runs, but be ready to explain in Q&A that confidence is low here |
| Extreme torque | Torque = 1350 Nm (Maserati-level) | Sane high-performance prediction, not negative or wildly high |
