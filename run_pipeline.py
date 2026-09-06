#!/usr/bin/env python3
"""
TECHTRACK 3.0 — Complete ML Pipeline Execution

This script runs the ENTIRE ML workflow end-to-end:
  1. Data loading & cleaning
  2. Feature engineering
  3. Train/test split
  4. EDA figure generation
  5. Model arena (cross-validation comparison)
  6. Hyperparameter tuning
  7. Feature ablation
  8. Ensemble investigation
  9. Final model evaluation
  10. Explainability
  11. Physics sanity check
  12. Save pipeline artifact

All results are saved to outputs/ for the notebook and report.

LEAKAGE POLICY: efficiency_wh_per_km is NEVER used as a model input,
engineered feature, or training target transformation. It appears ONLY in
EDA visualisation and post-prediction physics sanity checking.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_cleaning import clean_data, load_raw_data, build_feature_audit, audit_missing
from src.feature_engineering import (
    engineer_features, get_engineered_feature_names,
    get_feature_registry_df, leakage_audit,
)
from src.preprocessing import (
    NUMERIC_FEATURES_CORE, ENGINEERED_FEATURES, CATEGORICAL_FEATURES,
    TARGET, build_preprocessor, build_full_pipeline, get_feature_sets,
    KNOWN_CATEGORIES,
)
from src.modeling import (
    get_model_candidates, run_cross_validation, get_tuning_params,
    tune_model, build_ensemble, SEED,
)
from src.evaluation import (
    regression_metrics, residual_analysis, error_distribution_summary,
    physics_sanity_check,
)
from src.explainability import (
    compute_permutation_importance, extract_tree_importance,
    compute_shap_values, shap_summary_table,
)

warnings.filterwarnings("ignore")
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROC = os.path.join(PROJECT_ROOT, "data", "processed")
FIG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
METRICS_DIR = os.path.join(PROJECT_ROOT, "outputs", "metrics")
PRED_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

for d in [DATA_PROC, FIG_DIR, METRICS_DIR, PRED_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# Plot styling
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})
sns.set_style("whitegrid")
PALETTE = sns.color_palette("viridis", 8)


def save_fig(name, tight=True):
    """Save current figure to the figures directory."""
    path = os.path.join(FIG_DIR, f"{name}.png")
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [fig] {name}.png saved")


# ============================================================
# STEP 1: DATA LOADING & CLEANING
# ============================================================
print("\n" + "=" * 70)
print("STEP 1: DATA LOADING & CLEANING")
print("=" * 70)

raw_df = load_raw_data(DATA_RAW)
print(f"Raw data loaded: {raw_df.shape}")

# Save data quality audit
missing_summary = audit_missing(raw_df)
missing_summary.to_csv(os.path.join(METRICS_DIR, "missing_value_summary.csv"), index=False)
print("Missing value summary saved.")

# Save feature audit table
audit_table = build_feature_audit(raw_df)
audit_table.to_csv(os.path.join(METRICS_DIR, "feature_audit.csv"), index=False)
print("Feature audit saved.")

# Save data quality audit (combined)
quality_audit = pd.DataFrame({
    "column": raw_df.columns,
    "dtype": raw_df.dtypes.astype(str).values,
    "missing_count": raw_df.isnull().sum().values,
    "missing_pct": (raw_df.isnull().sum() / len(raw_df) * 100).round(2).values,
    "unique_count": raw_df.nunique().values,
    "has_zero_variance": [(raw_df[c].nunique() <= 1) for c in raw_df.columns],
    "sample_values": [str(raw_df[c].dropna().head(3).tolist()) for c in raw_df.columns],
})
quality_audit.to_csv(os.path.join(METRICS_DIR, "data_quality_audit.csv"), index=False)
print("Data quality audit saved.")

# Clean
df = clean_data(DATA_RAW)
df.to_csv(os.path.join(DATA_PROC, "ev_data_cleaned.csv"), index=False)
print(f"Cleaned data saved: {df.shape}")


# ============================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 70)

df = engineer_features(df)
print(f"Engineered features added. Shape: {df.shape}")

# Registry
registry = get_feature_registry_df()
registry.to_csv(os.path.join(METRICS_DIR, "feature_registry.csv"), index=False)
print("Feature registry saved.")

# Leakage audit
all_candidate_features = (
    NUMERIC_FEATURES_CORE + ENGINEERED_FEATURES + CATEGORICAL_FEATURES
)
audit_result = leakage_audit(df, all_candidate_features)
print(f"\nLeakage audit: {'PASSED' if audit_result['passed'] else 'FAILED'}")
for v in audit_result["violations"]:
    print(f"  {v}")
for w in audit_result["warnings"]:
    print(f"  {w}")

# Save leakage audit
with open(os.path.join(METRICS_DIR, "leakage_audit.json"), "w") as f:
    json.dump(audit_result, f, indent=2)


# ============================================================
# STEP 3: EDA FIGURES
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: EDA FIGURES")
print("=" * 70)

# -- Target distribution --
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(df[TARGET], bins=30, color=PALETTE[0], edgecolor="white", alpha=0.85)
axes[0].set_xlabel("Range (km)")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of range_km")
axes[0].axvline(df[TARGET].mean(), color="red", linestyle="--", label=f"Mean: {df[TARGET].mean():.0f} km")
axes[0].axvline(df[TARGET].median(), color="orange", linestyle="--", label=f"Median: {df[TARGET].median():.0f} km")
axes[0].legend()

import scipy.stats as stats
stats.probplot(df[TARGET], dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot of range_km")
save_fig("01_target_distribution")

# -- Correlation heatmap (numeric features + target + efficiency for EDA ONLY) --
# Load raw for efficiency EDA
raw_for_eda = load_raw_data(DATA_RAW)
eda_numeric = NUMERIC_FEATURES_CORE + [TARGET]
eda_cols_present = [c for c in eda_numeric if c in df.columns]
corr = df[eda_cols_present].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax, square=True)
ax.set_title("Correlation Matrix (legitimate features + target)")
save_fig("02_correlation_heatmap")

# -- Battery capacity vs Range (key relationship) --
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df["battery_capacity_kWh"], df[TARGET],
                     c=df["top_speed_kmh"], cmap="viridis", alpha=0.7, s=40)
ax.set_xlabel("Battery Capacity (kWh)")
ax.set_ylabel("Range (km)")
ax.set_title("Battery Capacity vs Range (coloured by Top Speed)")
plt.colorbar(scatter, label="Top Speed (km/h)")
# Add trend line
z = np.polyfit(df["battery_capacity_kWh"], df[TARGET], 1)
p = np.poly1d(z)
x_line = np.linspace(df["battery_capacity_kWh"].min(), df["battery_capacity_kWh"].max(), 100)
ax.plot(x_line, p(x_line), "r--", alpha=0.7, label=f"Trend: {z[0]:.1f}·kWh + {z[1]:.0f}")
ax.legend()
save_fig("03_battery_vs_range")

# -- Categorical vs target --
cat_cols = ["drivetrain", "car_body_type", "segment"]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, col in enumerate(cat_cols):
    order = df.groupby(col)[TARGET].median().sort_values().index
    sns.boxplot(data=df, x=col, y=TARGET, order=order, ax=axes[i], palette="viridis")
    axes[i].set_title(f"Range by {col}")
    axes[i].tick_params(axis="x", rotation=45)
save_fig("04_categorical_vs_range")

# -- Numeric features vs target (scatter matrix) --
key_numerics = ["battery_capacity_kWh", "top_speed_kmh", "torque_nm",
                "acceleration_0_100_s", "fast_charging_power_kw_dc"]
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
for i, col in enumerate(key_numerics):
    axes[i].scatter(df[col], df[TARGET], alpha=0.4, s=20, color=PALETTE[i % len(PALETTE)])
    axes[i].set_xlabel(col.replace("_", "\n"))
    axes[i].set_ylabel("range_km")
    # Show correlation
    valid = df[[col, TARGET]].dropna()
    r = valid[col].corr(valid[TARGET])
    axes[i].set_title(f"r = {r:.3f}")
save_fig("05_numeric_vs_range")

# -- Missing value heatmap --
missing_cols = df.columns[df.isnull().any()]
if len(missing_cols) > 0:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(df[missing_cols].isnull().T, cbar=False, yticklabels=True,
                cmap="YlOrRd", ax=ax)
    ax.set_title("Missing Value Pattern")
    save_fig("06_missing_values")

# -- Engineered features correlation with target --
eng_with_target = ENGINEERED_FEATURES + [TARGET]
eng_present = [c for c in eng_with_target if c in df.columns]
eng_corr = df[eng_present].corr()[TARGET].drop(TARGET).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
eng_corr.plot(kind="barh", color=[PALETTE[0] if v > 0 else PALETTE[3] for v in eng_corr], ax=ax)
ax.set_xlabel("Correlation with range_km")
ax.set_title("Engineered Features — Correlation with Range")
ax.axvline(0, color="black", linewidth=0.5)
save_fig("07_engineered_correlations")

# -- Dimensions analysis --
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
dims = ["length_mm", "width_mm", "height_mm"]
for i, col in enumerate(dims):
    axes[i].scatter(df[col], df[TARGET], alpha=0.4, s=20, color=PALETTE[i+1])
    r = df[[col, TARGET]].dropna().corr().iloc[0, 1]
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("range_km")
    axes[i].set_title(f"{col} vs range (r={r:.3f})")
save_fig("08_dimensions_vs_range")

# -- Efficiency EDA (FORBIDDEN FEATURE ANALYSIS — EDA ONLY) --
# We use the raw data which still has the efficiency column
if "efficiency_wh_per_km" in raw_for_eda.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    eff_vals = raw_for_eda["efficiency_wh_per_km"]
    range_vals = raw_for_eda["range_km"]
    batt_vals = raw_for_eda["battery_capacity_kWh"]
    
    axes[0].scatter(eff_vals, range_vals, alpha=0.5, s=20, color="red")
    r = eff_vals.corr(range_vals)
    axes[0].set_xlabel("efficiency_wh_per_km")
    axes[0].set_ylabel("range_km")
    axes[0].set_title(f"Efficiency vs Range (r={r:.3f})\nEDA ONLY — FORBIDDEN AS MODEL INPUT")
    
    # Show the algebraic relationship: range ≈ battery_capacity * 1000 / efficiency
    computed_range = batt_vals * 1000 / eff_vals
    axes[1].scatter(computed_range, range_vals, alpha=0.5, s=20, color="red")
    r2 = computed_range.corr(range_vals)
    axes[1].set_xlabel("kWh × 1000 / efficiency (computed)")
    axes[1].set_ylabel("actual range_km")
    axes[1].set_title(f"Algebraic reconstruction (r={r2:.3f})\nWHY efficiency is excluded")
    axes[1].plot([100, 700], [100, 700], "k--", alpha=0.5)
    save_fig("09_efficiency_leakage_analysis")

print(f"\nAll EDA figures saved to {FIG_DIR}")


# ============================================================
# STEP 4: TRAIN/TEST SPLIT
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: TRAIN/TEST SPLIT")
print("=" * 70)

y = df[TARGET].values
X = df.copy()

# Group rare body types for stable stratification
stratify_col = df["car_body_type"].replace({
    "Coupe": "Sedan",
    "Cabriolet": "Hatchback",
    "Small Passenger Van": "SUV"
})

# Stratified split using the binned vehicle body type
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=stratify_col,
)

print(f"Development training set: {X_train.shape[0]} rows")
print(f"Holdout test set:         {X_test.shape[0]} rows")
print(f"Train range stats: mean={y_train.mean():.1f}, std={y_train.std():.1f}")
print(f"Test  range stats: mean={y_test.mean():.1f}, std={y_test.std():.1f}")

# Save split info
split_info = {
    "development_train_size": int(X_train.shape[0]),
    "holdout_test_size": int(X_test.shape[0]),
    "final_deployment_fit_size": int(len(df)),
    "train_mean_range": round(float(y_train.mean()), 2),
    "test_mean_range": round(float(y_test.mean()), 2),
    "train_std_range": round(float(y_train.std()), 2),
    "test_std_range": round(float(y_test.std()), 2),
    "random_seed": SEED,
    "split_method": "stratified by car_body_type (rare types grouped)",
    "test_fraction": 0.2,
}
with open(os.path.join(METRICS_DIR, "split_info.json"), "w") as f:
    json.dump(split_info, f, indent=2)


# ============================================================
# STEP 5: MODEL ARENA
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: MODEL ARENA — Cross-Validation Comparison")
print("=" * 70)

feature_set = "specs_plus_engineered"
feat_config = get_feature_sets()[feature_set]
num_feats = feat_config["numeric"]
cat_feats = feat_config["categorical"]

candidates = get_model_candidates()
arena_results = []

cv_folds = KFold(n_splits=10, shuffle=True, random_state=SEED)

for name, (model, needs_scale, complexity) in candidates.items():
    pipeline = build_full_pipeline(model, num_feats, cat_feats, scale=needs_scale)

    try:
        cv_res = run_cross_validation(pipeline, X_train, y_train, cv=cv_folds)

        # Also fit on full train and predict on test for comparison
        pipeline.fit(X_train, y_train)
        test_pred = pipeline.predict(X_test)
        test_metrics = regression_metrics(y_test, test_pred, prefix="test")

        result = {"model": name, "complexity": complexity}
        result.update(cv_res)
        result.update(test_metrics)
        arena_results.append(result)

        print(f"  {name:30s} | CV MAE: {cv_res['cv_MAE']:6.2f} ± {cv_res['cv_MAE_std']:.2f} | "
              f"CV R²: {cv_res['cv_R2']:.4f} | Test MAE: {test_metrics['test_MAE']:6.2f}")
    except Exception as e:
        print(f"  {name:30s} | FAILED: {e}")

arena_df = pd.DataFrame(arena_results).sort_values("cv_MAE")
arena_df.to_csv(os.path.join(METRICS_DIR, "model_arena.csv"), index=False)
print(f"\nModel arena results saved. Best CV MAE: {arena_df.iloc[0]['model']}")

# Arena comparison chart
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
arena_plot = arena_df.sort_values("cv_MAE")
axes[0].barh(arena_plot["model"], arena_plot["cv_MAE"], color=PALETTE[0])
axes[0].set_xlabel("CV MAE (km)")
axes[0].set_title("Model Arena — CV MAE (lower is better)")
axes[0].invert_yaxis()

axes[1].barh(arena_plot["model"], arena_plot["cv_R2"], color=PALETTE[2])
axes[1].set_xlabel("CV R²")
axes[1].set_title("Model Arena — CV R² (higher is better)")
axes[1].invert_yaxis()

axes[2].barh(arena_plot["model"], arena_plot["test_MAE"], color=PALETTE[4])
axes[2].set_xlabel("Test MAE (km)")
axes[2].set_title("Model Arena — Test MAE (lower is better)")
axes[2].invert_yaxis()
save_fig("10_model_arena")


# ============================================================
# STEP 6: FEATURE ABLATION
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: FEATURE ABLATION")
print("=" * 70)

# Pick the best tree model from arena for ablation
best_tree_models = arena_df[
    arena_df["model"].isin(["Random Forest", "Extra Trees", "Gradient Boosting",
                            "HistGradientBoosting", "XGBoost", "LightGBM"])
].head(3)

if len(best_tree_models) > 0:
    ablation_model_name = best_tree_models.iloc[0]["model"]
else:
    ablation_model_name = "Random Forest"

print(f"Using {ablation_model_name} for ablation experiments")

ablation_results = []
feat_sets = get_feature_sets()

for set_name, config in feat_sets.items():
    model_inst, needs_scale, complexity = candidates[ablation_model_name]

    # Create a fresh model instance
    model_cls = type(model_inst)
    model_params = model_inst.get_params()
    fresh_model = model_cls(**model_params)

    pipeline = build_full_pipeline(
        fresh_model, config["numeric"], config["categorical"], scale=needs_scale
    )

    cv_res = run_cross_validation(pipeline, X_train, y_train, cv=cv_folds)
    ablation_results.append({
        "feature_set": set_name,
        "description": config["description"],
        "n_numeric": len(config["numeric"]),
        "n_categorical": len(config["categorical"]),
        **cv_res,
    })
    print(f"  {set_name:30s} | CV MAE: {cv_res['cv_MAE']:6.2f} | CV R²: {cv_res['cv_R2']:.4f}")

# Test with number_of_cells added
num_with_cells = NUMERIC_FEATURES_CORE + ENGINEERED_FEATURES + ["number_of_cells"]
model_inst, needs_scale, complexity = candidates[ablation_model_name]
pipeline_cells = build_full_pipeline(
    type(model_inst)(**model_inst.get_params()), num_with_cells, CATEGORICAL_FEATURES, scale=needs_scale
)
cv_res_cells = run_cross_validation(pipeline_cells, X_train, y_train, cv=cv_folds)
ablation_results.append({
    "feature_set": "specs_engineered_cells",
    "description": "Specs + engineered + number_of_cells",
    "n_numeric": len(num_with_cells),
    "n_categorical": len(CATEGORICAL_FEATURES),
    **cv_res_cells,
})
print(f"  {'specs_engineered_cells':30s} | CV MAE: {cv_res_cells['cv_MAE']:6.2f} | CV R²: {cv_res_cells['cv_R2']:.4f}")

ablation_df = pd.DataFrame(ablation_results).sort_values("cv_MAE")
ablation_df.to_csv(os.path.join(METRICS_DIR, "feature_ablation.csv"), index=False)
print("\nFeature ablation saved.")

# Ablation chart
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(ablation_df["feature_set"], ablation_df["cv_MAE"], color=PALETTE[1])
ax.set_xlabel("CV MAE (km)")
ax.set_title(f"Feature Ablation — {ablation_model_name}")
ax.invert_yaxis()
for i, (_, row) in enumerate(ablation_df.iterrows()):
    ax.text(row["cv_MAE"] + 0.3, i, f"R²={row['cv_R2']:.3f}", va="center", fontsize=9)
save_fig("11_feature_ablation")


# ============================================================
# STEP 7: HYPERPARAMETER TUNING
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: HYPERPARAMETER TUNING")
print("=" * 70)

# Select best feature set from ablation
best_ablation = ablation_df.iloc[0]
best_feat_set_name = best_ablation["feature_set"]
if best_feat_set_name in feat_sets:
    best_feat_config = feat_sets[best_feat_set_name]
elif best_feat_set_name == "specs_engineered_cells":
    best_feat_config = {
        "numeric": num_with_cells,
        "categorical": CATEGORICAL_FEATURES,
    }
else:
    best_feat_config = feat_sets["specs_plus_engineered"]

print(f"Best feature set: {best_feat_set_name}")

# Tune top 3 models
tuning_params = get_tuning_params()
top_models_to_tune = arena_df.head(5)["model"].tolist()
top_models_to_tune = [m for m in top_models_to_tune if m in tuning_params][:3]

tuning_results = {}
for model_name in top_models_to_tune:
    print(f"\n  Tuning {model_name}...")
    model_inst, needs_scale, complexity = candidates[model_name]

    model_cls = type(model_inst)
    fresh_model = model_cls(**model_inst.get_params())

    pipeline = build_full_pipeline(
        fresh_model,
        best_feat_config["numeric"],
        best_feat_config["categorical"],
        scale=needs_scale,
    )

    result = tune_model(
        pipeline, tuning_params[model_name], X_train, y_train,
        n_iter=60, cv=cv_folds,
    )

    tuning_results[model_name] = result
    print(f"    Best CV MAE: {result['best_cv_mae']:.2f}")
    print(f"    Best params: {result['best_params']}")

# Save tuning results
tuning_summary = []
for name, res in tuning_results.items():
    tuning_summary.append({
        "model": name,
        "tuned_cv_mae": res["best_cv_mae"],
        "best_params": str(res["best_params"]),
    })
pd.DataFrame(tuning_summary).to_csv(
    os.path.join(METRICS_DIR, "tuning_results.csv"), index=False
)


# ============================================================
# STEP 8: ENSEMBLE INVESTIGATION
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: ENSEMBLE INVESTIGATION")
print("=" * 70)

# Use the top tuned models as base learners
best_tuned_pipelines = {}
for name, res in tuning_results.items():
    best_tuned_pipelines[name] = res["best_pipeline"]

# Also run CV on each tuned model individually for fair comparison
ensemble_base_cv = {}
for name, pipe in best_tuned_pipelines.items():
    cv_res = run_cross_validation(pipe, X_train, y_train, cv=cv_folds)
    ensemble_base_cv[name] = cv_res
    print(f"  Tuned {name:30s} | CV MAE: {cv_res['cv_MAE']:.2f} | CV R²: {cv_res['cv_R2']:.4f}")

# Voting ensemble — average predictions of tuned pipelines
from sklearn.ensemble import VotingRegressor

voting_estimators = [(name, pipe) for name, pipe in best_tuned_pipelines.items()]
voting_ens = VotingRegressor(estimators=voting_estimators)
try:
    cv_voting = run_cross_validation(voting_ens, X_train, y_train, cv=cv_folds)
    print(f"  {'Voting Ensemble':30s} | CV MAE: {cv_voting['cv_MAE']:.2f} | CV R²: {cv_voting['cv_R2']:.4f}")
except Exception as e:
    print(f"  Voting ensemble failed: {e}")
    cv_voting = None

# Stacking ensemble
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

stacking_ens = StackingRegressor(
    estimators=voting_estimators,
    final_estimator=Ridge(alpha=1.0, random_state=SEED),
    cv=5,
)
try:
    cv_stacking = run_cross_validation(stacking_ens, X_train, y_train, cv=cv_folds)
    print(f"  {'Stacking Ensemble':30s} | CV MAE: {cv_stacking['cv_MAE']:.2f} | CV R²: {cv_stacking['cv_R2']:.4f}")
except Exception as e:
    print(f"  Stacking ensemble failed: {e}")
    cv_stacking = None

# Collect all results for comparison
ensemble_comparison = []
for name, cv_res in ensemble_base_cv.items():
    ensemble_comparison.append({"model": f"Tuned {name}", **cv_res})
if cv_voting:
    ensemble_comparison.append({"model": "Voting Ensemble", **cv_voting})
if cv_stacking:
    ensemble_comparison.append({"model": "Stacking Ensemble", **cv_stacking})

ensemble_df = pd.DataFrame(ensemble_comparison).sort_values("cv_MAE")
ensemble_df.to_csv(os.path.join(METRICS_DIR, "ensemble_comparison.csv"), index=False)
print(f"\nBest ensemble/tuned model: {ensemble_df.iloc[0]['model']} (CV MAE: {ensemble_df.iloc[0]['cv_MAE']:.2f})")


# ============================================================
# STEP 9: FINAL MODEL SELECTION & EVALUATION
# ============================================================
print("\n" + "=" * 70)
print("STEP 9: FINAL MODEL SELECTION & EVALUATION")
print("=" * 70)

# Select the best model (lowest CV MAE from tuned + ensemble)
best_entry = ensemble_df.iloc[0]
best_model_name = best_entry["model"]

if "Voting" in best_model_name:
    final_pipeline = voting_ens
elif "Stacking" in best_model_name:
    final_pipeline = stacking_ens
else:
    # It's a tuned individual model
    clean_name = best_model_name.replace("Tuned ", "")
    final_pipeline = best_tuned_pipelines[clean_name]

print(f"Selected final model: {best_model_name}")

# Fit on full training data
final_pipeline.fit(X_train, y_train)

# Predict on test set
y_test_pred = final_pipeline.predict(X_test)

# Final test metrics
final_test_metrics = regression_metrics(y_test, y_test_pred, prefix="test")
print(f"\n--- FINAL HOLDOUT TEST SET EVALUATION ---")
for k, v in final_test_metrics.items():
    print(f"  {k}: {v}")

# Save final metrics
final_metrics = {
    "model": best_model_name,
    "feature_set": best_feat_set_name,
}
final_metrics.update(best_entry.to_dict())
final_metrics.update(final_test_metrics)

with open(os.path.join(METRICS_DIR, "final_metrics.json"), "w") as f:
    # Convert numpy types to native Python
    clean_metrics = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                     for k, v in final_metrics.items()}
    json.dump(clean_metrics, f, indent=2)


# ============================================================
# STEP 10: ERROR ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("STEP 10: ERROR ANALYSIS")
print("=" * 70)

# Residual analysis
residuals_df = residual_analysis(y_test, y_test_pred, X_test)
residuals_df.to_csv(os.path.join(PRED_DIR, "test_residuals.csv"), index=False)

print("\n--- Worst 10 Predictions ---")
print(residuals_df[["brand", "model", "actual_range_km", "predicted_range_km",
                     "abs_error", "rel_error_pct"]].head(10).to_string(index=False))

# Error distribution
error_dist = error_distribution_summary(y_test, y_test_pred)
print(f"\n--- Error Distribution ---")
for k, v in error_dist.items():
    print(f"  {k}: {v}")

with open(os.path.join(METRICS_DIR, "error_distribution.json"), "w") as f:
    json.dump(error_dist, f, indent=2)

# Residual plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Actual vs predicted
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.6, s=30, color=PALETTE[0])
lims = [min(y_test.min(), y_test_pred.min()) - 20, max(y_test.max(), y_test_pred.max()) + 20]
axes[0, 0].plot(lims, lims, "r--", alpha=0.7)
axes[0, 0].set_xlabel("Actual Range (km)")
axes[0, 0].set_ylabel("Predicted Range (km)")
axes[0, 0].set_title("Actual vs Predicted")

# Residual distribution
residuals = y_test - y_test_pred
axes[0, 1].hist(residuals, bins=20, color=PALETTE[1], edgecolor="white", alpha=0.85)
axes[0, 1].axvline(0, color="red", linestyle="--")
axes[0, 1].set_xlabel("Residual (km)")
axes[0, 1].set_ylabel("Count")
axes[0, 1].set_title("Residual Distribution")

# Residuals vs predicted
axes[1, 0].scatter(y_test_pred, residuals, alpha=0.6, s=30, color=PALETTE[2])
axes[1, 0].axhline(0, color="red", linestyle="--")
axes[1, 0].set_xlabel("Predicted Range (km)")
axes[1, 0].set_ylabel("Residual (km)")
axes[1, 0].set_title("Residuals vs Predicted")

# Absolute error by range bucket
residuals_df["range_bucket"] = pd.cut(residuals_df["actual_range_km"],
                                       bins=[0, 200, 300, 400, 500, 700],
                                       labels=["<200", "200-300", "300-400", "400-500", ">500"])
bucket_errors = residuals_df.groupby("range_bucket", observed=True)["abs_error"].mean()
bucket_errors.plot(kind="bar", color=PALETTE[3], ax=axes[1, 1])
axes[1, 1].set_xlabel("Range Bucket (km)")
axes[1, 1].set_ylabel("Mean Absolute Error (km)")
axes[1, 1].set_title("Error by Range Bucket")
axes[1, 1].tick_params(axis="x", rotation=0)
save_fig("12_error_analysis")


# ============================================================
# STEP 11: EXPLAINABILITY
# ============================================================
print("\n" + "=" * 70)
print("STEP 11: EXPLAINABILITY")
print("=" * 70)

# Get feature names from the pipeline
try:
    if hasattr(final_pipeline, "named_steps") and "preprocessor" in final_pipeline.named_steps:
        preprocessor = final_pipeline.named_steps["preprocessor"]
    elif hasattr(final_pipeline, "estimators_"):
        # For voting/stacking, use first base estimator's preprocessor
        first_pipe = final_pipeline.estimators_[0]
        if hasattr(first_pipe, "named_steps"):
            preprocessor = first_pipe.named_steps["preprocessor"]
        else:
            preprocessor = first_pipe[1].named_steps["preprocessor"]
    feature_names = list(preprocessor.get_feature_names_out())
except Exception:
    feature_names = best_feat_config["numeric"] + best_feat_config["categorical"]

print(f"Feature names ({len(feature_names)}): {feature_names}")

# Permutation importance on test set
print("\n--- Permutation Importance (Test Set) ---")
try:
    # For ensemble models, permutation importance works on the raw input columns
    # The feature_names from the preprocessor may not match X_test dimensions
    # Use column names from X_test that are actually used by the pipeline
    input_feature_names = list(X_test.columns)
    perm_imp = compute_permutation_importance(
        final_pipeline, X_test, y_test, feature_names=input_feature_names, n_repeats=15
    )
    # Filter to only show features with non-trivial importance
    perm_imp = perm_imp[perm_imp["importance_mean"] > 0.01]
    print(perm_imp.head(15).to_string(index=False))
    perm_imp.to_csv(os.path.join(METRICS_DIR, "permutation_importance.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = min(20, len(perm_imp))
    top_imp = perm_imp.head(top_n).sort_values("importance_mean")
    ax.barh(top_imp["feature"], top_imp["importance_mean"],
            xerr=top_imp["importance_std"], color=PALETTE[0], alpha=0.85)
    ax.set_xlabel("Permutation Importance (MAE decrease)")
    ax.set_title("Top Feature Importance — Permutation")
    save_fig("13_permutation_importance")
except Exception as e:
    print(f"  Permutation importance failed: {e}")

# SHAP analysis
print("\n--- SHAP Analysis ---")
try:
    # For ensemble models, try to get the underlying model
    if hasattr(final_pipeline, "named_steps") and "model" in final_pipeline.named_steps:
        model_for_shap = final_pipeline.named_steps["model"]
        preprocessor_for_shap = final_pipeline.named_steps["preprocessor"]
        X_test_transformed = preprocessor_for_shap.transform(X_test)
    elif hasattr(final_pipeline, "estimators_"):
        # For voting/stacking, use first base model for SHAP
        first_est = final_pipeline.estimators_[0]
        if isinstance(first_est, tuple):
            first_est = first_est[1]
        model_for_shap = first_est.named_steps["model"]
        preprocessor_for_shap = first_est.named_steps["preprocessor"]
        X_test_transformed = preprocessor_for_shap.transform(X_test)
        print("  Note: SHAP shows first component model of ensemble. "
              "Permutation importance above reflects the complete model.")
    else:
        raise ValueError("Cannot extract model for SHAP")

    shap_result = compute_shap_values(
        model_for_shap, X_test_transformed, feature_names
    )

    if shap_result["success"]:
        summary = shap_summary_table(shap_result)
        print(summary.head(15).to_string(index=False))
        summary.to_csv(os.path.join(METRICS_DIR, "shap_importance.csv"), index=False)

        # SHAP summary plot
        import shap as shap_lib
        fig, ax = plt.subplots(figsize=(10, 8))
        shap_lib.summary_plot(
            shap_result["shap_values"],
            shap_result["X_sample"],
            feature_names=feature_names,
            show=False,
            max_display=15,
        )
        save_fig("14_shap_summary")
    else:
        print(f"  SHAP failed: {shap_result.get('error', 'unknown')}")
except Exception as e:
    print(f"  SHAP analysis failed: {e}")


# ============================================================
# STEP 12: PHYSICS-AWARE SANITY CHECK
# ============================================================
print("\n" + "=" * 70)
print("STEP 12: PHYSICS-AWARE SANITY CHECK (post-prediction only)")
print("=" * 70)

# Use efficiency_wh_per_km ONLY for post-prediction validation
# Pull from raw data since it was dropped during cleaning
efficiency_test = raw_df.loc[X_test.index, "efficiency_wh_per_km"].values
battery_test = X_test["battery_capacity_kWh"].values

sanity = physics_sanity_check(y_test_pred, battery_test, efficiency_test)
sanity.to_csv(os.path.join(PRED_DIR, "physics_sanity_check.csv"), index=False)

plausible_pct = sanity["plausible"].mean() * 100
print(f"Physically plausible predictions: {plausible_pct:.1f}%")

implausible = sanity[~sanity["plausible"]]
if len(implausible) > 0:
    print(f"Implausible predictions: {len(implausible)}")
    print(implausible.head().to_string(index=False))
else:
    print("All predictions are physically plausible.")

if "efficiency_error_pct" in sanity.columns:
    mean_eff_error = sanity["efficiency_error_pct"].mean()
    print(f"Mean implied efficiency error: {mean_eff_error:.1f}%")

# Sanity check plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(sanity["implied_wh_per_km"], sanity["actual_wh_per_km"],
                alpha=0.6, s=30, color=PALETTE[0])
lims = [80, 400]
axes[0].plot(lims, lims, "r--", alpha=0.7)
axes[0].set_xlabel("Implied Wh/km (from prediction)")
axes[0].set_ylabel("Actual Wh/km")
axes[0].set_title("Physics Sanity — Implied vs Actual Efficiency\n(post-prediction validation only — NOT a model input)")

axes[1].hist(sanity["implied_wh_per_km"], bins=20, alpha=0.6, color=PALETTE[0], label="Implied")
axes[1].hist(sanity["actual_wh_per_km"], bins=20, alpha=0.6, color=PALETTE[3], label="Actual")
axes[1].set_xlabel("Wh/km")
axes[1].set_title("Efficiency Distribution — Implied vs Actual")
axes[1].legend()
save_fig("15_physics_sanity")


# ============================================================
# STEP 13: SAVE FINAL PIPELINE
# ============================================================
print("\n" + "=" * 70)
print("STEP 13: SAVE FINAL PIPELINE")
print("=" * 70)

# Re-fit on ALL data for the final deployment model
y_all = df[TARGET].values
final_pipeline.fit(df, y_all)

pipeline_path = os.path.join(MODEL_DIR, "final_ev_range_pipeline.joblib")
joblib.dump(final_pipeline, pipeline_path)
print(f"Final pipeline saved: {pipeline_path}")
print(f"Pipeline size: {os.path.getsize(pipeline_path) / 1024:.1f} KB")

# Save metadata about what features the pipeline expects
pipeline_meta = {
    "final_model_name": best_model_name,
    "feature_set_name": best_feat_set_name,
    "numeric_features": best_feat_config["numeric"],
    "categorical_features": best_feat_config["categorical"],
    "engineered_features": [f for f in best_feat_config["numeric"] if f in ENGINEERED_FEATURES],
    "target": TARGET,
    "forbidden_features": ["efficiency_wh_per_km", "range_km", "source_url"],
    "development_train_size": int(X_train.shape[0]),
    "holdout_test_size": int(X_test.shape[0]),
    "final_deployment_fit_size": int(len(df)),
    "seed": SEED,
    "validation_strategy": "10-fold CV on training data, single holdout evaluation",
    "test_metrics": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                     for k, v in final_test_metrics.items()},
    "cv_metrics": {
        "cv_MAE": float(best_entry.get("cv_MAE", 0)),
        "cv_MAE_std": float(best_entry.get("cv_MAE_std", 0)),
        "cv_R2": float(best_entry.get("cv_R2", 0)),
        "cv_R2_std": float(best_entry.get("cv_R2_std", 0)),
    },
}
with open(os.path.join(MODEL_DIR, "pipeline_metadata.json"), "w") as f:
    json.dump(pipeline_meta, f, indent=2)

print("Pipeline metadata saved.")

# Quick inference test
print("\n--- Quick Inference Test ---")
test_input = X_test.iloc[[0]]
pred = final_pipeline.predict(test_input)
print(f"  Input: {test_input[['brand', 'model', 'battery_capacity_kWh']].values[0]}")
print(f"  Predicted: {pred[0]:.1f} km")
print(f"  Actual:    {y_test[0]:.1f} km")


# ============================================================
# FINAL VERIFICATION
# ============================================================
print("\n" + "=" * 70)
print("FINAL VERIFICATION CHECKLIST")
print("=" * 70)

checks = {
    "Dataset loads correctly": True,
    "No target leakage (audit passed)": audit_result["passed"],
    "efficiency_wh_per_km not in model features": "efficiency_wh_per_km" not in best_feat_config.get("numeric", []),
    "range_km not in input features": TARGET not in best_feat_config.get("numeric", []),
    "source_url not in features": "source_url" not in best_feat_config.get("numeric", []) + best_feat_config.get("categorical", []),
    "Pipeline saved successfully": os.path.exists(pipeline_path),
    "Pipeline loads successfully": joblib.load(pipeline_path) is not None,
    "Random seed fixed": SEED == 42,
    "All figures generated": len(os.listdir(FIG_DIR)) >= 10,
    "All metrics saved": len(os.listdir(METRICS_DIR)) >= 5,
}

all_passed = True
for check, result in checks.items():
    status = "PASS" if result else "FAIL"
    print(f"  [{status}] {check}")
    if not result:
        all_passed = False

print(f"\n{'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
print("=" * 70)
print("Pipeline execution complete.")
