"""
TECHTRACK 3.0 — Explainability Module

Provides model-agnostic and model-specific explainability tools:
  - Permutation importance
  - SHAP values (for tree-based models)
  - Feature importance extraction
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def compute_permutation_importance(
    model, X, y, feature_names, n_repeats=10, random_state=42
) -> pd.DataFrame:
    """
    Compute permutation importance on the given data.
    Returns a DataFrame sorted by importance.
    """
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="neg_mean_absolute_error",
    )

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": np.round(result.importances_mean, 4),
        "importance_std": np.round(result.importances_std, 4),
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return imp_df


def extract_tree_importance(model, feature_names) -> pd.DataFrame:
    """
    Extract built-in feature importance from tree-based models.
    Works with RandomForest, GradientBoosting, XGBoost, LightGBM, etc.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": np.round(importances, 4),
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return imp_df


def compute_shap_values(model, X, feature_names, max_samples=200):
    """
    Compute SHAP values for tree-based models.
    Returns SHAP explainer and values.
    Falls back gracefully if SHAP is unavailable.
    """
    try:
        import shap

        # Sample for computational efficiency
        if len(X) > max_samples:
            idx = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
            X_sample = X[idx] if isinstance(X, np.ndarray) else X.iloc[idx]
        else:
            X_sample = X

        # Try TreeExplainer first (faster for tree models)
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        except Exception:
            # Fallback to KernelExplainer
            explainer = shap.KernelExplainer(model.predict, X_sample[:50])
            shap_values = explainer.shap_values(X_sample)

        return {
            "explainer": explainer,
            "shap_values": shap_values,
            "X_sample": X_sample,
            "feature_names": feature_names,
            "success": True,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def shap_summary_table(shap_result: dict) -> pd.DataFrame:
    """
    Create a summary table from SHAP values showing mean absolute SHAP
    per feature.
    """
    if not shap_result.get("success"):
        return pd.DataFrame()

    shap_values = shap_result["shap_values"]
    feature_names = shap_result["feature_names"]

    mean_abs = np.abs(shap_values).mean(axis=0)

    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.round(mean_abs, 4),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return df
