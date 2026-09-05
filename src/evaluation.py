"""
TECHTRACK 3.0 — Evaluation Module

Provides regression evaluation metrics, residual analysis,
error distribution, and physics-aware sanity checks.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)


def regression_metrics(y_true, y_pred, prefix: str = "") -> dict:
    """Compute standard regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)

    # Relative error
    rel_errors = np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), 1)
    mape = np.mean(rel_errors) * 100

    label = f"{prefix}_" if prefix else ""
    return {
        f"{label}MAE": round(mae, 2),
        f"{label}RMSE": round(rmse, 2),
        f"{label}R2": round(r2, 4),
        f"{label}MedianAE": round(medae, 2),
        f"{label}MAPE": round(mape, 2),
    }


def residual_analysis(y_true, y_pred, df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Create a residual analysis DataFrame.
    Includes absolute error, relative error, and residual.
    """
    residuals = y_true - y_pred
    abs_error = np.abs(residuals)
    rel_error = abs_error / np.maximum(np.abs(y_true), 1) * 100

    result = pd.DataFrame({
        "actual_range_km": y_true,
        "predicted_range_km": np.round(y_pred, 1),
        "residual": np.round(residuals, 1),
        "abs_error": np.round(abs_error, 1),
        "rel_error_pct": np.round(rel_error, 1),
    })

    if df is not None and "brand" in df.columns:
        result["brand"] = df["brand"].values
    if df is not None and "model" in df.columns:
        result["model"] = df["model"].values

    return result.sort_values("abs_error", ascending=False).reset_index(drop=True)


def error_distribution_summary(y_true, y_pred) -> dict:
    """Summarise the error distribution."""
    errors = np.abs(y_true - y_pred)
    return {
        "mean_abs_error": round(np.mean(errors), 2),
        "std_abs_error": round(np.std(errors), 2),
        "min_abs_error": round(np.min(errors), 2),
        "p25_abs_error": round(np.percentile(errors, 25), 2),
        "p50_abs_error": round(np.percentile(errors, 50), 2),
        "p75_abs_error": round(np.percentile(errors, 75), 2),
        "p90_abs_error": round(np.percentile(errors, 90), 2),
        "max_abs_error": round(np.max(errors), 2),
        "within_10km": round(np.mean(errors <= 10) * 100, 1),
        "within_25km": round(np.mean(errors <= 25) * 100, 1),
        "within_50km": round(np.mean(errors <= 50) * 100, 1),
    }


def physics_sanity_check(y_pred, battery_capacity, efficiency_actual=None) -> pd.DataFrame:
    """
    Post-prediction physics sanity check.

    Uses predicted range and battery capacity to compute the implied
    energy consumption (Wh/km). Compares against known efficiency if provided.

    This function is ONLY used after prediction — never during model training.
    """
    # Implied consumption: battery_capacity_kWh * 1000 / predicted_range_km
    implied_wh_per_km = (battery_capacity * 1000) / np.maximum(y_pred, 1)

    result = pd.DataFrame({
        "predicted_range_km": np.round(y_pred, 1),
        "battery_kWh": battery_capacity,
        "implied_wh_per_km": np.round(implied_wh_per_km, 1),
    })

    if efficiency_actual is not None:
        result["actual_wh_per_km"] = efficiency_actual
        result["efficiency_error_pct"] = np.round(
            np.abs(implied_wh_per_km - efficiency_actual) / efficiency_actual * 100, 1
        )

    # Flag physically implausible predictions
    # Typical EV: 100-300 Wh/km; extreme: 80-400 Wh/km
    result["plausible"] = (implied_wh_per_km >= 80) & (implied_wh_per_km <= 400)

    return result
