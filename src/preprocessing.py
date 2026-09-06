"""
TECHTRACK 3.0 — Preprocessing Module

Builds sklearn Pipelines and ColumnTransformers for reproducible
preprocessing. All transformers are fitted on training data only.

LEAKAGE POLICY: No reference to efficiency_wh_per_km anywhere in this module.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder,
)
from sklearn.impute import SimpleImputer


# ---------------------------------------------------------------------------
# Feature sets — derived from the feature audit
# ---------------------------------------------------------------------------

# Core numeric predictors (always included)
NUMERIC_FEATURES_CORE = [
    "battery_capacity_kWh",
    "top_speed_kmh",
    "torque_nm",
    "acceleration_0_100_s",
    "fast_charging_power_kw_dc",
    "towing_capacity_kg",
    "cargo_volume_l",
    "seats",
    "length_mm",
    "width_mm",
    "height_mm",
]

# Engineered numeric features
ENGINEERED_FEATURES = [
    "battery_per_seat",
    "footprint_m2",
    "volume_proxy_m3",
    "battery_per_volume",
    "battery_per_footprint",
    "torque_per_seat",
    "torque_per_volume",
    "aspect_ratio",
    "height_ratio",
    "charging_per_battery",
    "battery_per_towing",
]

# Categorical predictors
CATEGORICAL_FEATURES = [
    "drivetrain",
    "segment",
    "car_body_type",
]

# Known categories for each categorical (for handling unseen values)
KNOWN_CATEGORIES = {
    "drivetrain": ["AWD", "FWD", "RWD"],
    "segment": [
        "A - Mini", "B - Compact", "C - Medium", "D - Large",
        "E - Executive", "F - Luxury", "G - Sports", "I - Luxury",
        "JA - Mini", "JB - Compact", "JC - Medium", "JD - Large",
        "JE - Executive", "JF - Luxury", "N - Passenger Van",
    ],
    "car_body_type": [
        "Cabriolet", "Coupe", "Hatchback", "Liftback Sedan",
        "SUV", "Sedan", "Small Passenger Van", "Station/Estate",
    ],
}

TARGET = "range_km"


def get_feature_sets():
    """Return the feature set configurations for ablation experiments."""
    base_numeric = NUMERIC_FEATURES_CORE.copy()
    engineered = ENGINEERED_FEATURES.copy()
    categorical = CATEGORICAL_FEATURES.copy()

    return {
        "raw_specs_only": {
            "numeric": base_numeric,
            "categorical": categorical,
            "description": "Raw specification columns only",
        },
        "specs_plus_engineered": {
            "numeric": base_numeric + engineered,
            "categorical": categorical,
            "description": "Specifications + all engineered features",
        },
        "specs_engineered_brand": {
            "numeric": base_numeric + engineered,
            "categorical": categorical + ["brand"],
            "description": "Specifications + engineered + brand encoding",
        },
    }


def build_preprocessor(
    numeric_features: list,
    categorical_features: list,
    scale: bool = False,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - Imputes missing numeric values with median
      - Optionally scales numeric features (for linear models)
      - Encodes categoricals with OrdinalEncoder (handles unknown as -1)
    """
    # Numeric pipeline
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",  # Drop any columns not explicitly listed
        verbose_feature_names_out=False,
    )

    preprocessor.set_output(transform="pandas")
    return preprocessor


def build_full_pipeline(model, numeric_features, categorical_features, scale=False):
    """
    Build a complete preprocessing + model pipeline.

    Returns a standard sklearn Pipeline: preprocessor → model.
    NO target leakage wrapper. The model predicts range_km directly.
    """
    preprocessor = build_preprocessor(numeric_features, categorical_features, scale)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return pipeline


def get_feature_names_from_pipeline(pipeline) -> list:
    """Extract feature names from a fitted pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback: reconstruct from transformer specification
        names = []
        for name, trans, cols in preprocessor.transformers_:
            if name != "remainder":
                names.extend(cols)
        return names


def prepare_input_for_pipeline(
    input_dict: dict,
    numeric_features: list,
    categorical_features: list,
) -> pd.DataFrame:
    """
    Convert a user-input dictionary to a DataFrame suitable for the pipeline.
    Handles missing values by setting them to NaN (the pipeline's imputer will handle them).
    """
    all_features = numeric_features + categorical_features
    row = {}
    for feat in all_features:
        val = input_dict.get(feat, np.nan)
        if val is None or (isinstance(val, str) and val.strip() == ""):
            val = np.nan
        row[feat] = val

    df = pd.DataFrame([row])

    # Ensure numeric columns are float
    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
