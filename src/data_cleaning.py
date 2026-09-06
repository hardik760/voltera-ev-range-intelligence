"""
TECHTRACK 3.0 — Data Cleaning Module

Handles loading raw EV specification data, identifying data quality issues,
and producing a clean DataFrame ready for feature engineering.

Design decisions documented inline with OBSERVATION → INTERPRETATION → ACTION pattern.
"""

import pandas as pd
import numpy as np
import re
import os


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_FILENAME = "ev_specs_2025.xls"

# Columns that must never be used as model inputs
FORBIDDEN_FEATURES = {"efficiency_wh_per_km", "range_km", "source_url"}

# Columns with zero or near-zero variance (confirmed from data audit)
ZERO_VARIANCE_COLS = {"battery_type", "fast_charge_port"}

# The regression target
TARGET = "range_km"


def load_raw_data(data_dir: str) -> pd.DataFrame:
    """Load the raw XLS file and return an unmodified DataFrame."""
    path = os.path.join(data_dir, RAW_FILENAME)
    df = pd.read_excel(path)
    return df


def audit_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of missing values per column."""
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    audit = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.values,
        "missing_count": missing.values,
        "missing_pct": pct.values,
        "unique_count": df.nunique().values,
    })
    return audit.sort_values("missing_count", ascending=False).reset_index(drop=True)


def clean_cargo_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    OBSERVATION: cargo_volume_l contains 3 'Banana Boxes' entries and 1 NaN.
    INTERPRETATION: 'Banana Boxes' is a non-standard measurement used by
    ev-database.org for cargo-volume of vans/large vehicles. The numeric prefix
    represents a count of standardised boxes, not litres.
    ACTION: Extract the numeric prefix where possible and convert to an
    approximate litre equivalent. For the NaN, impute with median of same
    body type. If no numeric extraction is possible, impute with body-type median.
    """
    df = df.copy()

    # Extract numeric part from 'Banana Boxes' entries
    # A standard banana box is ~72 litres (industry standard)
    BANANA_BOX_LITRES = 72

    def parse_cargo(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float)):
            return float(val)
        val_str = str(val).strip()
        # Try direct numeric conversion
        try:
            return float(val_str)
        except ValueError:
            pass
        # Extract leading number from 'X Banana Boxes'
        match = re.match(r"(\d+)\s+Banana\s+Box", val_str, re.IGNORECASE)
        if match:
            box_count = int(match.group(1))
            return float(box_count * BANANA_BOX_LITRES)
        return np.nan

    df["cargo_volume_l"] = df["cargo_volume_l"].apply(parse_cargo)

    # Impute remaining NaN with body-type median
    if df["cargo_volume_l"].isna().any():
        body_medians = df.groupby("car_body_type")["cargo_volume_l"].transform("median")
        df["cargo_volume_l"] = df["cargo_volume_l"].fillna(body_medians)

    # Final fallback: overall median
    if df["cargo_volume_l"].isna().any():
        df["cargo_volume_l"] = df["cargo_volume_l"].fillna(df["cargo_volume_l"].median())

    return df


def clean_model_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    OBSERVATION: Row 477 (firefly brand) has model = NaN.
    INTERPRETATION: The source URL suggests the model name is 'firefly'.
    ACTION: Fill with brand name as model name.
    """
    df = df.copy()
    mask = df["model"].isna()
    if mask.any():
        df.loc[mask, "model"] = df.loc[mask, "brand"]
    return df


def clean_brand_casing(df: pd.DataFrame) -> pd.DataFrame:
    """
    OBSERVATION: 'firefly' brand is lowercase while all others are capitalised.
    ACTION: Standardise brand casing to title case for consistency.
    """
    df = df.copy()
    df["brand"] = df["brand"].str.strip()
    # Preserve known brand casing (BMW, BYD, DS, MG, GWM, NIO, XPENG, KGM)
    # by only fixing pure-lowercase brands
    lowercase_mask = df["brand"].str.islower()
    df.loc[lowercase_mask, "brand"] = df.loc[lowercase_mask, "brand"].str.title()
    return df


def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that numeric columns are within physically plausible ranges.
    Flag but do not remove — document any issues.
    """
    checks = {
        "top_speed_kmh": (50, 500),
        "battery_capacity_kWh": (5, 250),
        "torque_nm": (50, 2000),
        "range_km": (50, 1000),
        "acceleration_0_100_s": (1.0, 30.0),
        "fast_charging_power_kw_dc": (10, 500),
        "towing_capacity_kg": (0, 5000),  # 0 is valid (cannot tow)
        "seats": (1, 12),
        "length_mm": (2000, 7000),
        "width_mm": (1200, 2500),
        "height_mm": (1000, 2500),
    }
    issues = []
    for col, (lo, hi) in checks.items():
        if col in df.columns:
            vals = df[col].dropna()
            out_of_range = vals[(vals < lo) | (vals > hi)]
            if len(out_of_range) > 0:
                issues.append({
                    "column": col,
                    "count": len(out_of_range),
                    "min_val": out_of_range.min(),
                    "max_val": out_of_range.max(),
                })
    if issues:
        print("⚠ Range validation issues found:")
        for iss in issues:
            print(f"  {iss['column']}: {iss['count']} values outside "
                  f"[{checks[iss['column']][0]}, {checks[iss['column']][1]}] "
                  f"(min={iss['min_val']}, max={iss['max_val']})")
    else:
        print("✓ All numeric columns within plausible ranges.")
    return df


def build_feature_audit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature audit table classifying each column's role.
    Categories:
      A. TARGET
      B. FORBIDDEN/LEAKAGE
      C. IDENTIFIER/METADATA
      D. NUMERIC PREDICTORS
      E. CATEGORICAL PREDICTORS
      F. OPTIONAL FEATURES (high missingness)
      G. EDA-ONLY FEATURES (zero variance, near-zero variance)
    """
    classifications = {
        "brand":                     ("E", "CATEGORICAL PREDICTOR",  "low",  "test with/without via ablation"),
        "model":                     ("C", "IDENTIFIER",            "high", "unique per row — not predictive"),
        "top_speed_kmh":             ("D", "NUMERIC PREDICTOR",     "none", "performance spec"),
        "battery_capacity_kWh":      ("D", "NUMERIC PREDICTOR",     "none", "primary range determinant"),
        "battery_type":              ("G", "EDA-ONLY",              "none", "zero variance (all Lithium-ion)"),
        "number_of_cells":           ("F", "OPTIONAL",              "none", "42% missing — test via ablation"),
        "torque_nm":                 ("D", "NUMERIC PREDICTOR",     "none", "motor spec, 1.5% missing"),
        "efficiency_wh_per_km":      ("B", "FORBIDDEN/LEAKAGE",     "high", "algebraic relationship with target"),
        "range_km":                  ("A", "TARGET",                "n/a",  "regression target"),
        "acceleration_0_100_s":      ("D", "NUMERIC PREDICTOR",     "none", "performance spec"),
        "fast_charging_power_kw_dc": ("D", "NUMERIC PREDICTOR",     "none", "charging spec, 1 missing"),
        "fast_charge_port":          ("G", "EDA-ONLY",              "none", "99.8% CCS — near-zero variance"),
        "towing_capacity_kg":        ("D", "NUMERIC PREDICTOR",     "none", "utility spec, 5.4% missing"),
        "cargo_volume_l":            ("D", "NUMERIC PREDICTOR",     "none", "utility spec, cleaned"),
        "seats":                     ("D", "NUMERIC PREDICTOR",     "none", "utility spec"),
        "drivetrain":                ("E", "CATEGORICAL PREDICTOR", "none", "AWD/FWD/RWD — 3 levels"),
        "segment":                   ("E", "CATEGORICAL PREDICTOR", "none", "market segment — 15 levels"),
        "length_mm":                 ("D", "NUMERIC PREDICTOR",     "none", "dimension"),
        "width_mm":                  ("D", "NUMERIC PREDICTOR",     "none", "dimension"),
        "height_mm":                 ("D", "NUMERIC PREDICTOR",     "none", "dimension"),
        "car_body_type":             ("E", "CATEGORICAL PREDICTOR", "none", "body style — 8 levels"),
        "source_url":                ("C", "IDENTIFIER/METADATA",   "high", "unique per row — metadata"),
    }

    rows = []
    for col in df.columns:
        cat, sem_type, leak_risk, reason = classifications.get(
            col, ("?", "UNKNOWN", "unknown", "not classified")
        )
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isna().sum()),
            "missing_pct": round(df[col].isna().sum() / len(df) * 100, 1),
            "unique_count": int(df[col].nunique()),
            "semantic_type": sem_type,
            "candidate_role": cat,
            "leakage_risk": leak_risk,
            "reason": reason,
        })
    return pd.DataFrame(rows)


def clean_data(data_dir: str) -> pd.DataFrame:
    """
    Master cleaning function. Runs all cleaning steps in order.
    Returns a cleaned DataFrame with documented transformations.
    """
    print("=" * 60)
    print("TECHTRACK 3.0 — Data Cleaning Pipeline")
    print("=" * 60)

    # Load
    df = load_raw_data(data_dir)
    print(f"✓ Loaded raw data: {df.shape[0]} rows × {df.shape[1]} columns")

    # Audit before cleaning
    print("\n--- Missing Value Audit (Pre-Cleaning) ---")
    audit = audit_missing(df)
    print(audit[audit["missing_count"] > 0].to_string(index=False))

    # Generalized Cleaning Rule: Any model with 'Convertible' or 'Cabrio' should be a Cabriolet
    cabrio_mask = df["model"].str.contains("Convertible|Cabrio", case=False, na=False)
    if cabrio_mask.any():
        df.loc[cabrio_mask, "car_body_type"] = "Cabriolet"
        print(f"✓ Applied generic cleaning rule: Standardized {cabrio_mask.sum()} Convertible/Cabrio models to Cabriolet")

    # Clean model name
    df = clean_model_name(df)
    print("\n✓ Cleaned missing model name (firefly)")

    # Clean brand casing
    df = clean_brand_casing(df)
    print("✓ Standardised brand casing")

    # Clean cargo volume
    df = clean_cargo_volume(df)
    print("✓ Cleaned cargo_volume_l (Banana Boxes → litres, NaN imputed)")

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # Validate ranges
    print("\n--- Range Validation ---")
    df = validate_ranges(df)

    # Drop zero-variance, near-zero-variance, and metadata columns
    drop_cols = ["battery_type", "fast_charge_port", "source_url"]
    df = df.drop(columns=drop_cols, errors="ignore")
    print(f"\n✓ Dropped zero-variance/metadata columns: {drop_cols}")

    # Post-cleaning audit
    print(f"\n--- Post-Cleaning Summary ---")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    remaining_missing = df.isnull().sum()
    remaining = remaining_missing[remaining_missing > 0]
    if len(remaining) > 0:
        print(f"  Remaining missing values:")
        for col, cnt in remaining.items():
            print(f"    {col}: {cnt} ({cnt/len(df)*100:.1f}%)")
    else:
        print("  No missing values remain.")

    return df


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    df = clean_data(data_dir)
    print("\n--- Feature Audit Table ---")
    # Reload raw for audit
    raw = load_raw_data(data_dir)
    audit_table = build_feature_audit(raw)
    print(audit_table.to_string(index=False))
