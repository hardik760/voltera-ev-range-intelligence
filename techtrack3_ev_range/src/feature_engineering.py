"""
TECHTRACK 3.0 — Feature Engineering Module

Creates domain-inspired engineered features from EV specifications.
Every feature is documented with:
  - formula
  - physical interpretation
  - leakage risk assessment
  - missing-value implications

HARD CONSTRAINT: No feature may use efficiency_wh_per_km or range_km.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Feature Registry — documents every engineered feature
# ---------------------------------------------------------------------------
FEATURE_REGISTRY = [
    {
        "name": "battery_per_seat",
        "formula": "battery_capacity_kWh / seats",
        "interpretation": "Energy available per passenger — higher values suggest"
                          " more range budget per seat",
        "inputs": ["battery_capacity_kWh", "seats"],
        "leakage_risk": "none",
    },
    {
        "name": "footprint_m2",
        "formula": "(length_mm * width_mm) / 1e6",
        "interpretation": "Vehicle ground footprint in m² — proxy for aerodynamic"
                          " frontal area and overall vehicle size",
        "inputs": ["length_mm", "width_mm"],
        "leakage_risk": "none",
    },
    {
        "name": "volume_proxy_m3",
        "formula": "(length_mm * width_mm * height_mm) / 1e9",
        "interpretation": "Approximate bounding-box volume — proxy for vehicle mass"
                          " and air resistance",
        "inputs": ["length_mm", "width_mm", "height_mm"],
        "leakage_risk": "none",
    },
    {
        "name": "battery_per_volume",
        "formula": "battery_capacity_kWh / volume_proxy_m3",
        "interpretation": "Battery energy density relative to vehicle size —"
                          " measures how efficiently the battery pack fills the vehicle",
        "inputs": ["battery_capacity_kWh", "length_mm", "width_mm", "height_mm"],
        "leakage_risk": "none",
    },
    {
        "name": "battery_per_footprint",
        "formula": "battery_capacity_kWh / footprint_m2",
        "interpretation": "Battery capacity normalised by ground footprint —"
                          " captures energy density per unit of vehicle area",
        "inputs": ["battery_capacity_kWh", "length_mm", "width_mm"],
        "leakage_risk": "none",
    },
    {
        "name": "torque_per_seat",
        "formula": "torque_nm / seats",
        "interpretation": "Motor torque per passenger — proxy for per-passenger"
                          " performance capability",
        "inputs": ["torque_nm", "seats"],
        "leakage_risk": "none",
    },
    {
        "name": "torque_per_volume",
        "formula": "torque_nm / volume_proxy_m3",
        "interpretation": "Torque relative to vehicle size — indicates power"
                          " density that affects energy consumption",
        "inputs": ["torque_nm", "length_mm", "width_mm", "height_mm"],
        "leakage_risk": "none",
    },
    {
        "name": "aspect_ratio",
        "formula": "length_mm / width_mm",
        "interpretation": "Length-to-width ratio — captures vehicle proportions"
                          " relevant to aerodynamics",
        "inputs": ["length_mm", "width_mm"],
        "leakage_risk": "none",
    },
    {
        "name": "height_ratio",
        "formula": "height_mm / length_mm",
        "interpretation": "Height-to-length ratio — taller vehicles relative to"
                          " length have worse aerodynamics",
        "inputs": ["height_mm", "length_mm"],
        "leakage_risk": "none",
    },
    {
        "name": "charging_per_battery",
        "formula": "fast_charging_power_kw_dc / battery_capacity_kWh",
        "interpretation": "Fast-charge C-rate proxy — higher values indicate the"
                          " battery can accept charge faster relative to its size,"
                          " which often correlates with premium battery technology",
        "inputs": ["fast_charging_power_kw_dc", "battery_capacity_kWh"],
        "leakage_risk": "none",
    },
    {
        "name": "battery_per_towing",
        "formula": "battery_capacity_kWh / (towing_capacity_kg + 1)",
        "interpretation": "Battery capacity relative to towing ability — vehicles"
                          " designed for towing need more energy per km",
        "inputs": ["battery_capacity_kWh", "towing_capacity_kg"],
        "leakage_risk": "none",
    },
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all engineered features. Returns DataFrame with new columns added.
    Does NOT modify original columns.
    """
    df = df.copy()

    # Fundamental derived dimensions (needed by several features)
    df["footprint_m2"] = (df["length_mm"] * df["width_mm"]) / 1e6
    df["volume_proxy_m3"] = (
        df["length_mm"] * df["width_mm"] * df["height_mm"]
    ) / 1e9

    # Battery-related ratios
    df["battery_per_seat"] = df["battery_capacity_kWh"] / df["seats"]
    df["battery_per_volume"] = df["battery_capacity_kWh"] / df["volume_proxy_m3"]
    df["battery_per_footprint"] = df["battery_capacity_kWh"] / df["footprint_m2"]

    # Torque-related
    df["torque_per_seat"] = df["torque_nm"] / df["seats"]
    df["torque_per_volume"] = df["torque_nm"] / df["volume_proxy_m3"]

    # Geometric ratios
    df["aspect_ratio"] = df["length_mm"] / df["width_mm"]
    df["height_ratio"] = df["height_mm"] / df["length_mm"]

    # Charging ratio
    df["charging_per_battery"] = (
        df["fast_charging_power_kw_dc"] / df["battery_capacity_kWh"]
    )

    # Battery vs towing (add 1 to avoid division by zero for non-towing vehicles)
    df["battery_per_towing"] = (
        df["battery_capacity_kWh"] / (df["towing_capacity_kg"] + 1)
    )

    return df


def get_engineered_feature_names() -> list:
    """Return list of all engineered feature names."""
    return [f["name"] for f in FEATURE_REGISTRY]


def get_feature_registry_df() -> pd.DataFrame:
    """Return the feature registry as a DataFrame for documentation."""
    return pd.DataFrame(FEATURE_REGISTRY)


def leakage_audit(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Automated leakage audit.
    Checks that no forbidden column is in the feature set and that no
    engineered feature was derived from forbidden columns.

    Returns a dict with audit results.
    """
    FORBIDDEN = {"efficiency_wh_per_km", "range_km", "source_url"}

    results = {
        "passed": True,
        "violations": [],
        "warnings": [],
    }

    # Check direct inclusion
    for col in feature_cols:
        if col in FORBIDDEN:
            results["passed"] = False
            results["violations"].append(
                f"VIOLATION: '{col}' is in the feature set but is FORBIDDEN."
            )

    # Check that no column name contains suspicious substrings
    suspicious_patterns = ["efficiency", "wh_per_km", "range_km"]
    for col in feature_cols:
        for pat in suspicious_patterns:
            if pat in col.lower() and col not in FORBIDDEN:
                results["warnings"].append(
                    f"WARNING: '{col}' contains suspicious pattern '{pat}' — verify."
                )

    # Verify engineered features only use allowed inputs
    allowed_inputs = set(df.columns) - FORBIDDEN
    for feat in FEATURE_REGISTRY:
        if feat["name"] in feature_cols:
            for inp in feat["inputs"]:
                if inp in FORBIDDEN:
                    results["passed"] = False
                    results["violations"].append(
                        f"VIOLATION: Engineered feature '{feat['name']}' uses "
                        f"forbidden input '{inp}'."
                    )

    return results


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, ".")
    from src.data_cleaning import clean_data

    df = clean_data("data/raw")
    df = engineer_features(df)

    print("\n--- Engineered Features ---")
    eng_cols = get_engineered_feature_names()
    print(df[eng_cols].describe().round(3).T)

    print("\n--- Feature Registry ---")
    print(get_feature_registry_df()[["name", "formula", "leakage_risk"]].to_string(index=False))

    print("\n--- Leakage Audit ---")
    all_features = [c for c in df.columns if c not in {"range_km", "model", "brand",
                                                         "efficiency_wh_per_km"}]
    audit = leakage_audit(df, all_features)
    print(f"  Passed: {audit['passed']}")
    for v in audit["violations"]:
        print(f"  {v}")
    for w in audit["warnings"]:
        print(f"  {w}")
