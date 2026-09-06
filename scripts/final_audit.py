#!/usr/bin/env python3
"""
VOLTERA — Final Submission Audit Script

Comprehensive automated audit checking every submission requirement.
Run this before final submission to verify everything is consistent.
"""

import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

FORBIDDEN = {"efficiency_wh_per_km", "range_km", "source_url"}


def check(name, condition, details=""):
    """Print check result."""
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if details and not condition:
        msg += f" — {details}"
    print(msg)
    return condition


def main():
    print()
    print("=" * 60)
    print("VOLTERA FINAL SUBMISSION AUDIT")
    print("=" * 60)
    print()

    results = []

    # --- Dataset ---
    print("--- Dataset ---")
    dataset_path = os.path.join(PROJECT_ROOT, "data", "raw", "ev_specs_2025.xls")
    results.append(check("Dataset exists", os.path.exists(dataset_path)))

    try:
        import pandas as pd
        raw_df = pd.read_excel(dataset_path)
        results.append(check("Dataset has expected rows", len(raw_df) == 478,
                             f"got {len(raw_df)}"))
    except Exception as e:
        results.append(check("Dataset loads", False, str(e)))

    # --- Model artifact ---
    print("\n--- Model Artifact ---")
    model_path = os.path.join(PROJECT_ROOT, "models", "final_ev_range_pipeline.joblib")
    results.append(check("Model artifact exists", os.path.exists(model_path)))

    try:
        import joblib
        pipeline = joblib.load(model_path)
        results.append(check("Model artifact loads", pipeline is not None))
    except Exception as e:
        results.append(check("Model artifact loads", False, str(e)))
        pipeline = None

    # Test prediction
    if pipeline is not None:
        try:
            import numpy as np
            from src.feature_engineering import engineer_features

            test_row = pd.DataFrame([{
                "battery_capacity_kWh": 75.0,
                "top_speed_kmh": 180,
                "torque_nm": 430.0,
                "acceleration_0_100_s": 6.5,
                "fast_charging_power_kw_dc": 120.0,
                "towing_capacity_kg": 0.0,
                "cargo_volume_l": 500.0,
                "seats": 5,
                "length_mm": 4700,
                "width_mm": 1850,
                "height_mm": 1600,
                "drivetrain": "FWD",
                "segment": "JC - Medium",
                "car_body_type": "SUV",
                "number_of_cells": 96.0,
            }])
            test_row = engineer_features(test_row)
            pred = pipeline.predict(test_row)
            results.append(check("Prediction works", pred is not None and len(pred) == 1,
                                 f"prediction={pred[0]:.1f}"))
            results.append(check("Prediction is positive",
                                 pred[0] > 0, f"prediction={pred[0]:.1f}"))
        except Exception as e:
            results.append(check("Prediction works", False, str(e)))

    # --- Forbidden feature audit ---
    print("\n--- Leakage / Forbidden Feature Audit ---")
    meta_path = os.path.join(PROJECT_ROOT, "models", "pipeline_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

        all_features = (
            meta.get("numeric_features", []) +
            meta.get("categorical_features", [])
        )

        for forbidden in FORBIDDEN:
            results.append(check(
                f"'{forbidden}' absent from model features",
                forbidden not in all_features,
                f"FOUND in feature list!"
            ))
    else:
        results.append(check("Pipeline metadata exists", False))

    # Check leakage audit result
    leakage_path = os.path.join(PROJECT_ROOT, "outputs", "metrics", "leakage_audit.json")
    if os.path.exists(leakage_path):
        with open(leakage_path) as f:
            audit = json.load(f)
        results.append(check("Leakage audit passed", audit.get("passed", False)))
    else:
        results.append(check("Leakage audit file exists", False))

    # Check source code for hidden access
    src_dir = os.path.join(PROJECT_ROOT, "src")
    for fname in ["preprocessing.py", "feature_engineering.py", "modeling.py"]:
        fpath = os.path.join(src_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                content = f.read()
            # Check for efficiency in non-comment context
            has_eff_usage = False
            for line in content.split("\n"):
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                    continue
                if "efficiency_wh_per_km" in stripped and not stripped.startswith("#"):
                    # Check if it's not a docstring or just text
                    if "=" in stripped or "import" in stripped or "[" in stripped:
                        if "FORBIDDEN" not in stripped and "forbidden" not in stripped:
                            has_eff_usage = True
            results.append(check(
                f"No efficiency usage in {fname}",
                not has_eff_usage,
                "Found non-comment reference to efficiency_wh_per_km"
            ))

    # --- Required outputs ---
    print("\n--- Required Output Files ---")
    required_files = {
        "outputs/metrics/model_arena.csv": "Model arena results",
        "outputs/metrics/feature_ablation.csv": "Feature ablation results",
        "outputs/metrics/tuning_results.csv": "Tuning results",
        "outputs/metrics/final_metrics.json": "Final metrics",
        "outputs/metrics/split_info.json": "Split info",
        "outputs/metrics/feature_registry.csv": "Feature registry",
        "outputs/metrics/feature_audit.csv": "Feature audit",
        "outputs/metrics/data_quality_audit.csv": "Data quality audit",
        "outputs/metrics/missing_value_summary.csv": "Missing value summary",
        "outputs/metrics/leakage_audit.json": "Leakage audit",
        "outputs/predictions/physics_sanity_check.csv": "Physics sanity check",
        "data/processed/ev_data_cleaned.csv": "Cleaned data",
    }

    for fpath, desc in required_files.items():
        full = os.path.join(PROJECT_ROOT, fpath)
        results.append(check(f"{desc} exists", os.path.exists(full)))

    # --- Figures ---
    print("\n--- Figures ---")
    fig_dir = os.path.join(PROJECT_ROOT, "outputs", "figures")
    if os.path.exists(fig_dir):
        figs = os.listdir(fig_dir)
        results.append(check("At least 10 figures generated",
                             len(figs) >= 10, f"found {len(figs)}"))
    else:
        results.append(check("Figures directory exists", False))

    # --- Notebook ---
    print("\n--- Notebook ---")
    nb_path = os.path.join(PROJECT_ROOT, "notebooks", "TECHTRACK3_Winning_Solution.ipynb")
    results.append(check("Notebook file exists", os.path.exists(nb_path)))

    if os.path.exists(nb_path):
        try:
            import nbformat
            with open(nb_path) as f:
                nb = nbformat.read(f, as_version=4)
            nbformat.validate(nb)
            results.append(check("Notebook schema valid", True))
        except Exception as e:
            results.append(check("Notebook schema valid", False, str(e)))

    # --- App imports ---
    print("\n--- App & API ---")
    try:
        # Test Streamlit app imports (not full run)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "app_module", os.path.join(PROJECT_ROOT, "app", "app.py")
        )
        results.append(check("Streamlit app file exists", spec is not None))
    except Exception as e:
        results.append(check("Streamlit app imports", False, str(e)))

    api_path = os.path.join(PROJECT_ROOT, "app", "api.py")
    results.append(check("API file exists", os.path.exists(api_path)))

    # --- Submission files ---
    print("\n--- Submission Package ---")
    submission_files = {
        "README.md": "README",
        "requirements.txt": "Requirements",
        "JUDGE_QA.md": "Judge QA",
    }
    for fpath, desc in submission_files.items():
        full = os.path.join(PROJECT_ROOT, fpath)
        results.append(check(f"{desc} exists", os.path.exists(full)))

    sub_dir = os.path.join(PROJECT_ROOT, "submission")
    sub_files = {
        "SUBMISSION_CHECKLIST.md": "Submission checklist",
        "FINAL_RESULTS.md": "Final results",
        "JUDGE_DEMO_GUIDE.md": "Judge demo guide",
    }
    for fpath, desc in sub_files.items():
        full = os.path.join(sub_dir, fpath)
        results.append(check(f"{desc} exists", os.path.exists(full)))

    # --- Consistency audit ---
    print("\n--- Consistency Audit ---")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

        metrics_path = os.path.join(PROJECT_ROOT, "outputs", "metrics", "final_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                saved_metrics = json.load(f)

            # Check model name consistency
            meta_model = meta.get("final_model_name", "")
            metrics_model = saved_metrics.get("model", "")
            results.append(check(
                "Model name consistent (metadata vs metrics)",
                meta_model == metrics_model,
                f"metadata='{meta_model}', metrics='{metrics_model}'"
            ))

    # Check for stale references
    stale_patterns = ["PhysicsResidualRegressor", "base_pipeline__"]
    for pattern in stale_patterns:
        found_in = []
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # Skip .git and .venv
            dirs[:] = [d for d in dirs if d not in {".git", ".venv", "__pycache__", "node_modules", "scripts", "reports"}]
            for fname in files:
                if fname.endswith((".py", ".json")):
                    if fname == "final_audit.py" or fname == "physics.py":
                        continue
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath) as f:
                            if pattern in f.read():
                                rel = os.path.relpath(fpath, PROJECT_ROOT)
                                found_in.append(rel)
                    except Exception:
                        pass
        results.append(check(
            f"No stale '{pattern}' references",
            len(found_in) == 0,
            f"found in: {', '.join(found_in)}"
        ))

    # --- Final Summary ---
    print()
    print("=" * 60)
    total = len(results)
    passed = sum(results)
    failed = total - passed

    print(f"Total checks: {total}")
    print(f"Passed:       {passed}")
    print(f"Failed:       {failed}")
    print()

    if failed == 0:
        print("SUBMISSION READY: YES")
    else:
        print("SUBMISSION READY: NO")
        print(f"Fix {failed} failing check(s) before submission.")

    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
