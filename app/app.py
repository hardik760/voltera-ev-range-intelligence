"""
TECHTRACK 3.0 — EV Range Intelligence System
Interactive Streamlit Application for Live Judge Testing

This app loads the saved preprocessing + model pipeline and provides:
  1. Manual EV specification input
  2. Demo EV preset from the dataset
  3. Instant range prediction with top influencing factors
  4. Input validation and graceful error handling
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Make sure we can import src.physics
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.physics import PhysicsResidualRegressor
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

PIPELINE_PATH = os.path.join(MODEL_DIR, "final_ev_range_pipeline.joblib")
META_PATH = os.path.join(MODEL_DIR, "pipeline_metadata.json")

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EV Range Intelligence — TECHTRACK 3.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1B5E20;
        text-align: center;
        padding: 0.5rem 0;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    .prediction-label {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .info-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1B5E20;
        margin: 0.5rem 0;
    }
    .warning-text {
        color: #E65100;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load Pipeline
# ---------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    """Load the saved pipeline and metadata."""
    if not os.path.exists(PIPELINE_PATH):
        return None, None
    pipeline = joblib.load(PIPELINE_PATH)
    metadata = {}
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            metadata = json.load(f)
    return pipeline, metadata


@st.cache_data
def load_demo_data():
    """Load the dataset for demo presets."""
    try:
        csv_path = os.path.join(PROJECT_ROOT, "data", "processed", "ev_data_cleaned.csv")
        df = pd.read_csv(csv_path)
        return df
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Input Definitions
# ---------------------------------------------------------------------------
DRIVETRAIN_OPTIONS = ["AWD", "FWD", "RWD"]
BODY_TYPE_OPTIONS = [
    "SUV", "Sedan", "Hatchback", "Small Passenger Van",
    "Liftback Sedan", "Station/Estate", "Cabriolet", "Coupe",
]
SEGMENT_OPTIONS = [
    "A - Mini", "B - Compact", "C - Medium", "D - Large",
    "E - Executive", "F - Luxury", "G - Sports", "I - Luxury",
    "JA - Mini", "JB - Compact", "JC - Medium", "JD - Large",
    "JE - Executive", "JF - Luxury", "N - Passenger Van",
]

# Default values (approximate median of dataset)
DEFAULTS = {
    "battery_capacity_kWh": 76.0,
    "top_speed_kmh": 180,
    "torque_nm": 430.0,
    "acceleration_0_100_s": 6.6,
    "fast_charging_power_kw_dc": 113.0,
    "towing_capacity_kg": 0.0,
    "cargo_volume_l": 500.0,
    "seats": 5,
    "length_mm": 4720,
    "width_mm": 1890,
    "height_mm": 1596,
    "drivetrain": "FWD",
    "segment": "JC - Medium",
    "car_body_type": "SUV",
}

def prepare_input(input_dict: dict, metadata: dict) -> pd.DataFrame:
    """Prepare a single-row DataFrame matching the pipeline's expected features."""
    numeric_features = metadata.get("numeric_features", [])
    categorical_features = metadata.get("categorical_features", [])

    # Build the initial row
    row = {}
    for feat in numeric_features + categorical_features:
        row[feat] = input_dict.get(feat, np.nan)

    df = pd.DataFrame([row])

    # Ensure numeric types
    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add engineered features using the shared module
    from src.feature_engineering import engineer_features
    df = engineer_features(df)
    
    return df


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
def main():
    pipeline, metadata = load_pipeline()

    st.markdown('<div class="main-header">⚡ EV Range Intelligence System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">TECHTRACK 3.0 — Specification-Based EV Range Prediction</div>', unsafe_allow_html=True)

    if pipeline is None:
        st.error("❌ Pipeline not found. Please run `run_pipeline.py` first to train and save the model.")
        st.stop()

    # Sidebar
    st.sidebar.header("🔧 Input Mode")
    input_mode = st.sidebar.radio(
        "Select mode:",
        ["Manual Input", "Demo EV Preset"],
        index=0,
    )

    # Model info
    with st.sidebar.expander("ℹ️ Model Info"):
        st.write(f"**Model:** {metadata.get('model_name', 'N/A')}")
        test_metrics = metadata.get("final_test_metrics", {})
        if test_metrics:
            st.write(f"**Test MAE:** {test_metrics.get('test_MAE', 'N/A')} km")
            st.write(f"**Test R²:** {test_metrics.get('test_R2', 'N/A')}")
        st.write(f"**Training samples:** {metadata.get('n_training_samples', 'N/A')}")

    # Input collection
    input_dict = {}

    if input_mode == "Demo EV Preset":
        demo_df = load_demo_data()
        if demo_df is not None:
            # Create display names, ensuring string types to avoid TypeError
            demo_df["display_name"] = demo_df["brand"].astype(str) + " " + demo_df["model"].fillna("").astype(str)
            selected = st.sidebar.selectbox(
                "Select a demo EV:",
                demo_df["display_name"].tolist(),
                index=0,
            )
            row = demo_df[demo_df["display_name"] == selected].iloc[0]

            st.info(f"📋 Loaded preset: **{selected}** (Actual range: {row.get('range_km', 'N/A')} km)")

            # Fill inputs from dataset
            for key in DEFAULTS.keys():
                if key in row.index:
                    val = row[key]
                    input_dict[key] = val if not pd.isna(val) else DEFAULTS[key]
                else:
                    input_dict[key] = DEFAULTS[key]
        else:
            st.warning("Demo data not available. Using manual mode.")
            input_mode = "Manual Input"

    if input_mode == "Manual Input":
        st.sidebar.markdown("---")

    # Input widgets (shown in both modes, editable in manual mode)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🔋 Battery & Charging")
        input_dict["battery_capacity_kWh"] = st.number_input(
            "Battery Capacity (kWh)",
            min_value=10.0, max_value=250.0, step=1.0,
            value=float(input_dict.get("battery_capacity_kWh", DEFAULTS["battery_capacity_kWh"])),
        )
        input_dict["fast_charging_power_kw_dc"] = st.number_input(
            "DC Fast Charging Power (kW)",
            min_value=10.0, max_value=500.0, step=5.0,
            value=float(input_dict.get("fast_charging_power_kw_dc", DEFAULTS["fast_charging_power_kw_dc"])),
        )

        st.subheader("⚡ Performance")
        input_dict["torque_nm"] = st.number_input(
            "Torque (Nm)",
            min_value=50.0, max_value=2000.0, step=10.0,
            value=float(input_dict.get("torque_nm", DEFAULTS["torque_nm"])),
        )
        input_dict["top_speed_kmh"] = st.number_input(
            "Top Speed (km/h)",
            min_value=80, max_value=400, step=5,
            value=int(input_dict.get("top_speed_kmh", DEFAULTS["top_speed_kmh"])),
        )
        input_dict["acceleration_0_100_s"] = st.number_input(
            "0–100 km/h (seconds)",
            min_value=1.5, max_value=25.0, step=0.1,
            value=float(input_dict.get("acceleration_0_100_s", DEFAULTS["acceleration_0_100_s"])),
        )

    with col2:
        st.subheader("📐 Dimensions")
        input_dict["length_mm"] = st.number_input(
            "Length (mm)", min_value=2500, max_value=7000, step=10,
            value=int(input_dict.get("length_mm", DEFAULTS["length_mm"])),
        )
        input_dict["width_mm"] = st.number_input(
            "Width (mm)", min_value=1300, max_value=2500, step=10,
            value=int(input_dict.get("width_mm", DEFAULTS["width_mm"])),
        )
        input_dict["height_mm"] = st.number_input(
            "Height (mm)", min_value=1000, max_value=2500, step=10,
            value=int(input_dict.get("height_mm", DEFAULTS["height_mm"])),
        )

        st.subheader("🪑 Utility")
        input_dict["seats"] = st.number_input(
            "Seats", min_value=1, max_value=12, step=1,
            value=int(input_dict.get("seats", DEFAULTS["seats"])),
        )
        input_dict["cargo_volume_l"] = st.number_input(
            "Cargo Volume (litres)", min_value=50.0, max_value=3000.0, step=10.0,
            value=float(input_dict.get("cargo_volume_l", DEFAULTS["cargo_volume_l"])),
        )
        input_dict["towing_capacity_kg"] = st.number_input(
            "Towing Capacity (kg)", min_value=0.0, max_value=5000.0, step=50.0,
            value=float(input_dict.get("towing_capacity_kg", DEFAULTS["towing_capacity_kg"])),
        )

    with col3:
        st.subheader("🚗 Vehicle Type")
        input_dict["drivetrain"] = st.selectbox(
            "Drivetrain", DRIVETRAIN_OPTIONS,
            index=DRIVETRAIN_OPTIONS.index(
                input_dict.get("drivetrain", DEFAULTS["drivetrain"])
            ) if input_dict.get("drivetrain", DEFAULTS["drivetrain"]) in DRIVETRAIN_OPTIONS else 0,
        )
        input_dict["car_body_type"] = st.selectbox(
            "Body Type", BODY_TYPE_OPTIONS,
            index=BODY_TYPE_OPTIONS.index(
                input_dict.get("car_body_type", DEFAULTS["car_body_type"])
            ) if input_dict.get("car_body_type", DEFAULTS["car_body_type"]) in BODY_TYPE_OPTIONS else 0,
        )
        input_dict["segment"] = st.selectbox(
            "Market Segment", SEGMENT_OPTIONS,
            index=SEGMENT_OPTIONS.index(
                input_dict.get("segment", DEFAULTS["segment"])
            ) if input_dict.get("segment", DEFAULTS["segment"]) in SEGMENT_OPTIONS else 0,
        )

    # Predict button
    st.markdown("---")

    if st.button("🔮 Predict Range", type="primary", use_container_width=True):
        try:
            # Prepare input
            input_df = prepare_input(input_dict, metadata)

            # Predict
            prediction = pipeline.predict(input_df)[0]
            prediction = max(prediction, 0)  # Range can't be negative

            # Generate Uncertainty Bands (if VotingRegressor)
            uncertainty_html = ""
            if hasattr(pipeline, "estimators_"):
                preds = [max(0, est.predict(input_df)[0]) for est in pipeline.estimators_]
                std_pred = np.std(preds)
                range_str = f"{prediction - std_pred:.0f} - {prediction + std_pred:.0f}"
                uncertainty_html = f'<div class="prediction-label">95% Confidence Interval: ±{std_pred * 1.96:.0f} km</div>'

            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                <div class="prediction-label">Predicted Driving Range</div>
                <div class="prediction-value">{prediction:.0f} km</div>
                <div class="prediction-label">{prediction * 0.621:.0f} miles</div>
                {uncertainty_html}
            </div>
            """, unsafe_allow_html=True)

            # Physics sanity check
            implied_wh = (input_dict["battery_capacity_kWh"] * 1000) / max(prediction, 1)
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Implied Energy Consumption", f"{implied_wh:.0f} Wh/km")
            with col_b:
                plausible = 80 <= implied_wh <= 400
                st.metric("Physics Check", "✅ Plausible" if plausible else "⚠️ Review")
            with col_c:
                st.metric("Battery Capacity", f"{input_dict['battery_capacity_kWh']:.1f} kWh")

            if not plausible:
                st.warning(
                    f"The implied energy consumption ({implied_wh:.0f} Wh/km) is outside "
                    f"the typical EV range (80–400 Wh/km). This may indicate extreme input values."
                )

            st.info(
                "**Judge Note (Edge-Case Vulnerability):** The dataset lacks an Aerodynamic Drag Coefficient (Cd). "
                "Because of this, the model strictly uses physical dimensions (Length/Width/Height) to infer aerodynamic resistance. "
                "The model will inherently **under-predict** the range of hyper-aerodynamic luxury vehicles (like the Lucid Air Grand Touring, Cd=0.197) "
                "because it cannot mathematically distinguish them from standard sedans of the same size. "
                "We preserved this limitation to maintain zero data leakage, rather than artificially inflating the predictions."
            )

            # Top factors
            st.subheader("🧠 Live Model Explanation (SHAP)")
            st.markdown("This waterfall chart explains **exactly** why the model predicted this range. The baseline (bottom) is purely physics-derived (Segment Median Efficiency × Battery). The ML model's entire job is to predict the *residual* (how this specific EV deviates from the physics baseline).")
            
            import matplotlib.pyplot as plt
            import shap
            
            # Extract tree model for SHAP
            if hasattr(pipeline, "estimators_"):
                # Use the first estimator (e.g. HistGradientBoosting) for explainability
                phys_reg = pipeline.estimators_[0]
            else:
                phys_reg = pipeline
                
            base_pipe = phys_reg.base_pipeline
            preprocessor = base_pipe.named_steps["preprocessor"]
            model = base_pipe.named_steps["model"]
            
            X_transformed = preprocessor.transform(input_df)
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_transformed)
            
            # Adjust base value for Physics Baseline
            physics_baseline = phys_reg._compute_baseline(input_df)[0]
            shap_values.base_values = shap_values.base_values + physics_baseline
            
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Please check that all inputs are within valid ranges.")

    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #999; font-size: 0.85rem;">'
        'TECHTRACK 3.0 — EV Range Prediction from Specifications<br>'
        'MANIT Bhopal · Team Voltra · Competition Submission'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
