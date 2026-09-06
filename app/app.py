"""
VOLTERA — EV Range Intelligence System
Interactive Streamlit Application for Live Judge Testing

This app loads the saved preprocessing + model pipeline and provides:
  1. Manual EV specification input
  2. Demo EV preset from the dataset
  3. Instant range prediction with physics sanity checking
  4. What-if analysis
  5. Model explainability
  6. Input validation and graceful error handling

LEAKAGE POLICY: This app NEVER requires or uses efficiency_wh_per_km.
The pipeline predicts range_km directly from specifications.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Project imports
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

PIPELINE_PATH = os.path.join(MODEL_DIR, "final_ev_range_pipeline.joblib")
META_PATH = os.path.join(MODEL_DIR, "pipeline_metadata.json")

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VOLTERA — EV Range Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS — clean EV engineering aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 2.0rem;
        font-weight: 800;
        color: #0d1b2a;
        text-align: center;
        padding: 0.5rem 0 0 0;
        letter-spacing: -0.02em;
    }
    .brand-sigma {
        color: #2563eb;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    .prediction-box {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b3a5c 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        border: 1px solid #1e40af;
    }
    .prediction-value {
        font-size: 3.2rem;
        font-weight: 800;
        margin: 0.5rem 0;
        letter-spacing: -0.02em;
    }
    .prediction-label {
        font-size: 1.0rem;
        opacity: 0.85;
        font-weight: 400;
    }
    .info-card {
        background: #f8fafc;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        border-left: 3px solid #2563eb;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .limitation-note {
        background: #fffbeb;
        border-left: 3px solid #d97706;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #92400e;
        margin-top: 1rem;
    }
    div[data-testid="stSidebar"] {
        background: #f8fafc;
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

    st.markdown(
        '<div class="main-header">VOLT<span class="brand-sigma">Σ</span>RA</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">EV Range Intelligence System — TECHTRACK 3.0</div>',
        unsafe_allow_html=True
    )

    if pipeline is None:
        st.error("Pipeline not found. Run `python run_pipeline.py` first.")
        st.stop()

    # Sidebar
    st.sidebar.header("Input Mode")
    input_mode = st.sidebar.radio(
        "Select mode:",
        ["Manual Input", "Demo EV Preset"],
        index=0,
    )

    # Model info in sidebar
    with st.sidebar.expander("Model Information"):
        model_name = metadata.get("final_model_name", metadata.get("model_name", "N/A"))
        st.write(f"**Final Model:** {model_name}")
        
        test_metrics = metadata.get("test_metrics", metadata.get("final_test_metrics", {}))
        cv_metrics = metadata.get("cv_metrics", {})
        
        if test_metrics:
            st.write(f"**Holdout Test MAE:** {test_metrics.get('test_MAE', 'N/A')} km")
            st.write(f"**Holdout Test R²:** {test_metrics.get('test_R2', 'N/A')}")
            st.write(f"**Holdout Test RMSE:** {test_metrics.get('test_RMSE', 'N/A')} km")
        
        if cv_metrics:
            st.write(f"**CV MAE:** {cv_metrics.get('cv_MAE', 'N/A')} km")
            st.write(f"**CV R²:** {cv_metrics.get('cv_R2', 'N/A')}")
        
        dev_train = metadata.get("development_train_size", "N/A")
        holdout = metadata.get("holdout_test_size", "N/A")
        deploy = metadata.get("final_deployment_fit_size", metadata.get("n_training_samples", "N/A"))
        
        st.write("---")
        st.write(f"**Development train:** {dev_train}")
        st.write(f"**Holdout test:** {holdout}")
        st.write(f"**Final deployment fit:** {deploy}")
        
        st.write("---")
        st.write("**Leakage status:** efficiency_wh_per_km excluded")
        st.write("**Validation:** 10-fold CV on training data")

    # Input collection
    input_dict = {}

    if input_mode == "Demo EV Preset":
        demo_df = load_demo_data()
        if demo_df is not None:
            demo_df["display_name"] = demo_df["brand"].astype(str) + " " + demo_df["model"].fillna("").astype(str)
            selected = st.sidebar.selectbox(
                "Select a demo EV:",
                demo_df["display_name"].tolist(),
                index=0,
            )
            row = demo_df[demo_df["display_name"] == selected].iloc[0]

            actual_range = row.get("range_km", "N/A")
            st.info(f"Loaded preset: **{selected}** (Actual range: {actual_range} km)")

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

    # Input widgets
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Battery & Charging")
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

        st.subheader("Performance")
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
            "0-100 km/h (seconds)",
            min_value=1.5, max_value=25.0, step=0.1,
            value=float(input_dict.get("acceleration_0_100_s", DEFAULTS["acceleration_0_100_s"])),
        )

    with col2:
        st.subheader("Dimensions")
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

        st.subheader("Utility")
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
        st.subheader("Vehicle Type")
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

    if st.button("Predict Range", type="primary", use_container_width=True):
        try:
            input_df = prepare_input(input_dict, metadata)
            prediction = pipeline.predict(input_df)[0]
            prediction = max(prediction, 0)  # Range can't be negative

            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                <div class="prediction-label">Predicted Driving Range</div>
                <div class="prediction-value">{prediction:.0f} km</div>
                <div class="prediction-label">{prediction * 0.621:.0f} miles</div>
            </div>
            """, unsafe_allow_html=True)

            # Physics sanity check
            implied_wh = (input_dict["battery_capacity_kWh"] * 1000) / max(prediction, 1)
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Implied Energy Consumption", f"{implied_wh:.0f} Wh/km")
            with col_b:
                plausible = 80 <= implied_wh <= 400
                st.metric("Physics Check", "Plausible" if plausible else "Review needed")
            with col_c:
                st.metric("Battery Capacity", f"{input_dict['battery_capacity_kWh']:.1f} kWh")

            if not plausible:
                st.warning(
                    f"The implied energy consumption ({implied_wh:.0f} Wh/km) is outside "
                    f"the typical EV range (80-400 Wh/km). This may indicate extreme input values."
                )

            # What-if analysis
            st.subheader("What-If Analysis")
            st.caption("Change one parameter to see how the predicted range responds.")
            
            wif_col1, wif_col2 = st.columns(2)
            with wif_col1:
                wif_param = st.selectbox("Parameter to vary", [
                    "battery_capacity_kWh", "top_speed_kmh", "torque_nm",
                    "acceleration_0_100_s", "length_mm", "width_mm", "height_mm"
                ])
            with wif_col2:
                current_val = float(input_dict.get(wif_param, 0))
                wif_range_pct = st.slider("Variation range (%)", 10, 100, 50)
            
            lo = current_val * (1 - wif_range_pct / 100)
            hi = current_val * (1 + wif_range_pct / 100)
            wif_values = np.linspace(lo, hi, 15)
            wif_preds = []
            for v in wif_values:
                wif_dict = input_dict.copy()
                wif_dict[wif_param] = v
                wif_df = prepare_input(wif_dict, metadata)
                wif_preds.append(max(0, pipeline.predict(wif_df)[0]))
            
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(wif_values, wif_preds, "o-", color="#2563eb", linewidth=2, markersize=4)
            ax.axvline(current_val, color="red", linestyle="--", alpha=0.6, label=f"Current: {current_val:.0f}")
            ax.axhline(prediction, color="gray", linestyle=":", alpha=0.4)
            ax.set_xlabel(wif_param.replace("_", " "))
            ax.set_ylabel("Predicted Range (km)")
            ax.set_title(f"Sensitivity: {wif_param.replace('_', ' ')}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

            # Explainability — permutation importance from saved file
            st.subheader("Feature Importance")
            perm_path = os.path.join(PROJECT_ROOT, "outputs", "metrics", "permutation_importance.csv")
            if os.path.exists(perm_path):
                perm_df = pd.read_csv(perm_path).head(10)
                fig, ax = plt.subplots(figsize=(8, 4))
                perm_plot = perm_df.sort_values("importance_mean")
                ax.barh(perm_plot["feature"], perm_plot["importance_mean"],
                        xerr=perm_plot["importance_std"], color="#2563eb", alpha=0.85)
                ax.set_xlabel("Permutation Importance")
                ax.set_title("Top 10 Features by Importance")
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Run the pipeline to generate feature importance data.")

            # Limitation note
            st.markdown("""
            <div class="limitation-note">
            <strong>Limitations:</strong> This model predicts official rated range from static specifications only.
            It does not account for weather, driving style, terrain, HVAC usage, or battery degradation.
            The dataset lacks aerodynamic drag coefficient (Cd) and vehicle weight, which limits accuracy
            for hyper-aerodynamic or unusually heavy vehicles. Tree-based models cannot extrapolate
            beyond training data ranges.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Please check that all inputs are within valid ranges.")

    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #94a3b8; font-size: 0.85rem;">'
        'TECHTRACK 3.0 — EV Range Prediction from Specifications<br>'
        'MANIT Bhopal &middot; Team Voltra &middot; Competition Submission'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
