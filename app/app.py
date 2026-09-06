"""
VOLTERA — EV Range Intelligence System
Interactive Streamlit Application for Live Judge Testing

This app loads the saved preprocessing + model pipeline and provides:
  1. Manual EV specification input with validation
  2. Demo EV presets (Compact / Mid-size / Premium)
  3. Instant range prediction with physics sanity checking
  4. What-if sensitivity analysis
  5. Live local SHAP explanation for the current prediction
  6. Global feature importance
  7. Model information and methodology transparency

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
# Custom CSS
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
        margin-bottom: 0.5rem;
        font-weight: 400;
    }
    .disclaimer {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-size: 0.82rem;
        color: #0c4a6e;
        text-align: center;
        margin-bottom: 1.5rem;
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
# Demo Presets
# ---------------------------------------------------------------------------
DEMO_PRESETS = {
    "🚗 Compact EV (City Car)": {
        "battery_capacity_kWh": 40.0,
        "top_speed_kmh": 150,
        "torque_nm": 220.0,
        "acceleration_0_100_s": 9.0,
        "fast_charging_power_kw_dc": 80.0,
        "towing_capacity_kg": 0.0,
        "cargo_volume_l": 310.0,
        "seats": 5,
        "length_mm": 4060,
        "width_mm": 1770,
        "height_mm": 1520,
        "drivetrain": "FWD",
        "segment": "B - Compact",
        "car_body_type": "Hatchback",
    },
    "🚙 Mid-Size EV (Family SUV)": {
        "battery_capacity_kWh": 77.0,
        "top_speed_kmh": 185,
        "torque_nm": 450.0,
        "acceleration_0_100_s": 6.2,
        "fast_charging_power_kw_dc": 135.0,
        "towing_capacity_kg": 1600.0,
        "cargo_volume_l": 540.0,
        "seats": 5,
        "length_mm": 4695,
        "width_mm": 1890,
        "height_mm": 1650,
        "drivetrain": "AWD",
        "segment": "JC - Medium",
        "car_body_type": "SUV",
    },
    "💎 Premium EV (Luxury Sedan)": {
        "battery_capacity_kWh": 111.0,
        "top_speed_kmh": 250,
        "torque_nm": 760.0,
        "acceleration_0_100_s": 3.5,
        "fast_charging_power_kw_dc": 270.0,
        "towing_capacity_kg": 1000.0,
        "cargo_volume_l": 620.0,
        "seats": 5,
        "length_mm": 5021,
        "width_mm": 1961,
        "height_mm": 1440,
        "drivetrain": "AWD",
        "segment": "F - Luxury",
        "car_body_type": "Sedan",
    },
}

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

    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    from src.feature_engineering import engineer_features
    df = engineer_features(df)

    return df


def compute_live_shap(pipeline, input_df, feature_names):
    """Compute live SHAP values for a single prediction."""
    try:
        import shap

        # Extract model and preprocessor
        if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
            model = pipeline.named_steps["model"]
            preprocessor = pipeline.named_steps["preprocessor"]
        elif hasattr(pipeline, "estimators_"):
            first_est = pipeline.estimators_[0]
            if isinstance(first_est, tuple):
                first_est = first_est[1]
            model = first_est.named_steps["model"]
            preprocessor = first_est.named_steps["preprocessor"]
        else:
            return None

        X_transformed = preprocessor.transform(input_df)

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_transformed)
        except Exception:
            return None

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Get values for first (only) row
        sv = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0]

        return {
            "shap_values": sv,
            "base_value": float(base_value),
            "feature_names": feature_names,
        }
    except Exception:
        return None


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
        '<div class="sub-header">EV Range Intelligence — Specification-Based Range Prediction</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="disclaimer">'
        '⚡ This model estimates EV range from vehicle specifications. '
        'It is <strong>not</strong> a real-time driving-range predictor and does not model '
        'traffic, weather, HVAC, driver behaviour, battery degradation, or live telemetry.'
        '</div>',
        unsafe_allow_html=True
    )

    if pipeline is None:
        st.error("Pipeline not found. Run `python run_pipeline.py` first.")
        st.stop()

    # Sidebar — Input Mode
    st.sidebar.header("Input Mode")
    input_mode = st.sidebar.radio(
        "Select mode:",
        ["Manual Input", "Demo Preset", "Dataset EV"],
        index=0,
    )

    # Model info in sidebar
    with st.sidebar.expander("📊 Model Information", expanded=False):
        model_name = metadata.get("final_model_name", "N/A")
        st.write(f"**Final Model:** {model_name}")

        test_metrics = metadata.get("test_metrics", {})
        cv_metrics = metadata.get("cv_metrics", {})

        if test_metrics:
            st.write(f"**Test MAE:** {test_metrics.get('test_MAE', 'N/A')} km")
            st.write(f"**Test R²:** {test_metrics.get('test_R2', 'N/A')}")
            st.write(f"**Test RMSE:** {test_metrics.get('test_RMSE', 'N/A')} km")

        if cv_metrics:
            st.write(f"**CV MAE:** {cv_metrics.get('cv_MAE', 'N/A')} ± {cv_metrics.get('cv_MAE_std', 'N/A')} km")
            st.write(f"**CV R²:** {cv_metrics.get('cv_R2', 'N/A')}")

        st.write("---")
        st.write(f"**Dev train:** {metadata.get('development_train_size', 'N/A')}")
        st.write(f"**Holdout test:** {metadata.get('holdout_test_size', 'N/A')}")
        st.write(f"**Deployment fit:** {metadata.get('final_deployment_fit_size', 'N/A')}")
        st.write("---")
        st.write("**Leakage status:** `efficiency_wh_per_km` excluded")
        st.write("**Validation:** 10-fold CV on training data")
        st.write(f"**Seed:** {metadata.get('seed', 42)}")

    # Input collection
    input_dict = {}

    if input_mode == "Demo Preset":
        preset_name = st.sidebar.selectbox("Select preset:", list(DEMO_PRESETS.keys()))
        input_dict = DEMO_PRESETS[preset_name].copy()
        st.info(f"Loaded preset: **{preset_name}**")

    elif input_mode == "Dataset EV":
        demo_df = load_demo_data()
        if demo_df is not None:
            demo_df["display_name"] = demo_df["brand"].astype(str) + " " + demo_df["model"].fillna("").astype(str)
            selected = st.sidebar.selectbox("Select EV:", demo_df["display_name"].tolist(), index=0)
            row = demo_df[demo_df["display_name"] == selected].iloc[0]
            actual_range = row.get("range_km", "N/A")
            st.info(f"Loaded: **{selected}** (Actual range: {actual_range} km)")
            for key in DEFAULTS.keys():
                if key in row.index:
                    val = row[key]
                    input_dict[key] = val if not pd.isna(val) else DEFAULTS[key]
                else:
                    input_dict[key] = DEFAULTS[key]
        else:
            st.warning("Dataset not available. Using manual mode.")
            input_mode = "Manual Input"

    # Input widgets
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🔋 Battery & Charging")
        input_dict["battery_capacity_kWh"] = st.number_input(
            "Battery Capacity (kWh)", min_value=10.0, max_value=250.0, step=1.0,
            value=float(input_dict.get("battery_capacity_kWh", DEFAULTS["battery_capacity_kWh"])),
        )
        input_dict["fast_charging_power_kw_dc"] = st.number_input(
            "DC Fast Charging Power (kW)", min_value=10.0, max_value=500.0, step=5.0,
            value=float(input_dict.get("fast_charging_power_kw_dc", DEFAULTS["fast_charging_power_kw_dc"])),
        )

        st.subheader("⚡ Performance")
        input_dict["torque_nm"] = st.number_input(
            "Torque (Nm)", min_value=50.0, max_value=2000.0, step=10.0,
            value=float(input_dict.get("torque_nm", DEFAULTS["torque_nm"])),
        )
        input_dict["top_speed_kmh"] = st.number_input(
            "Top Speed (km/h)", min_value=80, max_value=400, step=5,
            value=int(input_dict.get("top_speed_kmh", DEFAULTS["top_speed_kmh"])),
        )
        input_dict["acceleration_0_100_s"] = st.number_input(
            "0-100 km/h (seconds)", min_value=1.5, max_value=25.0, step=0.1,
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

        st.subheader("🚛 Utility")
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
        st.subheader("🏷️ Vehicle Type")
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

    if st.button("⚡ Predict Range", type="primary", use_container_width=True):
        try:
            input_df = prepare_input(input_dict, metadata)
            prediction = pipeline.predict(input_df)[0]
            prediction = max(prediction, 0)

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
                st.metric("Physics Check", "✅ Plausible" if plausible else "⚠️ Review")
            with col_c:
                st.metric("Battery Capacity", f"{input_dict['battery_capacity_kWh']:.1f} kWh")

            if not plausible:
                st.warning(
                    f"The implied energy consumption ({implied_wh:.0f} Wh/km) is outside "
                    f"the typical EV range (80-400 Wh/km). This may indicate extreme input values."
                )

            # --- Tabs for detailed analysis ---
            tab1, tab2, tab3, tab4 = st.tabs([
                "🔍 Local Explanation", "📊 Global Importance",
                "🔄 What-If Analysis", "ℹ️ Methodology"
            ])

            # --- Tab 1: Local Explanation (Live SHAP) ---
            with tab1:
                st.subheader("Why This Prediction?")
                st.caption("Shows which features pushed the prediction higher or lower than average.")

                # Get feature names
                try:
                    if hasattr(pipeline, "named_steps") and "preprocessor" in pipeline.named_steps:
                        feature_names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
                    elif hasattr(pipeline, "estimators_"):
                        first = pipeline.estimators_[0]
                        if isinstance(first, tuple):
                            first = first[1]
                        feature_names = list(first.named_steps["preprocessor"].get_feature_names_out())
                    else:
                        feature_names = metadata.get("numeric_features", []) + metadata.get("categorical_features", [])
                except Exception:
                    feature_names = metadata.get("numeric_features", []) + metadata.get("categorical_features", [])

                shap_result = compute_live_shap(pipeline, input_df, feature_names)

                if shap_result is not None:
                    sv = shap_result["shap_values"]
                    fn = shap_result["feature_names"]
                    base = shap_result["base_value"]

                    # Sort by absolute SHAP value
                    sorted_idx = np.argsort(np.abs(sv))[::-1][:12]

                    fig, ax = plt.subplots(figsize=(9, 5))
                    sorted_feats = [fn[i] if i < len(fn) else f"feat_{i}" for i in sorted_idx]
                    sorted_vals = [sv[i] for i in sorted_idx]
                    colors = ["#2563eb" if v > 0 else "#dc2626" for v in sorted_vals]

                    ax.barh(range(len(sorted_feats)), sorted_vals, color=colors, alpha=0.85)
                    ax.set_yticks(range(len(sorted_feats)))
                    ax.set_yticklabels(sorted_feats, fontsize=9)
                    ax.set_xlabel("SHAP Value (impact on prediction)")
                    ax.set_title(f"Local Explanation — Base: {base:.0f} km → Prediction: {prediction:.0f} km")
                    ax.axvline(0, color="black", linewidth=0.5)
                    ax.invert_yaxis()
                    ax.grid(True, alpha=0.2)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    st.markdown(
                        '<div class="info-card">'
                        '<strong>Reading the chart:</strong> Blue bars push the prediction higher (more range). '
                        'Red bars push it lower. The base value is the average prediction across all training data.'
                        '</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.info("Live SHAP explanation not available for this model type. See global importance tab.")

            # --- Tab 2: Global Feature Importance ---
            with tab2:
                st.subheader("Global Feature Importance")
                st.caption("Which features matter most across all predictions (from permutation importance).")

                perm_path = os.path.join(PROJECT_ROOT, "outputs", "metrics", "permutation_importance.csv")
                shap_path = os.path.join(PROJECT_ROOT, "outputs", "metrics", "shap_importance.csv")

                if os.path.exists(perm_path):
                    perm_df = pd.read_csv(perm_path).head(15)
                    fig, ax = plt.subplots(figsize=(9, 5))
                    perm_plot = perm_df.sort_values("importance_mean")
                    ax.barh(perm_plot["feature"], perm_plot["importance_mean"],
                            xerr=perm_plot["importance_std"], color="#2563eb", alpha=0.85)
                    ax.set_xlabel("Permutation Importance (MAE increase when shuffled)")
                    ax.set_title("Top Features by Permutation Importance")
                    ax.grid(True, alpha=0.2)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                if os.path.exists(shap_path):
                    shap_df = pd.read_csv(shap_path).head(15)
                    fig, ax = plt.subplots(figsize=(9, 5))
                    shap_plot = shap_df.sort_values("mean_abs_shap")
                    ax.barh(shap_plot["feature"], shap_plot["mean_abs_shap"],
                            color="#059669", alpha=0.85)
                    ax.set_xlabel("Mean |SHAP Value|")
                    ax.set_title("Top Features by SHAP Importance")
                    ax.grid(True, alpha=0.2)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                if not os.path.exists(perm_path) and not os.path.exists(shap_path):
                    st.info("Run the pipeline to generate importance data.")

            # --- Tab 3: What-If Analysis ---
            with tab3:
                st.subheader("Sensitivity Analysis")
                st.caption("Change one parameter to see how the predicted range responds.")

                wif_col1, wif_col2 = st.columns(2)
                with wif_col1:
                    wif_param = st.selectbox("Parameter to vary", [
                        "battery_capacity_kWh", "top_speed_kmh", "torque_nm",
                        "acceleration_0_100_s", "fast_charging_power_kw_dc",
                        "length_mm", "width_mm", "height_mm",
                        "cargo_volume_l", "towing_capacity_kg", "seats",
                    ])
                with wif_col2:
                    current_val = float(input_dict.get(wif_param, 0))
                    wif_range_pct = st.slider("Variation range (±%)", 10, 100, 50)

                lo = current_val * (1 - wif_range_pct / 100)
                hi = current_val * (1 + wif_range_pct / 100)
                wif_values = np.linspace(max(lo, 0.1), hi, 20)
                wif_preds = []
                for v in wif_values:
                    wif_dict = input_dict.copy()
                    wif_dict[wif_param] = v
                    wif_df = prepare_input(wif_dict, metadata)
                    wif_preds.append(max(0, pipeline.predict(wif_df)[0]))

                fig, ax = plt.subplots(figsize=(9, 4))
                ax.plot(wif_values, wif_preds, "o-", color="#2563eb", linewidth=2, markersize=4)
                ax.axvline(current_val, color="red", linestyle="--", alpha=0.6,
                           label=f"Current: {current_val:.0f}")
                ax.axhline(prediction, color="gray", linestyle=":", alpha=0.4)
                ax.set_xlabel(wif_param.replace("_", " ").title())
                ax.set_ylabel("Predicted Range (km)")
                ax.set_title(f"Sensitivity: {wif_param.replace('_', ' ').title()}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Summary
                range_change = wif_preds[-1] - wif_preds[0]
                direction = "increases" if range_change > 0 else "decreases"
                st.markdown(
                    f'<div class="info-card">'
                    f'When <strong>{wif_param.replace("_", " ")}</strong> varies from '
                    f'{wif_values[0]:.0f} to {wif_values[-1]:.0f}, '
                    f'predicted range {direction} by <strong>{abs(range_change):.0f} km</strong>.'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # --- Tab 4: Methodology ---
            with tab4:
                st.subheader("Model Methodology")
                st.markdown("""
                **Problem:** Predict EV driving range from static vehicle specifications.

                **Approach:**
                1. **Data cleaning** — handle missing values, non-numeric text, zero-variance features
                2. **Feature engineering** — 11 domain-inspired features (battery per seat, footprint, volume proxy, etc.)
                3. **Model selection** — 12+ models compared via 10-fold CV (Model Arena)
                4. **Hyperparameter tuning** — RandomizedSearchCV on top 3 models
                5. **Ensemble investigation** — Voting/Stacking ensembles tested
                6. **Final evaluation** — single untouched holdout test set

                **Leakage Prevention:**
                `efficiency_wh_per_km` is algebraically related to range (`range ≈ kWh × 1000 / efficiency`).
                It is excluded from all model training, feature engineering, and inference.
                It is used only for post-prediction physics sanity checks.

                **Limitations:**
                - Static specifications only — no weather, traffic, HVAC, terrain, or driver behaviour
                - 478-row dataset limits generalization to novel vehicle designs
                - Official rated range (WLTP/NEDC) differs from real-world driving range
                - Missing vehicle weight and aerodynamic drag coefficient (Cd)
                """)

            # Limitation note at bottom
            st.markdown("""
            <div class="limitation-note">
            <strong>Limitations:</strong> This model predicts official rated range from static specifications only.
            It does not account for weather, driving style, terrain, HVAC usage, or battery degradation.
            The dataset lacks aerodynamic drag coefficient (Cd) and vehicle weight, which limits accuracy
            for hyper-aerodynamic or unusually heavy vehicles.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Please check that all inputs are within valid ranges.")

    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #94a3b8; font-size: 0.85rem;">'
        'TECHTRACK 3.0 — Specification-Based EV Range Prediction<br>'
        'MANIT Bhopal &middot; Team Voltra &middot; Competition Submission'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
