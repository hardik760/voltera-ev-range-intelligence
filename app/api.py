"""
VOLTERA — EV Range Prediction API
FastAPI backend for TECHTRACK 3.0

Endpoints:
  GET  /health  — Health check
  POST /predict — Predict EV range from specifications

Uses the SAME saved pipeline as the Streamlit app.
No duplicate model logic. No efficiency dependency.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Project imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "final_ev_range_pipeline.joblib")
META_PATH = os.path.join(PROJECT_ROOT, "models", "pipeline_metadata.json")

# Load model globally
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model artifact not found at {MODEL_PATH}. Run pipeline first.")

pipeline = joblib.load(MODEL_PATH)

# Load metadata
metadata = {}
if os.path.exists(META_PATH):
    with open(META_PATH) as f:
        metadata = json.load(f)

app = FastAPI(
    title="VOLTERA — EV Range Prediction API",
    description="TECHTRACK 3.0 — Predict EV driving range from specifications",
    version="2.0.0",
)


class EVSpecification(BaseModel):
    battery_capacity_kWh: float = Field(..., gt=0, description="Battery capacity in kWh")
    acceleration_0_100_s: float = Field(..., gt=0, description="0-100 km/h time in seconds")
    top_speed_kmh: int = Field(..., gt=0, description="Maximum speed in km/h")
    length_mm: int = Field(..., gt=0, description="Vehicle length in mm")
    width_mm: int = Field(..., gt=0, description="Vehicle width in mm")
    height_mm: int = Field(..., gt=0, description="Vehicle height in mm")
    cargo_volume_l: float = Field(..., ge=0, description="Cargo volume in litres")
    seats: int = Field(..., gt=0, description="Number of seats")
    torque_nm: float = Field(..., gt=0, description="Maximum torque in Nm")
    fast_charging_power_kw_dc: float = Field(..., ge=0, description="DC fast charging power in kW")
    towing_capacity_kg: float = Field(..., ge=0, description="Towing capacity in kg (0 if none)")
    drivetrain: str = Field(..., description="Drivetrain type: AWD, FWD, or RWD")
    car_body_type: str = Field(..., description="Body type (e.g., SUV, Sedan)")
    segment: str = Field(..., description="Market segment (e.g., C - Medium)")

    class Config:
        json_schema_extra = {
            "example": {
                "battery_capacity_kWh": 75.0,
                "acceleration_0_100_s": 4.5,
                "top_speed_kmh": 230,
                "length_mm": 4694,
                "width_mm": 1849,
                "height_mm": 1443,
                "cargo_volume_l": 425.0,
                "seats": 5,
                "torque_nm": 490.0,
                "fast_charging_power_kw_dc": 250.0,
                "towing_capacity_kg": 1000.0,
                "drivetrain": "AWD",
                "car_body_type": "Sedan",
                "segment": "D - Large",
            }
        }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": metadata.get("final_model_name", "unknown"),
    }


@app.post("/predict")
def predict_range(spec: EVSpecification):
    """Predict EV range from specifications."""
    try:
        input_data = pd.DataFrame([spec.model_dump()])

        # Add engineered features using shared module
        from src.feature_engineering import engineer_features
        input_data = engineer_features(input_data)

        predicted_range = float(pipeline.predict(input_data)[0])
        predicted_range = max(0.0, predicted_range)

        # Post-prediction physics sanity check
        implied_wh_km = (spec.battery_capacity_kWh * 1000) / max(predicted_range, 1.0)
        is_plausible = 80 <= implied_wh_km <= 400

        return {
            "predicted_range_km": round(predicted_range, 2),
            "predicted_range_miles": round(predicted_range * 0.621371, 2),
            "physics_sanity_check": {
                "implied_efficiency_wh_km": round(implied_wh_km, 2),
                "is_plausible": is_plausible,
                "note": "This check validates plausibility after prediction; "
                        "efficiency is not a model input.",
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
