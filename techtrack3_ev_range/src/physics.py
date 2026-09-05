import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class PhysicsResidualRegressor(BaseEstimator, RegressorMixin):
    """
    A meta-estimator that implements Physics-Informed Residual Modeling.
    It predicts the deviation from a physics baseline, rather than raw range.
    """
    def __init__(self, base_pipeline):
        self.base_pipeline = base_pipeline

    def fit(self, X, y):
        # Compute medians strictly on the training data X provided to fit()
        self.segment_medians_ = X.groupby("segment")["efficiency_wh_per_km"].median().to_dict()
        self.global_median_ = X["efficiency_wh_per_km"].median()
        
        baseline = self._compute_baseline(X)
        y_residual = y - baseline
        # Train the underlying ML pipeline on the RESIDUAL error of the physics baseline
        self.base_pipeline.fit(X, y_residual)
        return self

    def _compute_baseline(self, X):
        """Computes the physical baseline: (battery_kWh * 1000) / segment_efficiency_Wh_km"""
        baselines = []
        for _, row in X.iterrows():
            eff = self.segment_medians_.get(row.get('segment'), self.global_median_)
            batt = row.get('battery_capacity_kWh', 0)
            
            # Fallback for missing/zero battery or efficiency
            if pd.isna(batt) or batt <= 0 or pd.isna(eff) or eff <= 0:
                baselines.append(0.0)
            else:
                baselines.append((batt * 1000) / eff)
                
        return np.array(baselines)

    def predict(self, X):
        baseline = self._compute_baseline(X)
        residual = self.base_pipeline.predict(X)
        # Final range is physics baseline + ML residual
        return baseline + residual

    def get_base_estimator(self):
        """Helper to extract the actual tree estimator for SHAP"""
        return self.base_pipeline.named_steps["model"]
