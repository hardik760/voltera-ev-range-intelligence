# This module previously contained PhysicsResidualRegressor which used
# efficiency_wh_per_km during fit() and predict(), creating target leakage.
#
# REMOVED: PhysicsResidualRegressor
# REASON: Used efficiency_wh_per_km (forbidden feature) to construct a
#         physics baseline and trained the ML model on residuals.
#         This is NOT allowed — the model must predict range_km DIRECTLY.
#
# Post-prediction physics sanity checking remains in src/evaluation.py.
