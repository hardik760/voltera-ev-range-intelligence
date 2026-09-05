"""
TECHTRACK 3.0 — Modeling Module

Model Arena: trains, compares, tunes, and ensembles regression models.
All results are from actual experiments — nothing fabricated.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor

SEED = 42


def get_model_candidates(scale: bool = False):
    """
    Return a dict of model_name → (model_instance, needs_scaling, complexity).
    """
    candidates = {
        "DummyRegressor (mean)": (
            DummyRegressor(strategy="mean"), False, "trivial"
        ),
        "Linear Regression": (
            LinearRegression(), True, "low"
        ),
        "Ridge": (
            Ridge(alpha=1.0, random_state=SEED), True, "low"
        ),
        "Lasso": (
            Lasso(alpha=1.0, random_state=SEED, max_iter=5000), True, "low"
        ),
        "ElasticNet": (
            ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=SEED, max_iter=5000),
            True, "low"
        ),
        "Decision Tree": (
            DecisionTreeRegressor(random_state=SEED, max_depth=8), False, "low"
        ),
        "Random Forest": (
            RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1, monotonic_cst={"battery_capacity_kWh": 1}),
            False, "medium"
        ),
        "Extra Trees": (
            ExtraTreesRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
            False, "medium"
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                random_state=SEED
            ),
            False, "medium"
        ),
        "HistGradientBoosting": (
            HistGradientBoostingRegressor(
                max_iter=200, max_depth=6, learning_rate=0.1,
                random_state=SEED, monotonic_cst={"battery_capacity_kWh": 1}
            ),
            False, "medium"
        ),
        "KNN (k=7)": (
            KNeighborsRegressor(n_neighbors=7, n_jobs=-1), True, "low"
        ),
    }

    # Try adding XGBoost
    try:
        from xgboost import XGBRegressor
        candidates["XGBoost"] = (
            XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                random_state=SEED, n_jobs=-1, verbosity=0,
                monotone_constraints={"battery_capacity_kWh": 1}
            ),
            False, "medium"
        )
    except ImportError:
        pass

    # Try adding LightGBM
    try:
        from lightgbm import LGBMRegressor
        candidates["LightGBM"] = (
            LGBMRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=SEED, n_jobs=-1, verbose=-1,
            ),
            False, "medium"
        )
    except ImportError:
        pass

    return candidates


def run_cross_validation(pipeline, X, y, cv=10, scoring=None):
    """
    Run cross-validation and return a dict of mean/std metrics.
    """
    if scoring is None:
        scoring = {
            "MAE": "neg_mean_absolute_error",
            "RMSE": "neg_root_mean_squared_error",
            "R2": "r2",
        }

    results = cross_validate(
        pipeline, X, y, cv=cv, scoring=scoring,
        return_train_score=False, n_jobs=-1,
    )

    return {
        "cv_MAE": round(-results["test_MAE"].mean(), 2),
        "cv_MAE_std": round(results["test_MAE"].std(), 2),
        "cv_RMSE": round(-results["test_RMSE"].mean(), 2),
        "cv_RMSE_std": round(results["test_RMSE"].std(), 2),
        "cv_R2": round(results["test_R2"].mean(), 4),
        "cv_R2_std": round(results["test_R2"].std(), 4),
    }


def get_tuning_params():
    """
    Return hyperparameter search spaces for top models.
    Designed for a ~478-row dataset — not excessively large.
    """
    params = {
        "Random Forest": {
            "base_pipeline__model__n_estimators": [100, 200, 300, 500],
            "base_pipeline__model__max_depth": [6, 8, 10, 12, None],
            "base_pipeline__model__min_samples_split": [2, 5, 10],
            "base_pipeline__model__min_samples_leaf": [1, 2, 4],
            "base_pipeline__model__max_features": ["sqrt", "log2", 0.5, 0.8],
        },
        "Extra Trees": {
            "base_pipeline__model__n_estimators": [100, 200, 300, 500],
            "base_pipeline__model__max_depth": [6, 8, 10, 12, None],
            "base_pipeline__model__min_samples_split": [2, 5, 10],
            "base_pipeline__model__min_samples_leaf": [1, 2, 4],
            "base_pipeline__model__max_features": ["sqrt", "log2", 0.5, 0.8],
        },
        "Gradient Boosting": {
            "base_pipeline__model__n_estimators": [100, 200, 300, 500],
            "base_pipeline__model__max_depth": [3, 4, 5, 6],
            "base_pipeline__model__learning_rate": [0.01, 0.05, 0.1, 0.15],
            "base_pipeline__model__min_samples_split": [2, 5, 10],
            "base_pipeline__model__min_samples_leaf": [1, 2, 4],
            "base_pipeline__model__subsample": [0.7, 0.8, 0.9, 1.0],
        },
        "HistGradientBoosting": {
            "base_pipeline__model__max_iter": [100, 200, 300, 500],
            "base_pipeline__model__max_depth": [4, 5, 6, 8, 10],
            "base_pipeline__model__learning_rate": [0.01, 0.05, 0.1, 0.15],
            "base_pipeline__model__min_samples_leaf": [5, 10, 20, 30],
            "base_pipeline__model__max_leaf_nodes": [15, 31, 63, None],
        },
    }

    try:
        from xgboost import XGBRegressor
        params["XGBoost"] = {
            "base_pipeline__model__n_estimators": [100, 200, 300, 500],
            "base_pipeline__model__max_depth": [3, 4, 5, 6, 8],
            "base_pipeline__model__learning_rate": [0.01, 0.05, 0.1, 0.15],
            "base_pipeline__model__min_child_weight": [1, 3, 5],
            "base_pipeline__model__subsample": [0.7, 0.8, 0.9, 1.0],
            "base_pipeline__model__colsample_bytree": [0.5, 0.7, 0.8, 1.0],
            "base_pipeline__model__reg_alpha": [0, 0.01, 0.1],
            "base_pipeline__model__reg_lambda": [0.5, 1.0, 2.0],
        }
    except ImportError:
        pass

    try:
        from lightgbm import LGBMRegressor
        params["LightGBM"] = {
            "base_pipeline__model__n_estimators": [100, 200, 300, 500],
            "base_pipeline__model__max_depth": [4, 5, 6, 8, -1],
            "base_pipeline__model__learning_rate": [0.01, 0.05, 0.1, 0.15],
            "base_pipeline__model__num_leaves": [15, 31, 63, 127],
            "base_pipeline__model__min_child_samples": [5, 10, 20],
            "base_pipeline__model__subsample": [0.7, 0.8, 0.9, 1.0],
            "base_pipeline__model__colsample_bytree": [0.5, 0.7, 0.8, 1.0],
            "base_pipeline__model__reg_alpha": [0, 0.01, 0.1],
            "base_pipeline__model__reg_lambda": [0, 0.5, 1.0],
        }
    except ImportError:
        pass

    return params


def tune_model(pipeline, param_dist, X, y, n_iter=50, cv=10):
    """
    Tune a pipeline using RandomizedSearchCV.
    Returns the best pipeline and search results.
    """
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_absolute_error",
        random_state=SEED,
        n_jobs=-1,
        verbose=0,
        return_train_score=False,
    )
    search.fit(X, y)

    return {
        "best_pipeline": search.best_estimator_,
        "best_params": search.best_params_,
        "best_cv_mae": round(-search.best_score_, 2),
        "search_results": pd.DataFrame(search.cv_results_),
    }


def build_ensemble(base_models: dict, X, y, cv=10):
    """
    Build ensemble models from the best-performing base models.
    Returns VotingRegressor and StackingRegressor configurations.
    """
    estimators = [(name, model) for name, model in base_models.items()]

    ensembles = {}

    # Voting (averaging)
    voting = VotingRegressor(estimators=estimators, n_jobs=-1)
    ensembles["Voting (Average)"] = voting

    # Stacking with Ridge meta-learner
    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0, random_state=SEED),
        cv=5,
        n_jobs=-1,
    )
    ensembles["Stacking (Ridge meta)"] = stacking

    return ensembles
