"""
Multivariate interpretability using LASSO regression and Random Forest.

Explains deep embedding axes (e.g., t-SNE components) using classical phenotypic properties.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

from ..._logging import get_logger

logger = get_logger(__name__)


def compute_lasso_interpretability(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    cv: int = 5,
    seed: int | None = None,
) -> dict:
    """Explain a target variable y using a LASSO model on features x.

    To ensure comparable coefficients and correct mathematical representation,
    both features and target y are standardized internally within this function to mean=0, var=1.

    Args:
        x: Feature matrix, shape (n_samples, n_features).
        y: Target variable, shape (n_samples,).
        feature_names: Names of the features in x.
        cv: Number of cross-validation folds for LassoCV.
        seed: Random seed for reproducibility (passed to scikit-learn as ``random_state``).

    Returns:
        Dict with:
            - r2: Cross-validated R^2 score of the model.
            - drivers: List of dicts with 'feature' and 'weight' for non-zero coefficients,
              sorted by absolute weight descending.
            - intercept: Model intercept (should be close to 0 as data is standardized).
            - n_samples: Number of samples used.
            - n_features: Number of input features.
    """
    _, n_features = x.shape

    # Ensure inputs are finite
    mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x_clean = x[mask]
    y_clean = y[mask]

    if len(y_clean) < cv + 1:
        logger.warning(
            "Insufficient samples for LassoCV (got %d, need at least %d).", len(y_clean), cv + 1
        )
        return {
            "r2": 0.0,
            "drivers": [],
            "intercept": 0.0,
            "n_samples": len(y_clean),
            "n_features": n_features,
        }

    # Standardize features and y for valid, comparable LASSO coefficients
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    x_scaled = scaler_x.fit_transform(x_clean)
    y_scaled = scaler_y.fit_transform(y_clean.reshape(-1, 1)).ravel()

    # LassoCV automatically finds the best alpha using cross-validation
    model = LassoCV(cv=cv, random_state=seed, max_iter=10000)

    # Compute cross-validated R2 score robustly using cross_val_predict
    # This prevents using the training R2 which might be overfitted
    cv_n_jobs = 1 if seed is not None else -1
    y_cv_pred = cross_val_predict(model, x_scaled, y_scaled, cv=cv, n_jobs=cv_n_jobs)
    cv_r2 = float(r2_score(y_scaled, y_cv_pred))

    # Fit final model on all data to get final coefficients
    model.fit(x_scaled, y_scaled)

    coefs = model.coef_
    intercept = float(model.intercept_)

    drivers = []
    for name, weight in zip(feature_names, coefs, strict=True):
        if weight != 0:
            drivers.append({"feature": name, "weight": float(weight)})

    # Sort by absolute weight descending
    drivers.sort(key=lambda x: abs(x["weight"]), reverse=True)

    return {
        "r2": cv_r2,
        "drivers": drivers,
        "intercept": intercept,
        "n_samples": len(y_clean),
        "n_features": int(n_features),
    }


def compute_rf_interpretability(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_estimators: int = 100,
    seed: int | None = None,
) -> dict:
    """Explain a target variable y using a Random Forest model on features x.

    Unlike LASSO, Random Forest captures non-linear relationships. Weights in 'drivers'
    correspond to Gini feature importances (always non-negative).
    The R2 score is calculated out-of-bag (OOB) to give a generalized estimate.

    Args:
        x: Feature matrix, shape (n_samples, n_features).
        y: Target variable, shape (n_samples,).
        feature_names: Names of the features in x.
        n_estimators: Number of trees in the forest.
        seed: Random seed for reproducibility (passed to scikit-learn as ``random_state``).

    Returns:
        Dict with:
            - r2: OOB R^2 score of the model generalization.
            - drivers: List of dicts with 'feature' and 'weight' for importances,
              sorted by weight descending.
            - n_samples: Number of samples used.
            - n_features: Number of input features.
    """
    _, n_features = x.shape

    # Ensure inputs are finite
    mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x_clean = x[mask]
    y_clean = y[mask]

    if len(y_clean) < 2:
        logger.warning(
            "Insufficient samples for Random Forest (got %d, need at least 2).", len(y_clean)
        )
        return {"r2": 0.0, "drivers": [], "n_samples": len(y_clean), "n_features": n_features}

    # Use out-of-bag score to prevent inflated R2 values from Random Forest overfitting
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=1 if seed is not None else -1,
        oob_score=True,
    )

    # We do not strictly need scaling for Random Forest, but we can fit it directly
    model.fit(x_clean, y_clean)

    # Get OOB R2 score instead of standard training score
    r2 = float(model.oob_score_)
    importances = model.feature_importances_

    drivers = []
    for name, weight in zip(feature_names, importances, strict=True):
        if weight > 0:
            drivers.append({"feature": name, "weight": float(weight)})

    # Sort by importance weight descending
    drivers.sort(key=lambda x: x["weight"], reverse=True)

    return {
        "r2": r2,
        "drivers": drivers,
        "n_samples": len(y_clean),
        "n_features": int(n_features),
    }
