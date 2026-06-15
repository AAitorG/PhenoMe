"""
Multivariate interpretability using LASSO regression and Random Forest.

Explains deep embedding axes (e.g., t-SNE components) using classical phenotypic properties.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..._logging import get_logger

logger = get_logger(__name__)


class _MetadataStratifiedKFold(BaseCrossValidator):
    """K-fold CV stratified on metadata labels, not the regression target."""

    def __init__(
        self,
        labels: np.ndarray,
        n_splits: int = 5,
        *,
        shuffle: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.labels = np.asarray(labels)
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        groups: Any | None = None,
    ) -> int:
        return self.n_splits

    def split(
        self,
        x: np.ndarray,
        y: np.ndarray | None = None,
        groups: Any | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(x)
        if n != len(self.labels):
            raise ValueError(
                f"_MetadataStratifiedKFold expects the full sample matrix "
                f"({len(self.labels)} rows) but received {n} rows. "
                "Use integer cv for nested LassoCV on training folds."
            )
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )
        yield from skf.split(x, self.labels)


def prepare_interpretability_inputs(
    y: np.ndarray,
    dr_indices: list[int],
    matrix: np.ndarray,
    prop_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, list[int]] | None:
    """Align DR target to property rows and drop non-finite samples.

    Returns ``(matrix, y, prop_indices)`` or ``None`` when no valid rows remain.
    """
    from .combined_features import align_rows_by_global_indices

    if len(matrix) < len(y):
        y = align_rows_by_global_indices(dr_indices, y, prop_indices)

    finite_mask = np.isfinite(y) & np.isfinite(matrix).all(axis=1)
    if not finite_mask.all():
        matrix = matrix[finite_mask]
        y = y[finite_mask]
        prop_indices = [idx for idx, keep in zip(prop_indices, finite_mask, strict=True) if keep]

    if len(matrix) == 0:
        return None
    return matrix, y, prop_indices


def _min_samples_for_lasso_cv(cv: int) -> int:
    """Minimum n so the smallest outer-CV training fold can run inner LassoCV."""
    if cv <= 1:
        return 2
    # Outer train size is roughly n*(cv-1)/cv; inner LassoCV needs at least cv+1 rows.
    return max(cv + 1, int(np.ceil(cv * (cv + 1) / (cv - 1))))


def _prepare_stratify_labels(stratify: np.ndarray) -> np.ndarray:
    """Coerce metadata values to string class labels for StratifiedKFold."""
    out = np.empty(len(stratify), dtype=object)
    for i, value in enumerate(stratify):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            out[i] = "__missing__"
        else:
            out[i] = str(value)
    return out


def _resolve_lasso_cv(
    cv: int,
    seed: int | None,
    stratify: np.ndarray | None,
    n_samples: int,
) -> tuple[int | _MetadataStratifiedKFold | KFold, str]:
    """Return (outer_cv_splitter, cv_strategy) for cross_val_predict.

    Stratification balances metadata classes across outer folds only. Inner
    LassoCV always uses unstratified folds (integer ``cv``) so subset training
    matrices do not misalign metadata labels.
    """
    if n_samples < _min_samples_for_lasso_cv(cv):
        return cv, "insufficient_samples"

    if stratify is None:
        if seed is None:
            return cv, "kfold"
        return KFold(n_splits=cv, shuffle=True, random_state=seed), "kfold"

    labels = _prepare_stratify_labels(stratify)
    _, counts = np.unique(labels, return_counts=True)
    min_class = int(counts.min())
    if min_class < cv:
        logger.warning(
            "Cannot stratify LASSO CV: smallest class has %d sample(s) but cv=%d. "
            "Using unstratified KFold.",
            min_class,
            cv,
        )
        if seed is None:
            return cv, "stratified_fallback"
        return KFold(n_splits=cv, shuffle=True, random_state=seed), "stratified_fallback"

    return (
        _MetadataStratifiedKFold(labels, n_splits=cv, shuffle=True, random_state=seed),
        "stratified",
    )


def compute_lasso_interpretability(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    cv: int = 5,
    seed: int | None = None,
    stratify: np.ndarray | None = None,
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
        stratify: Optional per-sample metadata labels (same length as ``x``). When provided
            and every class has at least ``cv`` samples, folds are stratified on these
            labels rather than the regression target. Balances treatment/batch across folds
            for ``lambda`` selection; does not correct global embedding target leakage.

    Returns:
        Dict with:
            - r2: Fold-wise R^2 on the fixed target ``y`` (see Notes).
            - drivers: List of dicts with 'feature' and 'weight' for non-zero coefficients,
              sorted by absolute weight descending.
            - intercept: Model intercept (should be close to 0 as data is standardized).
            - n_samples: Number of samples used.
            - n_features: Number of input features.
            - cv_strategy: ``"stratified"``, ``"kfold"``, ``"stratified_fallback"``, or
              ``"insufficient_samples"``.

    Notes:
        Cross-validation here serves two internal purposes: (1) ``LassoCV`` selects the
        regularization strength on training folds with fold-wise scaling to avoid feature
        leakage, and (2) ``cross_val_predict`` estimates how well properties linearly track
        the supplied target axis on the same samples.

        When ``y`` comes from a globally fit transductive embedding (e.g. t-SNE or UMAP
        coordinates computed on the full dataset), validation-fold targets already depend on
        training samples through the embedding step. The returned ``r2`` is therefore a
        **descriptive on-axis fit score**, not an unbiased out-of-sample generalization
        metric. Treat ``drivers`` as exploratory associations with the chosen axis; for
        rigorous generalization, hold out batches/plates before embedding or use nested CV
        with out-of-sample projection (feasible for PCA, not standard t-SNE/UMAP).
    """
    _, n_features = x.shape

    # Ensure inputs are finite
    mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x_clean = x[mask]
    y_clean = y[mask]
    stratify_clean = stratify[mask] if stratify is not None else None

    min_samples = _min_samples_for_lasso_cv(cv)
    if len(y_clean) < min_samples:
        logger.warning(
            "Insufficient samples for LassoCV (got %d, need at least %d for cv=%d).",
            len(y_clean),
            min_samples,
            cv,
        )
        return {
            "r2": 0.0,
            "drivers": [],
            "intercept": 0.0,
            "n_samples": len(y_clean),
            "n_features": n_features,
            "cv_strategy": "insufficient_samples",
        }

    outer_cv, cv_strategy = _resolve_lasso_cv(cv, seed, stratify_clean, len(y_clean))
    if cv_strategy == "insufficient_samples":
        return {
            "r2": 0.0,
            "drivers": [],
            "intercept": 0.0,
            "n_samples": len(y_clean),
            "n_features": n_features,
            "cv_strategy": "insufficient_samples",
        }

    # Inner LassoCV: always unstratified integer folds on each outer training subset.
    inner_cv = cv

    # Standardize features and y for valid, comparable LASSO coefficients
    from sklearn.preprocessing import StandardScaler

    # Standardize inside the pipeline so scaling is fit on training folds only.
    base_model = make_pipeline(
        StandardScaler(),
        LassoCV(cv=inner_cv, random_state=seed, max_iter=10000),
    )
    model = TransformedTargetRegressor(regressor=base_model, transformer=StandardScaler())

    # Outer fold-wise R² on the fixed target y (see docstring Notes on global DR leakage).
    cv_n_jobs = 1 if seed is not None else -1
    y_cv_pred = cross_val_predict(model, x_clean, y_clean, cv=outer_cv, n_jobs=cv_n_jobs)
    cv_r2 = float(r2_score(y_clean, y_cv_pred))

    # Fit final model on all data to get final coefficients
    model.fit(x_clean, y_clean)

    # Extract coefficients from the fitted LassoCV model inside the pipeline
    lasso_model = model.regressor_.named_steps["lassocv"]
    coefs = lasso_model.coef_
    intercept = float(lasso_model.intercept_)

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
        "cv_strategy": cv_strategy,
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
    correspond to Gini feature importances (always non-negative). Features are
    standardized before fitting so importances are comparable across property scales.
    The R2 score is calculated out-of-bag (OOB) to give a generalized estimate on the
    fixed target ``y`` supplied by the caller.

    Notes:
        When ``y`` is a globally fit t-SNE/UMAP axis (typical in
        ``compute_multivariate_interpretability``), OOB R² still shares the same
        descriptive-on-axis interpretation as LASSO; see ``compute_lasso_interpretability``.

    Args:
        x: Feature matrix, shape (n_samples, n_features).
        y: Target variable, shape (n_samples,).
        feature_names: Names of the features in x.
        n_estimators: Number of trees in the forest.
        seed: Random seed for reproducibility (passed to scikit-learn as ``random_state``).

    Returns:
        Dict with:
            - r2: OOB R^2 on the fixed target ``y`` (descriptive when ``y`` is a global DR axis).
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

    # Standardize features so Gini importances are not dominated by property scale
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_clean)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=1 if seed is not None else -1,
        oob_score=True,
    )

    model.fit(x_scaled, y_clean)

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
