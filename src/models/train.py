"""
train.py — Model training for customer future-revenue prediction.

Responsibility
--------------
Fit a regression pipeline on the customer-level dataset and persist the
fitted pipeline to disk.

Supported estimators
--------------------
LinearRegression : Unregularised OLS baseline (Step 6).
Ridge            : L2-regularised regression (Step 7).
Lasso            : L1-regularised regression (Step 8).
ElasticNet       : Combined L1+L2 regularisation (Step 8).

Pipeline design
----------------
The sklearn Pipeline encapsulates preprocessing + model in one object so that:
* The same fitted transformers are used at training and prediction time.
* The pipeline can be pickled and loaded as a single unit.
* Cross-validation applies preprocessing inside each fold (no leakage).

Preprocessing applied
----------------------
Numeric features   : StandardScaler (essential for fair regularisation comparison).
Categorical feature: OneHotEncoder (country).

Model factory
-------------
Use ``create_model(model_name, **hyperparameters)`` to obtain an sklearn estimator
by name.  Supported names: "linear", "ridge", "lasso", "elasticnet".
"""

import json
from pathlib import Path
import pickle
from typing import Any

import joblib
from loguru import logger
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ecommerce_customer_value_prediction.config import MODELS_DIR

# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------

NUMERIC_FEATURES: list[str] = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_basket_value",
    "unique_products",
    "tenure_days",
    "purchase_rate",
]

CATEGORICAL_FEATURES: list[str] = ["country"]

TARGET_COL: str = "log1p_future_revenue"

# ---------------------------------------------------------------------------
# Serialisation paths
# ---------------------------------------------------------------------------

# Generic / default path (kept for backward compatibility)
MODEL_PATH: Path = MODELS_DIR / "ridge_pipeline.pkl"
METADATA_PATH: Path = MODELS_DIR / "ridge_pipeline_metadata.json"

# Per-model paths
OLS_MODEL_PATH: Path = MODELS_DIR / "ols_pipeline.pkl"
OLS_METADATA_PATH: Path = MODELS_DIR / "ols_pipeline_metadata.json"

RIDGE_MODEL_PATH: Path = MODELS_DIR / "ridge_pipeline.pkl"
RIDGE_METADATA_PATH: Path = MODELS_DIR / "ridge_pipeline_metadata.json"

LASSO_MODEL_PATH: Path = MODELS_DIR / "lasso_pipeline.pkl"
LASSO_METADATA_PATH: Path = MODELS_DIR / "lasso_pipeline_metadata.json"

ELASTICNET_MODEL_PATH: Path = MODELS_DIR / "elasticnet_pipeline.pkl"
ELASTICNET_METADATA_PATH: Path = MODELS_DIR / "elasticnet_pipeline_metadata.json"

FINAL_MODEL_PATH: Path = MODELS_DIR / "customer_value_model.pkl"
FINAL_METADATA_PATH: Path = MODELS_DIR / "customer_value_model_metadata.json"
PACKAGED_MODEL_PATH: Path = MODELS_DIR / "customer_value_model.joblib"
PACKAGED_METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"

# Mapping from model name → (model_path, metadata_path)
MODEL_PATHS: dict[str, tuple[Path, Path]] = {
    "linear": (OLS_MODEL_PATH, OLS_METADATA_PATH),
    "ridge": (RIDGE_MODEL_PATH, RIDGE_METADATA_PATH),
    "lasso": (LASSO_MODEL_PATH, LASSO_METADATA_PATH),
    "elasticnet": (ELASTICNET_MODEL_PATH, ELASTICNET_METADATA_PATH),
}


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = ("linear", "ridge", "lasso", "elasticnet")


def create_model(model_name: str, **hyperparameters: Any) -> BaseEstimator:
    """
    Instantiate an sklearn regression estimator by name.

    Parameters
    ----------
    model_name : str
        One of "linear", "ridge", "lasso", "elasticnet" (case-insensitive).
    **hyperparameters
        Keyword arguments forwarded to the estimator constructor.
        Common parameters:
          - ridge:      alpha (default 1.0)
          - lasso:      alpha (default 1.0), max_iter (default 10_000)
          - elasticnet: alpha (default 1.0), l1_ratio (default 0.5),
                        max_iter (default 10_000)
          - linear:     fit_intercept (default True)

    Returns
    -------
    sklearn estimator (not fitted)

    Examples
    --------
    >>> create_model("ridge", alpha=0.1)
    Ridge(alpha=0.1)
    >>> create_model("lasso", alpha=0.01, max_iter=5000)
    Lasso(alpha=0.01, max_iter=5000)
    >>> create_model("elasticnet", alpha=0.1, l1_ratio=0.5)
    ElasticNet(alpha=0.1, l1_ratio=0.5)
    >>> create_model("linear")
    LinearRegression()
    """
    name = model_name.lower().strip()
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model_name '{model_name}'. Supported: {SUPPORTED_MODELS}")

    # Lasso and ElasticNet use coordinate descent; needs more iterations for
    # convergence on standardised data.  Use 10_000 as a safe default.
    if name == "ridge":
        return Ridge(**hyperparameters)
    if name == "lasso":
        hyperparameters.setdefault("max_iter", 10_000)
        return Lasso(**hyperparameters)
    if name == "elasticnet":
        hyperparameters.setdefault("max_iter", 10_000)
        return ElasticNet(**hyperparameters)
    # "linear"
    return LinearRegression(**hyperparameters)


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------


def build_pipeline(estimator: BaseEstimator | None = None, alpha: float = 1.0) -> Pipeline:
    """
    Construct (but do not fit) the preprocessing + regression pipeline.

    Parameters
    ----------
    estimator : sklearn estimator or None
        The regression model to use.  Defaults to ``Ridge(alpha=alpha)`` when
        ``None``.  Pass ``LinearRegression()`` for the unregularised baseline.
    alpha : float
        Regularisation strength used when ``estimator`` is ``None`` (Ridge default).

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    if estimator is None:
        estimator = Ridge(alpha=alpha)

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", estimator),
        ]
    )
    return pipeline


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train(
    customer_df: pd.DataFrame,
    estimator: BaseEstimator | None = None,
    alpha: float = 1.0,
    random_state: int = 42,
) -> tuple[Pipeline, dict]:
    """
    Fit a regression pipeline on the supplied customer dataset.

    The caller is responsible for any train/test split before calling this
    function if held-out evaluation is required.

    Parameters
    ----------
    customer_df : pd.DataFrame
        Modelling-ready customer dataset (output of ``build_customer_dataset``).
        Must contain all NUMERIC_FEATURES, CATEGORICAL_FEATURES, and TARGET_COL.
    estimator : sklearn estimator or None
        Regression model to use.  Defaults to ``Ridge(alpha=alpha)`` when ``None``.
        Pass ``LinearRegression()`` for the unregularised OLS baseline, or use
        ``create_model("lasso", alpha=0.01)`` etc.
    alpha : float
        Ridge regularisation strength (ignored when ``estimator`` is provided).
    random_state : int
        Kept for API consistency; not used by LinearRegression or Ridge.

    Returns
    -------
    pipeline : fitted sklearn Pipeline
    metadata : dict with training metadata (feature names, shape, estimator name)
    """
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL]
    missing = [c for c in required if c not in customer_df.columns]
    if missing:
        raise ValueError(f"customer_df is missing columns: {missing}")

    X = customer_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = customer_df[TARGET_COL]

    pipeline = build_pipeline(estimator=estimator, alpha=alpha)
    pipeline.fit(X, y)

    estimator_obj = pipeline.named_steps["regressor"]
    estimator_name = type(estimator_obj).__name__

    # Collect hyperparameters that are meaningful for serialisation
    hparams: dict[str, Any] = {}
    if hasattr(estimator_obj, "alpha"):
        hparams["alpha"] = float(estimator_obj.alpha)
    if hasattr(estimator_obj, "l1_ratio"):
        hparams["l1_ratio"] = float(estimator_obj.l1_ratio)
    if hasattr(estimator_obj, "fit_intercept"):
        hparams["fit_intercept"] = bool(estimator_obj.fit_intercept)

    metadata = {
        "n_customers": len(customer_df),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_col": TARGET_COL,
        "estimator": estimator_name,
        "hyperparameters": hparams,
        # legacy key kept for backward compatibility
        "alpha": alpha if estimator is None else hparams.get("alpha"),
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
    }

    logger.success(
        f"[train] Fitted {estimator_name} pipeline on {len(customer_df):,} customers "
        f"| hyperparams={hparams}"
    )
    return pipeline, metadata


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(
    pipeline: Pipeline,
    metadata: dict,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
) -> None:
    """
    Persist the fitted pipeline and its metadata to disk.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    metadata : dict  (from ``train``)
    model_path : Path  where to write the ``.pkl`` file
    metadata_path : Path  where to write the ``.json`` metadata
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.success(f"[save_model] Pipeline saved → {model_path}")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.success(f"[save_model] Metadata saved → {metadata_path}")


def load_model(model_path: Path = MODEL_PATH) -> Pipeline:
    """
    Load a previously fitted pipeline from disk.

    Parameters
    ----------
    model_path : Path

    Returns
    -------
    fitted sklearn Pipeline
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found at {model_path}. Run the training pipeline first: make train"
        )
    if model_path.suffix == ".joblib":
        pipeline = joblib.load(model_path)
    else:
        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"{model_path} does not contain a sklearn Pipeline.")
    logger.info(f"[load_model] Pipeline loaded from {model_path}")
    return pipeline
