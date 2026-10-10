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
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
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

# Default serialisation paths (Ridge kept for backward compatibility)
MODEL_PATH: Path = MODELS_DIR / "ridge_pipeline.pkl"
METADATA_PATH: Path = MODELS_DIR / "ridge_pipeline_metadata.json"

# OLS baseline paths
OLS_MODEL_PATH: Path = MODELS_DIR / "ols_pipeline.pkl"
OLS_METADATA_PATH: Path = MODELS_DIR / "ols_pipeline_metadata.json"


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
        Pass ``LinearRegression()`` for the unregularised OLS baseline.
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

    estimator_name = type(pipeline.named_steps["regressor"]).__name__
    metadata = {
        "n_customers": int(len(customer_df)),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_col": TARGET_COL,
        "estimator": estimator_name,
        "alpha": alpha if estimator is None else None,
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
    }

    logger.success(
        f"[train] Fitted {estimator_name} pipeline on {len(customer_df):,} customers."
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
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found at {model_path}. "
            "Run the training pipeline first: make train"
        )
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    logger.info(f"[load_model] Pipeline loaded from {model_path}")
    return pipeline
