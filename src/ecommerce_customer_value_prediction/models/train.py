"""
train.py — Model training for customer future-revenue prediction.

Responsibility
--------------
Fit a regularised regression pipeline on the customer-level dataset and
persist the fitted pipeline to disk.

Algorithm choice
-----------------
Ridge regression is the baseline estimator.  Reasons:
* The target (``log1p_future_revenue``) is continuous and approximately
  normal after log-transform.
* Ridge handles multicollinearity between RFM features (e.g., frequency
  and monetary are correlated).
* It is fast, interpretable, and provides a strong baseline against which
  more complex models can be benchmarked.

Pipeline design
----------------
The sklearn Pipeline encapsulates preprocessing + model in one object so that:
* The same fitted transformers are used at training and prediction time.
* The pipeline can be pickled and loaded as a single unit.
* Cross-validation applies preprocessing inside each fold (no leakage).

Preprocessing applied
----------------------
Numeric features   : StandardScaler (Ridge is sensitive to feature scale).
Categorical feature: OneHotEncoder (country).
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
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

# Default serialisation path
MODEL_PATH: Path = MODELS_DIR / "ridge_pipeline.pkl"
METADATA_PATH: Path = MODELS_DIR / "ridge_pipeline_metadata.json"


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------


def build_pipeline(alpha: float = 1.0) -> Pipeline:
    """
    Construct (but do not fit) the preprocessing + Ridge pipeline.

    Parameters
    ----------
    alpha : float
        Ridge regularisation strength. Larger values → stronger regularisation.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
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
            ("regressor", Ridge(alpha=alpha)),
        ]
    )
    return pipeline


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train(
    customer_df: pd.DataFrame,
    alpha: float = 1.0,
    random_state: int = 42,
) -> tuple[Pipeline, dict]:
    """
    Fit the Ridge pipeline on *all* available labelled customers.

    The caller is responsible for any train/test split before calling this
    function if held-out evaluation is required.  The ``pipelines`` module
    performs the full train/evaluate workflow.

    Parameters
    ----------
    customer_df : pd.DataFrame
        Modelling-ready customer dataset (output of ``build_customer_dataset``).
        Must contain all NUMERIC_FEATURES, CATEGORICAL_FEATURES, and TARGET_COL.
    alpha : float
        Ridge regularisation strength.
    random_state : int
        Passed to Ridge (no effect on Ridge itself, but kept for consistency).

    Returns
    -------
    pipeline : fitted sklearn Pipeline
    metadata : dict with training metadata (feature names, shape, alpha)
    """
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL]
    missing = [c for c in required if c not in customer_df.columns]
    if missing:
        raise ValueError(f"customer_df is missing columns: {missing}")

    X = customer_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = customer_df[TARGET_COL]

    pipeline = build_pipeline(alpha=alpha)
    pipeline.fit(X, y)

    metadata = {
        "n_customers": int(len(customer_df)),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_col": TARGET_COL,
        "alpha": alpha,
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
    }

    logger.success(
        f"[train] Fitted Ridge pipeline on {len(customer_df):,} customers "
        f"(alpha={alpha})."
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
