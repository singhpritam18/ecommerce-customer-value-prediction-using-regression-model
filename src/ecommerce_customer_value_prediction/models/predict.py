"""
predict.py — Inference using a fitted Ridge pipeline.

Responsibility
--------------
Load a serialised pipeline from disk and generate predictions for a
customer-level feature DataFrame.

Predictions are returned in both log-space (model output) and revenue-space
(expm1 applied) so callers can choose the most useful representation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.pipeline import Pipeline

from ecommerce_customer_value_prediction.models.train import (
    CATEGORICAL_FEATURES,
    MODEL_PATH,
    NUMERIC_FEATURES,
    load_model,
)


def predict(
    customer_df: pd.DataFrame,
    pipeline: Pipeline | None = None,
    model_path: Path = MODEL_PATH,
) -> pd.DataFrame:
    """
    Generate future-revenue predictions for a customer feature DataFrame.

    Parameters
    ----------
    customer_df : pd.DataFrame
        Must contain all NUMERIC_FEATURES and CATEGORICAL_FEATURES columns.
        Typically the output of ``build_rfm_features`` (features only,
        no target column required).
    pipeline : fitted sklearn Pipeline, optional
        If ``None``, the pipeline is loaded from *model_path*.
    model_path : Path
        Used only when *pipeline* is ``None``.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with two additional columns appended:
        * ``pred_log1p_future_revenue`` — raw model output (log-space).
        * ``pred_future_revenue``       — expm1-transformed GBP estimate.
    """
    if pipeline is None:
        pipeline = load_model(model_path)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = [c for c in feature_cols if c not in customer_df.columns]
    if missing:
        raise ValueError(f"customer_df is missing feature columns: {missing}")

    X = customer_df[feature_cols]
    log_preds = pipeline.predict(X)

    result = customer_df.copy()
    result["pred_log1p_future_revenue"] = log_preds
    result["pred_future_revenue"] = np.expm1(log_preds)

    logger.info(
        f"[predict] Generated predictions for {len(result):,} customers. "
        f"Median predicted revenue: GBP {np.median(result['pred_future_revenue']):.2f}"
    )
    return result


# Re-export load_model so callers can do:
#   from ecommerce_customer_value_prediction.models.predict import load_model
__all__ = ["predict", "load_model"]
