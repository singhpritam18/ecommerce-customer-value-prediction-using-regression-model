"""
evaluate.py — Evaluation metrics for the revenue regression model.

Responsibility
--------------
Compute and persist evaluation metrics for a fitted pipeline on a held-out
test set.  All metrics are reported both in log-space (the modelled target)
and in original revenue space (GBP) for business interpretability.

Metrics reported
-----------------
Log-space  (what the model optimises):
    rmse_log    : Root Mean Squared Error of log1p_future_revenue predictions.
    mae_log     : Mean Absolute Error of log1p_future_revenue predictions.
    r2_log      : R² of log1p_future_revenue predictions.

Revenue-space (business interpretation):
    rmse_gbp    : RMSE after applying expm1 to both predictions and actuals.
    mae_gbp     : MAE in GBP.
    median_ae_gbp : Median Absolute Error in GBP (robust to outliers).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from ecommerce_customer_value_prediction.config import METRICS_DIR
from ecommerce_customer_value_prediction.models.train import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COL,
)

METRICS_PATH: Path = METRICS_DIR / "evaluation_metrics.json"


def evaluate(
    pipeline: Pipeline,
    test_df: pd.DataFrame,
    metrics_path: Path = METRICS_PATH,
    save: bool = True,
) -> dict:
    """
    Evaluate a fitted pipeline on a held-out test set.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    test_df : pd.DataFrame
        Customer-level test set with features and ``log1p_future_revenue``.
    metrics_path : Path
        Where to save the metrics JSON file.
    save : bool
        If ``True``, write metrics to *metrics_path*.

    Returns
    -------
    dict
        All computed metrics.
    """
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL]
    missing = [c for c in required if c not in test_df.columns]
    if missing:
        raise ValueError(f"test_df is missing columns: {missing}")

    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_log_true = test_df[TARGET_COL].values

    y_log_pred = pipeline.predict(X_test)

    # Log-space metrics
    rmse_log = float(np.sqrt(mean_squared_error(y_log_true, y_log_pred)))
    mae_log = float(mean_absolute_error(y_log_true, y_log_pred))
    r2_log = float(r2_score(y_log_true, y_log_pred))

    # Revenue-space metrics (inverse of log1p)
    y_rev_true = np.expm1(y_log_true)
    y_rev_pred = np.expm1(y_log_pred)

    rmse_gbp = float(np.sqrt(mean_squared_error(y_rev_true, y_rev_pred)))
    mae_gbp = float(mean_absolute_error(y_rev_true, y_rev_pred))
    median_ae_gbp = float(np.median(np.abs(y_rev_true - y_rev_pred)))

    metrics = {
        "n_test_customers": int(len(test_df)),
        "rmse_log": round(rmse_log, 6),
        "mae_log": round(mae_log, 6),
        "r2_log": round(r2_log, 6),
        "rmse_gbp": round(rmse_gbp, 2),
        "mae_gbp": round(mae_gbp, 2),
        "median_ae_gbp": round(median_ae_gbp, 2),
    }

    logger.info(
        f"[evaluate] n={metrics['n_test_customers']:,} | "
        f"R²={r2_log:.4f} | RMSE_log={rmse_log:.4f} | MAE_GBP={mae_gbp:.2f}"
    )

    if save:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.success(f"[evaluate] Metrics saved → {metrics_path}")

    return metrics
