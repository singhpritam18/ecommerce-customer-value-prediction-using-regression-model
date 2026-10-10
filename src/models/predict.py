"""Validated customer-level inference; loading and prediction never fit a model."""

import argparse
import hashlib
import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from ecommerce_customer_value_prediction.models.train import (
    PACKAGED_MODEL_PATH,
)
from ecommerce_customer_value_prediction.models.train import (
    load_model as _load_model,
)

OUTPUT_COLUMNS = ["pred_log1p_future_revenue", "pred_future_revenue"]


def load_model(model_path: Path = PACKAGED_MODEL_PATH) -> Pipeline:
    """Load a trusted fitted pipeline (joblib or legacy pickle); never train."""
    model_path = Path(model_path)
    if model_path.name == PACKAGED_MODEL_PATH.name:
        metadata = json.loads(
            (model_path.parent / "model_metadata.json").read_text(encoding="utf-8")
        )
        if metadata["sklearn_version"] != sklearn.__version__:
            raise ValueError("The packaged model requires its metadata's sklearn version.")
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata["model_sha256"]:
            raise ValueError("Model checksum does not match model_metadata.json.")
        if metadata["target_transformation"] != "log1p":
            raise ValueError("The packaged model must use the project's log1p target.")
    pipeline = _load_model(model_path)
    check_is_fitted(pipeline)
    return pipeline


def validate_features(customer_df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    """Validate against the fitted preprocessing schema without coercing input."""
    if not isinstance(customer_df, pd.DataFrame):
        raise TypeError("customer_df must be a pandas DataFrame.")
    if customer_df.empty:
        raise ValueError("customer_df must contain at least one customer.")
    if not customer_df.columns.is_unique:
        raise ValueError("customer_df contains duplicate column names.")
    check_is_fitted(pipeline)
    preprocessor = pipeline.named_steps["preprocessor"]
    numeric = list(preprocessor.transformers_[0][2])
    categorical = list(preprocessor.transformers_[1][2])
    features = numeric + categorical
    missing = [name for name in features if name not in customer_df.columns]
    if missing:
        raise ValueError(f"customer_df is missing feature columns: {missing}")
    for name in numeric:
        values = customer_df[name]
        if (
            not is_numeric_dtype(values.dtype)
            or is_bool_dtype(values.dtype)
            or is_complex_dtype(values.dtype)
        ):
            raise TypeError(f"{name} must contain real numeric values, not {values.dtype}.")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} contains missing or non-finite values.")
    for name in categorical:
        values = customer_df[name]
        if values.isna().any():
            raise ValueError(f"{name} contains missing required values.")
        if not values.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise TypeError(f"{name} must contain non-empty strings.")
    return customer_df[features]


def predict(
    customer_df: pd.DataFrame,
    pipeline: Pipeline | None = None,
    model_path: Path = PACKAGED_MODEL_PATH,
) -> pd.DataFrame:
    """Append log1p and expm1 GBP predictions, preserving input rows and dtypes.

    Extra fields (including a caller's customer identifier) are retained but not
    used as predictors. Unknown countries use the fitted encoder's ignore policy.
    No clipping or bias correction is applied to the inverse transformation.
    """
    if pipeline is None:
        pipeline = load_model(model_path)
    features = validate_features(customer_df, pipeline)
    log_preds = pipeline.predict(features)
    with np.errstate(over="raise", invalid="raise"):
        revenue_preds = np.expm1(log_preds)
    if not np.isfinite(log_preds).all() or not np.isfinite(revenue_preds).all():
        raise ValueError("The model generated non-finite predictions.")
    result = customer_df.copy()
    result[OUTPUT_COLUMNS[0]] = log_preds
    result[OUTPUT_COLUMNS[1]] = revenue_preds
    logger.info(f"[predict] Generated predictions for {len(result):,} customers.")
    return result


def predict_csv(
    input_path: Path, output_path: Path, model_path: Path = PACKAGED_MODEL_PATH
) -> pd.DataFrame:
    """Read customer features and write predictions without overwriting a file."""
    customer_df = pd.read_csv(input_path)
    result = predict(customer_df, model_path=model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="") as stream:
        result.to_csv(stream, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict from customer-level feature CSV records."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=PACKAGED_MODEL_PATH)
    args = parser.parse_args()
    predict_csv(args.input, args.output, args.model)


__all__ = ["load_model", "predict", "predict_csv", "validate_features"]


if __name__ == "__main__":
    main()
