"""Training-CV selection, one-time holdout evaluation, and locked-model packaging."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import joblib
from loguru import logger
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone

from ecommerce_customer_value_prediction.config import METRICS_DIR, MODELS_DIR
from ecommerce_customer_value_prediction.data.clean_data import CLEANED_CSV
from ecommerce_customer_value_prediction.data.validate_data import validate_cleaned
from ecommerce_customer_value_prediction.features.build_features import (
    CUSTOMER_DATASET_CSV,
    PREDICTION_DATE,
    build_customer_dataset,
)
from ecommerce_customer_value_prediction.models.evaluate import evaluate
from ecommerce_customer_value_prediction.models.train import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PACKAGED_METADATA_PATH,
    PACKAGED_MODEL_PATH,
    SUPPORTED_MODELS,
    TARGET_COL,
    create_model,
    train,
)
from ecommerce_customer_value_prediction.models.tune import (
    CV_FOLDS,
    PARAM_GRIDS,
    RANDOM_STATE,
    run_cv,
    run_grid_search,
)
from ecommerce_customer_value_prediction.pipelines.training_pipeline import _make_stratified_split

EXPERIMENT_DIR = METRICS_DIR / "final_experiment"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = (
        NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL, "future_revenue", "Customer ID"]
    )
    missing = [name for name in required if name not in df]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    if df[required].isna().any().any():
        raise ValueError("Dataset contains missing values.")
    if df["Customer ID"].duplicated().any():
        raise ValueError("Dataset must contain one row per customer.")
    if (df["future_revenue"] < 0).any() or not np.allclose(
        df[TARGET_COL], np.log1p(df["future_revenue"]), rtol=1e-12, atol=1e-12
    ):
        raise ValueError("Dataset target does not match log1p(future_revenue).")
    return df


def run_experiment(
    dataset_path: Path = CUSTOMER_DATASET_CSV,
    output_dir: Path = EXPERIMENT_DIR,
) -> dict:
    """Lock all configurations and the CV winner before accessing test metrics.

    Each output directory is single-use, including failed/partial experiments.
    A repeat experiment must explicitly name a new output directory.
    """
    df = load_dataset(dataset_path)
    dataset_context = {}
    if dataset_path.resolve() == CUSTOMER_DATASET_CSV.resolve() and CLEANED_CSV.exists():
        transactions = pd.read_csv(
            CLEANED_CSV, parse_dates=["InvoiceDate"], dtype={"Invoice": str, "StockCode": str}
        )
        transactions["Customer ID"] = transactions["Customer ID"].astype("Int64")
        validate_cleaned(transactions)
        rebuilt = build_customer_dataset(transactions)
        pd.testing.assert_frame_equal(rebuilt, df, check_dtype=False, rtol=1e-12, atol=1e-12)
        dataset_context = {
            "cleaned_transactions_sha256": file_hash(CLEANED_CSV),
            "n_cleaned_transactions": len(transactions),
            "history_start": str(transactions["InvoiceDate"].min()),
            "observation_end": str(transactions["InvoiceDate"].max()),
            "observed_target_days": int(
                (transactions["InvoiceDate"].max().normalize() - PREDICTION_DATE).days
            ),
            "validation": "Cleaned-data checks and feature rebuild equivalence passed",
        }
    output_dir.mkdir(parents=True, exist_ok=False)
    train_df, test_df = _make_stratified_split(df)
    candidates = {}
    pipelines = {}
    search_rows = []
    for name in SUPPORTED_MODELS:
        if name == "linear":
            pipeline, _ = train(train_df, estimator=create_model(name))
            summary = run_cv(name, train_df)
            rows = pd.DataFrame(
                [
                    {
                        "model": name,
                        "mean_cv_rmse_log": summary["mean_cv_rmse_log"],
                        "std_cv_rmse_log": summary["std_cv_rmse_log"],
                        "rank": 1,
                    }
                ]
            )
            score = summary["mean_cv_rmse_log"]
        else:
            search, rows = run_grid_search(name, train_df, n_jobs=1)
            pipeline = search.best_estimator_
            params = pipeline.named_steps["regressor"].get_params()
            summary = run_cv(name, train_df, hyperparameters=params)
            score = -float(search.best_score_)
        candidates[name] = {
            "model_type": name,
            "estimator_class": type(pipeline.named_steps["regressor"]).__name__,
            "hyperparameters": pipeline.named_steps["regressor"].get_params(),
            "selection_cv_rmse_log": score,
            "cv_metrics": summary,
        }
        pipelines[name] = pipeline
        search_rows.append(rows)

    selected = min(candidates, key=lambda name: candidates[name]["selection_cv_rmse_log"])
    lock = {
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "selected_model": selected,
        "selection_rule": "Minimum training-only mean 5-fold CV RMSE on log1p target",
        "candidates": candidates,
        "parameter_grids": PARAM_GRIDS,
        "dataset": {
            "name": "UCI Online Retail II",
            "path": dataset_path.name,
            "sha256": file_hash(dataset_path),
            "n_customers": len(df),
            "zero_revenue_fraction": float((df["future_revenue"] == 0).mean()),
            **dataset_context,
        },
        "split": {
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
            "stratification": "future_revenue > 0",
            "n_train": len(train_df),
            "n_test": len(test_df),
            "train_customer_ids_sha256": hashlib.sha256(
                train_df["Customer ID"].to_numpy().tobytes()
            ).hexdigest(),
            "test_customer_ids_sha256": hashlib.sha256(
                test_df["Customer ID"].to_numpy().tobytes()
            ).hexdigest(),
        },
        "cv": {
            "n_folds": CV_FOLDS,
            "random_state": RANDOM_STATE,
            "stratification": "future_revenue > 0",
        },
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET_COL,
        "target_transformation": "log1p",
        "inverse_transformation": "expm1, without clipping or smearing correction",
        "prediction_cutoff": str(PREDICTION_DATE.date()),
        "prediction_horizon": "Approximately four months; available post-cutoff transactions",
        "preprocessing": {
            "numeric": "StandardScaler",
            "categorical": "OneHotEncoder(handle_unknown='ignore', sparse_output=False)",
            "remainder": "drop",
        },
        "post_selection_policy": "Refit selected pipeline on all customer rows after holdout evaluation",
        "sklearn_version": sklearn.__version__,
    }
    pd.concat(search_rows, ignore_index=True).to_csv(output_dir / "cv_results.csv", index=False)
    write_json(output_dir / "selection.json", lock)
    logger.info(f"Locked {selected} using training CV; starting one-time test evaluation.")

    metrics = {
        name: evaluate(pipeline, test_df, save=False) for name, pipeline in pipelines.items()
    }
    write_json(
        output_dir / "test_metrics.json",
        {
            "selection_sha256": file_hash(output_dir / "selection.json"),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "models": metrics,
            "scope": "Held-out customers, training-partition-fitted pipelines; not full-data refits",
        },
    )
    pipeline = pipelines[selected]
    predictions = pipeline.predict(test_df[lock["features"]])
    actual = test_df["future_revenue"].to_numpy()
    revenue = np.expm1(predictions)
    segments = {}
    for name, mask in {
        "zero": actual == 0,
        "positive": actual > 0,
        "over_1000_gbp": actual > 1000,
    }.items():
        if mask.any():
            segments[name] = {
                "n_customers": int(mask.sum()),
                "mae_gbp": float(np.abs(actual[mask] - revenue[mask]).mean()),
                "mean_actual_gbp": float(actual[mask].mean()),
                "mean_predicted_gbp": float(revenue[mask].mean()),
            }
    write_json(
        output_dir / "error_analysis.json",
        {
            "model": selected,
            "segments": segments,
            "negative_predictions": int((revenue < 0).sum()),
            "mean_actual_gbp": float(actual.mean()),
            "mean_predicted_gbp": float(revenue.mean()),
            "largest_absolute_error_gbp": float(np.abs(actual - revenue).max()),
            "note": "Diagnostic only; does not alter selection, target, or inverse transform.",
        },
    )
    return lock


def package_selected_model(
    dataset_path: Path = CUSTOMER_DATASET_CSV,
    experiment_dir: Path = EXPERIMENT_DIR,
    model_path: Path = PACKAGED_MODEL_PATH,
    metadata_path: Path = PACKAGED_METADATA_PATH,
) -> dict:
    """Refit exactly the locked configuration; no search or test-set evaluation."""
    selection_path = experiment_dir / "selection.json"
    lock = json.loads(selection_path.read_text(encoding="utf-8"))
    evaluation = json.loads((experiment_dir / "test_metrics.json").read_text(encoding="utf-8"))
    if file_hash(dataset_path) != lock["dataset"]["sha256"]:
        raise ValueError("Dataset differs from the locked experiment; refusing to refit.")
    if evaluation["selection_sha256"] != file_hash(selection_path):
        raise ValueError("Evaluation is not linked to this selection artifact.")
    if sklearn.__version__ != lock["sklearn_version"]:
        raise ValueError("Use the experiment's sklearn version when refitting.")
    if (
        lock["numeric_features"] != NUMERIC_FEATURES
        or lock["categorical_features"] != CATEGORICAL_FEATURES
        or lock["target"] != TARGET_COL
        or lock["target_transformation"] != "log1p"
    ):
        raise ValueError("Source feature/target contract differs from the locked experiment.")
    candidate = lock["candidates"][lock["selected_model"]]
    df = load_dataset(dataset_path)
    estimator = create_model(candidate["model_type"], **candidate["hyperparameters"])
    pipeline, training_metadata = train(df, estimator=clone(estimator))
    features = df[lock["features"]].head(10)
    before = pipeline.predict(features)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists() or metadata_path.exists():
        archive = (
            MODELS_DIR
            / "archive"
            / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
        )
        archive.mkdir(parents=True)
        for path in (model_path, metadata_path):
            if path.exists():
                shutil.copy2(path, archive / path.name)
        logger.warning(f"Preserved previous artifacts in {archive}.")
    temporary = model_path.with_name(model_path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        joblib.dump(pipeline, temporary)
        restored = joblib.load(temporary)
        np.testing.assert_allclose(restored.predict(features), before, rtol=1e-12, atol=1e-12)
        temporary.replace(model_path)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = {
        **lock,
        **candidate,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "n_training_customers": len(df),
        "training_summary": training_metadata,
        "final_test_metrics": evaluation["models"][lock["selected_model"]],
        "test_metric_scope": evaluation["scope"],
        "selection_sha256": file_hash(selection_path),
        "model_sha256": file_hash(model_path),
        "model_path": model_path.name,
        "input_schema": {
            name: "finite real number (integer or float)" for name in NUMERIC_FEATURES
        }
        | {name: "non-empty string" for name in CATEGORICAL_FEATURES},
        "output_schema": {
            "pred_log1p_future_revenue": "raw log1p-scale prediction",
            "pred_future_revenue": "expm1 prediction in GBP; not a calibrated expected value",
        },
        "versions": {
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }
    temporary_metadata = metadata_path.with_name(
        metadata_path.name + "." + uuid.uuid4().hex + ".tmp"
    )
    try:
        write_json(temporary_metadata, metadata)
        temporary_metadata.replace(metadata_path)
    finally:
        temporary_metadata.unlink(missing_ok=True)
    logger.success(f"Packaged locked model: {model_path}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["experiment", "package"])
    parser.add_argument("--dataset", type=Path, default=CUSTOMER_DATASET_CSV)
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT_DIR)
    args = parser.parse_args()
    if args.action == "experiment":
        run_experiment(args.dataset, args.experiment_dir)
    else:
        package_selected_model(args.dataset, args.experiment_dir)


if __name__ == "__main__":
    main()
