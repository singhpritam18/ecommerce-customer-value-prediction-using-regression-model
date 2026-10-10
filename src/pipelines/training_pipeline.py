"""
training_pipeline.py — End-to-end training workflow.

Responsibility
--------------
Orchestrate the full lifecycle from cleaned data to a persisted, evaluated model.

Supports four model types via the --model flag:
    linear      OLS LinearRegression (no regularisation)
    ridge       Ridge regression (L2)
    lasso       Lasso regression (L1)
    elasticnet  ElasticNet regression (L1 + L2)

Also supports --experiment all which trains and evaluates all four models
on the SAME frozen train/test split and writes a comparison CSV.

CLI examples
------------
    python -m ecommerce_customer_value_prediction.pipelines.training_pipeline --model linear
    python -m ecommerce_customer_value_prediction.pipelines.training_pipeline --model ridge --alpha 10.0
    python -m ecommerce_customer_value_prediction.pipelines.training_pipeline --model lasso --alpha 0.01
    python -m ecommerce_customer_value_prediction.pipelines.training_pipeline --model elasticnet --alpha 0.1 --l1-ratio 0.5
    python -m ecommerce_customer_value_prediction.pipelines.training_pipeline --experiment all
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from loguru import logger
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from ecommerce_customer_value_prediction.config import (
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)
from ecommerce_customer_value_prediction.data.clean_data import (
    CLEANED_CSV,
    run_cleaning_pipeline,
)
from ecommerce_customer_value_prediction.features.build_features import (
    CUSTOMER_DATASET_CSV,
    PREDICTION_DATE,
    build_customer_dataset,
    save_customer_dataset,
)
from ecommerce_customer_value_prediction.models.evaluate import evaluate, METRICS_PATH
from ecommerce_customer_value_prediction.models.train import (
    CATEGORICAL_FEATURES,
    FINAL_METADATA_PATH,
    FINAL_MODEL_PATH,
    MODEL_PATHS,
    NUMERIC_FEATURES,
    SUPPORTED_MODELS,
    TARGET_COL,
    create_model,
    load_model,
    save_model,
    train,
)

app = typer.Typer(add_completion=False)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5

# Canonical hyperparameter defaults for each model (used in --experiment all)
DEFAULT_HYPERPARAMS: dict[str, dict] = {
    "linear": {},
    "ridge": {"alpha": 1.0},
    "lasso": {"alpha": 0.01},
    "elasticnet": {"alpha": 0.1, "l1_ratio": 0.5},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_cleaned(cleaned_csv: Path = CLEANED_CSV) -> pd.DataFrame:
    """Load cleaned transactions, running the cleaning pipeline if absent."""
    if cleaned_csv.exists():
        logger.info(f"Loading existing cleaned data: {cleaned_csv}")
        df = pd.read_csv(cleaned_csv, parse_dates=["InvoiceDate"])
        df["Customer ID"] = df["Customer ID"].astype("Int64")
        return df
    logger.warning(f"Cleaned CSV not found at {cleaned_csv}. Running cleaning pipeline …")
    return run_cleaning_pipeline(save=True)


def _make_stratified_split(
    customer_df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split customer_df into train/test using stratified 80/20 split.

    Stratification is on a binary 'has_revenue' flag so that churned and
    active customers are proportionally represented in both splits.

    This exact split must be used by ALL four models for fair comparison.
    """
    strat_col = (customer_df["future_revenue"] > 0).astype(int)
    train_df, test_df = train_test_split(
        customer_df,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_col,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _cv_metrics(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """
    Run K-fold cross-validation on the training set and return mean metrics.

    Returns a dict with keys: cv_mae_log, cv_rmse_log, cv_r2.
    """

    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    from sklearn.metrics import make_scorer

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=kf,
            scoring={
                "mae": make_scorer(mean_absolute_error),
                "rmse": make_scorer(rmse),
                "r2": make_scorer(r2_score),
            },
        )
    return {
        "cv_mae_log": float(scores["test_mae"].mean()),
        "cv_rmse_log": float(scores["test_rmse"].mean()),
        "cv_r2": float(scores["test_r2"].mean()),
        "cv_mae_log_std": float(scores["test_mae"].std()),
        "cv_rmse_log_std": float(scores["test_rmse"].std()),
        "cv_r2_std": float(scores["test_r2"].std()),
    }


# ---------------------------------------------------------------------------
# Single-model training pipeline
# ---------------------------------------------------------------------------


def run_training_pipeline(
    model_name: str = "ridge",
    cleaned_csv: Path = CLEANED_CSV,
    customer_dataset_csv: Path = CUSTOMER_DATASET_CSV,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    prediction_date: pd.Timestamp = PREDICTION_DATE,
    save_final: bool = True,
) -> dict:
    """
    Run the full training pipeline for a single model.

    Steps
    -----
    1. Load cleaned transactions (run cleaning if absent).
    2. Build customer-level features + target.
    3. Stratified 80/20 train/test split.
    4. Cross-validate on training set (5-fold).
    5. Evaluate on held-out test set.
    6. Refit on full dataset and save final artifact.

    Parameters
    ----------
    model_name : str
        One of "linear", "ridge", "lasso", "elasticnet".
    alpha : float
        Regularisation strength (ignored for "linear").
    l1_ratio : float
        ElasticNet mixing parameter (ignored for other models).
    save_final : bool
        If True, save the model artifact to models/.

    Returns
    -------
    dict with cv_metrics, test_metrics, and model metadata.
    """
    logger.info("=" * 60)
    logger.info(f"Training pipeline: model={model_name}")
    logger.info("=" * 60)

    # Build estimator
    if model_name == "linear":
        estimator = create_model("linear")
    elif model_name == "ridge":
        estimator = create_model("ridge", alpha=alpha)
    elif model_name == "lasso":
        estimator = create_model("lasso", alpha=alpha)
    elif model_name == "elasticnet":
        estimator = create_model("elasticnet", alpha=alpha, l1_ratio=l1_ratio)
    else:
        raise ValueError(f"Unknown model_name '{model_name}'. Supported: {SUPPORTED_MODELS}")

    # Load / build customer dataset
    df = _load_cleaned(cleaned_csv)
    customer_df = build_customer_dataset(df, prediction_date=prediction_date)
    save_customer_dataset(customer_df, customer_dataset_csv)
    logger.info(f"Customer dataset: {len(customer_df):,} customers")

    # Train/test split
    train_df, test_df = _make_stratified_split(customer_df, test_size, random_state)
    logger.info(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

    # Cross-validation on training set
    from ecommerce_customer_value_prediction.models.train import build_pipeline

    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df[TARGET_COL]
    cv_pipeline = build_pipeline(estimator=create_model(
        model_name,
        **({} if model_name == "linear" else
           {"alpha": alpha} if model_name in ("ridge", "lasso") else
           {"alpha": alpha, "l1_ratio": l1_ratio}),
    ))
    cv_scores = _cv_metrics(cv_pipeline, X_train, y_train)
    logger.info(f"CV R²={cv_scores['cv_r2']:.4f} | CV RMSE={cv_scores['cv_rmse_log']:.4f}")

    # Fit on training set, evaluate on test set
    pipeline_train, _ = train(train_df, estimator=estimator)
    model_path, metadata_path = MODEL_PATHS[model_name]
    metrics_path = METRICS_DIR / f"{model_name}_test_metrics.json"
    test_metrics = evaluate(pipeline_train, test_df, metrics_path=metrics_path, save=True)
    logger.info(f"Test R²={test_metrics['r2_log']:.4f} | Test RMSE={test_metrics['rmse_log']:.4f}")

    # Refit on full dataset
    if save_final:
        final_pipeline, metadata = train(customer_df, estimator=estimator)
        metadata["cv_metrics"] = cv_scores
        metadata["test_metrics"] = test_metrics
        metadata["split"] = {"random_state": random_state, "test_size": test_size}
        metadata["prediction_date"] = str(prediction_date.date())
        metadata["trained_at"] = datetime.utcnow().isoformat()
        save_model(final_pipeline, metadata, model_path=model_path, metadata_path=metadata_path)
        logger.success(f"Model saved → {model_path}")

    result = {"model": model_name, **cv_scores, **{f"test_{k}": v for k, v in test_metrics.items()}}
    return result


# ---------------------------------------------------------------------------
# All-models experiment
# ---------------------------------------------------------------------------


def run_experiment_all(
    cleaned_csv: Path = CLEANED_CSV,
    customer_dataset_csv: Path = CUSTOMER_DATASET_CSV,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    prediction_date: pd.Timestamp = PREDICTION_DATE,
) -> pd.DataFrame:
    """
    Train and evaluate all four models on the same frozen train/test split.

    Uses the canonical DEFAULT_HYPERPARAMS for each model.
    Results are saved to reports/metrics/model_comparison.csv.

    Returns
    -------
    pd.DataFrame with one row per model and all CV + test metrics.
    """
    logger.info("=" * 60)
    logger.info("Running full experiment: OLS, Ridge, Lasso, ElasticNet")
    logger.info("=" * 60)

    # Build customer dataset once
    df = _load_cleaned(cleaned_csv)
    customer_df = build_customer_dataset(df, prediction_date=prediction_date)
    save_customer_dataset(customer_df, customer_dataset_csv)

    # Freeze the split — used by ALL models
    train_df, test_df = _make_stratified_split(customer_df, test_size, random_state)
    logger.info(
        f"Frozen split: train={len(train_df):,} | test={len(test_df):,} "
        f"(random_state={random_state})"
    )

    from ecommerce_customer_value_prediction.models.train import build_pipeline

    rows = []
    for model_name in SUPPORTED_MODELS:
        hparams = DEFAULT_HYPERPARAMS[model_name]
        estimator = create_model(model_name, **hparams)

        # CV on training set
        X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        y_train = train_df[TARGET_COL]
        cv_pipeline = build_pipeline(estimator=create_model(model_name, **hparams))
        cv_scores = _cv_metrics(cv_pipeline, X_train, y_train)

        # Fit on training set, evaluate on test
        pipeline_train, _ = train(train_df, estimator=estimator)
        metrics_path = METRICS_DIR / f"{model_name}_test_metrics.json"
        test_metrics = evaluate(pipeline_train, test_df, metrics_path=metrics_path, save=True)

        # Refit on full data and save
        final_pipeline, metadata = train(customer_df, estimator=create_model(model_name, **hparams))
        metadata["cv_metrics"] = cv_scores
        metadata["test_metrics"] = test_metrics
        metadata["split"] = {"random_state": random_state, "test_size": test_size}
        metadata["prediction_date"] = str(prediction_date.date())
        metadata["trained_at"] = datetime.utcnow().isoformat()
        model_path, meta_path = MODEL_PATHS[model_name]
        save_model(final_pipeline, metadata, model_path=model_path, metadata_path=meta_path)

        row = {
            "model": model_name,
            "hyperparameters": json.dumps(hparams),
            **{f"cv_{k}": v for k, v in cv_scores.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        rows.append(row)
        logger.success(
            f"[{model_name}] CV R²={cv_scores['cv_r2']:.4f} | "
            f"Test R²={test_metrics['r2_log']:.4f} | "
            f"Test MAE_gbp=£{test_metrics['mae_gbp']:.2f}"
        )

    results_df = pd.DataFrame(rows)

    # Save comparison CSV
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = METRICS_DIR / "model_comparison.csv"
    results_df.to_csv(comparison_path, index=False)
    logger.success(f"Model comparison saved → {comparison_path}")

    # Identify and save the best model (by CV R²)
    best_idx = results_df["cv_cv_r2"].idxmax() if "cv_cv_r2" in results_df.columns else results_df["cv_r2"].idxmax()
    best_model_name = results_df.loc[best_idx, "model"]
    logger.info(f"Best model by CV R²: {best_model_name}")

    # Copy the best model as the final artifact
    best_hparams = DEFAULT_HYPERPARAMS[best_model_name]
    best_estimator = create_model(best_model_name, **best_hparams)
    best_pipeline, best_meta = train(customer_df, estimator=best_estimator)
    best_meta["selected_as_final"] = True
    best_meta["selection_criterion"] = "CV R²"
    best_meta["all_model_cv_r2"] = {
        r["model"]: r.get("cv_cv_r2", r.get("cv_r2")) for r in rows
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_model(best_pipeline, best_meta, model_path=FINAL_MODEL_PATH, metadata_path=FINAL_METADATA_PATH)
    logger.success(f"Final model artifact → {FINAL_MODEL_PATH}")

    return results_df


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------


@app.command()
def main(
    model: str = typer.Option(
        "ridge",
        help="Model to train: linear, ridge, lasso, elasticnet.",
    ),
    experiment: Optional[str] = typer.Option(
        None,
        help="Set to 'all' to train and compare all four models.",
    ),
    alpha: float = typer.Option(1.0, help="Regularisation strength (Ridge/Lasso/ElasticNet)."),
    l1_ratio: float = typer.Option(0.5, help="ElasticNet L1/L2 mixing ratio (0=Ridge, 1=Lasso)."),
    test_size: float = typer.Option(TEST_SIZE, help="Held-out test fraction."),
    random_state: int = typer.Option(RANDOM_STATE, help="Random seed for reproducibility."),
) -> None:
    """
    Run the end-to-end training pipeline from the command line.

    Examples:

      Train OLS baseline:
        python -m ecommerce_customer_value_prediction.pipelines.training_pipeline --model linear

      Train Ridge with custom alpha:
        python -m ecommerce_customer_value_prediction.pipelines.training_pipeline --model ridge --alpha 10.0

      Train all four models and compare:
        python -m ecommerce_customer_value_prediction.pipelines.training_pipeline --experiment all
    """
    if experiment and experiment.lower() == "all":
        results = run_experiment_all(test_size=test_size, random_state=random_state)
        print("\n=== Model Comparison ===")
        print(results[["model", "cv_cv_r2", "cv_cv_rmse_log", "test_r2_log", "test_rmse_log", "test_mae_gbp"]].to_string(index=False))
        return

    model_lower = model.lower().strip()
    if model_lower not in SUPPORTED_MODELS:
        raise typer.BadParameter(f"--model must be one of {SUPPORTED_MODELS}")

    run_training_pipeline(
        model_name=model_lower,
        alpha=alpha,
        l1_ratio=l1_ratio,
        test_size=test_size,
        random_state=random_state,
    )


if __name__ == "__main__":
    app()
