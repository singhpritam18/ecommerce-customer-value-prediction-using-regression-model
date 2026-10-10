"""
training_pipeline.py — End-to-end training workflow.

Responsibility
--------------
Orchestrate the full lifecycle from raw data to a persisted, evaluated model:

1. Load cleaned transaction data (or run the cleaning pipeline if needed).
2. Build customer-level RFM features and the future-revenue target.
3. Save the customer dataset to ``data/processed/``.
4. Perform an 80/20 train/test split stratified by a binary "has revenue" flag.
5. Fit the Ridge pipeline on the training set.
6. Evaluate on the held-out test set and save metrics.
7. Refit the final pipeline on the full dataset and save the model artifact.

This module is the single entry point for the CLI command ``make train``.
"""

import typer
from loguru import logger
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ecommerce_customer_value_prediction.config import METRICS_DIR  # noqa: F401
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
    MODEL_PATH,
    METADATA_PATH,
    train,
    save_model,
)

app = typer.Typer()


def _load_cleaned(cleaned_csv: Path = CLEANED_CSV) -> pd.DataFrame:
    """
    Load the cleaned transaction CSV.  Run the cleaning pipeline if absent.
    """
    if cleaned_csv.exists():
        logger.info(f"Loading existing cleaned data: {cleaned_csv}")
        df = pd.read_csv(cleaned_csv, parse_dates=["InvoiceDate"])
        # Restore Int64 for Customer ID (CSV round-trips as float64)
        df["Customer ID"] = df["Customer ID"].astype("Int64")
        return df

    logger.warning(
        f"Cleaned CSV not found at {cleaned_csv}. "
        "Running cleaning pipeline first…"
    )
    return run_cleaning_pipeline(save=True)


def run_training_pipeline(
    cleaned_csv: Path = CLEANED_CSV,
    customer_dataset_csv: Path = CUSTOMER_DATASET_CSV,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
    metrics_path: Path = METRICS_PATH,
    alpha: float = 1.0,
    test_size: float = 0.20,
    random_state: int = 42,
    prediction_date: pd.Timestamp = PREDICTION_DATE,
) -> dict:
    """
    Run the full training pipeline.

    Parameters
    ----------
    cleaned_csv : Path
        Cleaned transaction-level CSV (``data/interim/transactions_cleaned.csv``).
    customer_dataset_csv : Path
        Output path for the customer-level dataset (``data/processed/``).
    model_path : Path
        Where to save the final fitted pipeline (``.pkl``).
    metadata_path : Path
        Where to save training metadata (``.json``).
    metrics_path : Path
        Where to save evaluation metrics (``.json``).
    alpha : float
        Ridge regularisation strength.
    test_size : float
        Fraction of customers held out for evaluation.
    random_state : int
        Controls train/test split reproducibility.
    prediction_date : pd.Timestamp
        Cutoff date for feature / target construction.

    Returns
    -------
    dict
        Evaluation metrics from the held-out test set.
    """
    logger.info("=" * 60)
    logger.info("Starting training pipeline")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1 — Load cleaned transactions
    # ------------------------------------------------------------------
    df = _load_cleaned(cleaned_csv)
    logger.info(f"Cleaned transactions loaded: {len(df):,} rows")

    # ------------------------------------------------------------------
    # Step 2 — Build customer dataset
    # ------------------------------------------------------------------
    customer_df = build_customer_dataset(df, prediction_date=prediction_date)
    save_customer_dataset(customer_df, customer_dataset_csv)
    logger.info(f"Customer dataset: {len(customer_df):,} customers")

    # ------------------------------------------------------------------
    # Step 3 — Train / test split
    #   Stratify on binary has_revenue so both splits contain churned and
    #   active customers in roughly equal proportions.
    # ------------------------------------------------------------------
    customer_df["_has_revenue"] = (customer_df["future_revenue"] > 0).astype(int)

    train_df, test_df = train_test_split(
        customer_df,
        test_size=test_size,
        random_state=random_state,
        stratify=customer_df["_has_revenue"],
    )
    train_df = train_df.drop(columns=["_has_revenue"])
    test_df = test_df.drop(columns=["_has_revenue"])
    customer_df = customer_df.drop(columns=["_has_revenue"])

    logger.info(f"Train: {len(train_df):,} customers | Test: {len(test_df):,} customers")

    # ------------------------------------------------------------------
    # Step 4 — Fit on training set and evaluate on test set
    # ------------------------------------------------------------------
    pipeline_cv, _ = train(train_df, alpha=alpha)
    metrics = evaluate(pipeline_cv, test_df, metrics_path=metrics_path, save=True)

    logger.info("Hold-out test metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k:<25}: {v}")

    # ------------------------------------------------------------------
    # Step 5 — Refit final pipeline on the full dataset, then save
    # ------------------------------------------------------------------
    final_pipeline, metadata = train(customer_df, alpha=alpha)
    metadata["test_metrics"] = metrics
    save_model(final_pipeline, metadata, model_path=model_path, metadata_path=metadata_path)

    logger.success("Training pipeline complete.")
    logger.info(f"  Model    : {model_path}")
    logger.info(f"  Metadata : {metadata_path}")
    logger.info(f"  Metrics  : {metrics_path}")

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@app.command()
def main(
    alpha: float = typer.Option(1.0, help="Ridge regularisation strength."),
    test_size: float = typer.Option(0.20, help="Held-out test fraction."),
    random_state: int = typer.Option(42, help="Random seed for reproducibility."),
) -> None:
    """Run the end-to-end training pipeline from the command line."""
    run_training_pipeline(alpha=alpha, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    app()
