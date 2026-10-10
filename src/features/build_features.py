"""
build_features.py — Customer-level RFM feature engineering.

Responsibility
--------------
Transform the cleaned transaction-level DataFrame (one row = one line item)
into a customer-level modelling dataset (one row = one customer) with:

* RFM-style input features derived from the *pre-cutoff* period.
* A ``future_revenue`` regression target from the *post-cutoff* period.

Design decisions
-----------------
* The prediction date (cutoff) separates feature window from target window.
* Features are computed only on pre-cutoff transactions so there is no
  data leakage.
* Customers with no pre-cutoff history are excluded (no feature vector).
* Customers with pre-cutoff history but no post-cutoff purchases get
  ``future_revenue = 0`` (valid target for churn / low-value prediction).
* ``log1p_future_revenue = log(future_revenue + 1)`` is safe for all rows
  because future_revenue >= 0.

Features engineered
--------------------
recency_days        : Days between last purchase and prediction date (lower = more recent).
frequency           : Number of unique invoices in the pre-cutoff window.
monetary            : Total spend (sum of line_total) in the pre-cutoff window.
avg_basket_value    : Mean spend per invoice in the pre-cutoff window.
unique_products     : Number of distinct StockCodes purchased pre-cutoff.
tenure_days         : Days between first and last purchase pre-cutoff.
purchase_rate       : frequency / tenure_days (purchases per day; 0 if tenure=0).
country             : Country of most recent purchase (categorical).

Target
------
future_revenue          : Sum of line_total in the post-cutoff window (>= 0).
log1p_future_revenue    : log(future_revenue + 1) — primary regression target.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ecommerce_customer_value_prediction.config import PROCESSED_DATA_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# The prediction date separates the feature-construction window (pre-cutoff)
# from the revenue-measurement window (post-cutoff).
# Choice: 2011-09-01 leaves ~21 months of history and ~4 months of target.
PREDICTION_DATE: pd.Timestamp = pd.Timestamp("2011-09-01")

CUSTOMER_DATASET_CSV: Path = PROCESSED_DATA_DIR / "customer_dataset.csv"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def build_rfm_features(
    df: pd.DataFrame,
    prediction_date: pd.Timestamp = PREDICTION_DATE,
) -> pd.DataFrame:
    """
    Build per-customer RFM features from *pre-cutoff* transactions.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction-level DataFrame (output of ``run_cleaning_pipeline``).
        Must contain columns: ``Customer ID``, ``InvoiceDate``, ``Invoice``,
        ``StockCode``, ``line_total``, ``Country``.
    prediction_date : pd.Timestamp
        Cutoff date.  Only rows with ``InvoiceDate < prediction_date`` are used.

    Returns
    -------
    pd.DataFrame
        One row per customer with columns:
        ``Customer ID``, ``recency_days``, ``frequency``, ``monetary``,
        ``avg_basket_value``, ``unique_products``, ``tenure_days``,
        ``purchase_rate``, ``country``.
    """
    pre = df[df["InvoiceDate"] < prediction_date].copy()

    if pre.empty:
        raise ValueError(
            f"No transactions before prediction_date={prediction_date.date()}. "
            "Check that InvoiceDate is parsed as datetime."
        )

    # Invoice-level spend (for avg_basket_value)
    invoice_spend = (
        pre.groupby(["Customer ID", "Invoice"])["line_total"]
        .sum()
        .reset_index()
        .rename(columns={"line_total": "invoice_spend"})
    )

    features = pre.groupby("Customer ID").agg(
        last_purchase=("InvoiceDate", "max"),
        first_purchase=("InvoiceDate", "min"),
        frequency=("Invoice", "nunique"),
        monetary=("line_total", "sum"),
        unique_products=("StockCode", "nunique"),
        # Country of most recent transaction
        country=("Country", "last"),
    )

    features["recency_days"] = (prediction_date - features["last_purchase"]).dt.days
    features["tenure_days"] = (
        features["last_purchase"] - features["first_purchase"]
    ).dt.days

    # Average basket value from invoice-level spend
    avg_basket = (
        invoice_spend.groupby("Customer ID")["invoice_spend"].mean().rename("avg_basket_value")
    )
    features = features.join(avg_basket)

    # Purchase rate: invoices per day of tenure (0 when tenure == 0)
    features["purchase_rate"] = np.where(
        features["tenure_days"] > 0,
        features["frequency"] / features["tenure_days"],
        0.0,
    )

    features = features.drop(columns=["last_purchase", "first_purchase"])
    features = features.reset_index()

    logger.info(
        f"[build_rfm_features] Built features for {len(features):,} customers "
        f"(pre-cutoff: {prediction_date.date()})."
    )
    return features


def build_target(
    df: pd.DataFrame,
    prediction_date: pd.Timestamp = PREDICTION_DATE,
) -> pd.DataFrame:
    """
    Compute the future-revenue target from *post-cutoff* transactions.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction-level DataFrame.
    prediction_date : pd.Timestamp
        Cutoff date.  Only rows with ``InvoiceDate >= prediction_date`` are used.

    Returns
    -------
    pd.DataFrame
        One row per customer with columns:
        ``Customer ID``, ``future_revenue``, ``log1p_future_revenue``.
        Only customers who appear in the post-cutoff window are included.
        Customers absent from the post-cutoff window receive
        ``future_revenue = 0`` after a left-join in ``build_customer_dataset``.
    """
    post = df[df["InvoiceDate"] >= prediction_date].copy()

    target = (
        post.groupby("Customer ID")["line_total"]
        .sum()
        .rename("future_revenue")
        .reset_index()
    )
    target["log1p_future_revenue"] = np.log1p(target["future_revenue"])

    logger.info(
        f"[build_target] Computed target for {len(target):,} customers "
        f"(post-cutoff: {prediction_date.date()})."
    )
    return target


def build_customer_dataset(
    df: pd.DataFrame,
    prediction_date: pd.Timestamp = PREDICTION_DATE,
) -> pd.DataFrame:
    """
    Join RFM features with the future-revenue target.

    Customers in the feature set but not in the target set (Group A —
    pre-cutoff only) receive ``future_revenue = 0``.

    Customers in the target set but not in the feature set (Group B —
    post-cutoff only) are excluded because no feature vector can be built.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction-level DataFrame.
    prediction_date : pd.Timestamp
        Cutoff date.

    Returns
    -------
    pd.DataFrame
        Modelling-ready customer dataset with features + target columns.
    """
    features = build_rfm_features(df, prediction_date)
    target = build_target(df, prediction_date)

    customer_df = features.merge(target, on="Customer ID", how="left")

    # Group A customers (pre-only) get future_revenue = 0
    customer_df["future_revenue"] = customer_df["future_revenue"].fillna(0.0)
    customer_df["log1p_future_revenue"] = np.log1p(customer_df["future_revenue"])

    n_group_a = int((customer_df["future_revenue"] == 0).sum())
    n_group_c = int((customer_df["future_revenue"] > 0).sum())
    logger.info(
        f"[build_customer_dataset] {len(customer_df):,} modellable customers. "
        f"Group A (future_revenue=0): {n_group_a:,}. "
        f"Group C (future_revenue>0): {n_group_c:,}."
    )
    return customer_df


def save_customer_dataset(
    customer_df: pd.DataFrame,
    output_path: Path = CUSTOMER_DATASET_CSV,
) -> None:
    """Save the customer-level modelling dataset to ``data/processed/``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    customer_df.to_csv(output_path, index=False)
    logger.success(
        f"Customer dataset saved → {output_path}  ({len(customer_df):,} customers)"
    )
