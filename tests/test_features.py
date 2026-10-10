"""
tests/test_features.py — Unit tests for the features sub-package.

Uses synthetic in-memory DataFrames; no real dataset required.
"""

import numpy as np
import pandas as pd
import pytest

from ecommerce_customer_value_prediction.features.build_features import (
    PREDICTION_DATE,
    build_customer_dataset,
    build_rfm_features,
    build_target,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_cleaned_df(prediction_date: pd.Timestamp = PREDICTION_DATE) -> pd.DataFrame:
    """
    Synthetic cleaned transaction DataFrame (post-cleaning, pre-feature-engineering).

    Three customers:
    - cust_pre_only  (17850): only pre-cutoff transactions → Group A (future_revenue=0)
    - cust_both      (13047): pre and post-cutoff transactions → Group C
    - cust_post_only (12583): only post-cutoff transactions → Group B (excluded)
    """
    pre_date = prediction_date - pd.Timedelta(days=30)
    post_date = prediction_date + pd.Timedelta(days=15)

    rows = [
        # cust_pre_only: 2 pre-cutoff invoices
        ("INV001", "85123A", 6, 2.55, pre_date - pd.Timedelta(days=60), 17850, "United Kingdom"),
        ("INV002", "71053",  3, 3.39, pre_date,                         17850, "United Kingdom"),
        # cust_both: 1 pre-cutoff, 1 post-cutoff
        ("INV003", "22771", 12, 1.25, pre_date - pd.Timedelta(days=10), 13047, "Germany"),
        ("INV004", "22772",  5, 4.00, post_date,                        13047, "Germany"),
        # cust_post_only: post-cutoff only (Group B — will be excluded from features)
        ("INV005", "85123B",  2, 5.50, post_date,                       12583, "France"),
    ]

    df = pd.DataFrame(rows, columns=[
        "Invoice", "StockCode", "Quantity", "Price", "InvoiceDate", "Customer ID", "Country"
    ])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Customer ID"] = pd.array(df["Customer ID"].tolist(), dtype="Int64")
    df["line_total"] = df["Quantity"] * df["Price"]
    return df


# ---------------------------------------------------------------------------
# Tests: build_rfm_features
# ---------------------------------------------------------------------------


def test_build_rfm_features_returns_one_row_per_customer():
    df = _make_cleaned_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    # Only pre-cutoff customers: 17850 and 13047
    assert len(features) == 2
    assert set(features["Customer ID"].tolist()) == {17850, 13047}


def test_build_rfm_features_columns_present():
    df = _make_cleaned_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    expected_cols = {
        "Customer ID", "recency_days", "frequency", "monetary",
        "avg_basket_value", "unique_products", "tenure_days",
        "purchase_rate", "country",
    }
    assert expected_cols.issubset(set(features.columns))


def test_build_rfm_features_recency_non_negative():
    df = _make_cleaned_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    assert (features["recency_days"] >= 0).all()


def test_build_rfm_features_frequency_positive():
    df = _make_cleaned_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    assert (features["frequency"] >= 1).all()


def test_build_rfm_features_monetary_positive():
    df = _make_cleaned_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    assert (features["monetary"] > 0).all()


def test_build_rfm_features_cust_pre_only_has_two_invoices():
    df = _make_cleaned_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    row = features[features["Customer ID"] == 17850].iloc[0]
    assert row["frequency"] == 2


def test_build_rfm_features_excludes_post_cutoff_customers():
    df = _make_cleaned_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    # cust_post_only (12583) has no pre-cutoff transactions
    assert 12583 not in features["Customer ID"].tolist()


def test_build_rfm_features_raises_on_empty_pre_cutoff():
    df = _make_cleaned_df()
    # Use a prediction date before all transactions
    early_date = pd.Timestamp("2000-01-01")
    with pytest.raises(ValueError, match="No transactions before"):
        build_rfm_features(df, early_date)


# ---------------------------------------------------------------------------
# Tests: build_target
# ---------------------------------------------------------------------------


def test_build_target_correct_customers():
    df = _make_cleaned_df()
    target = build_target(df, PREDICTION_DATE)
    # Post-cutoff: cust_both (13047) and cust_post_only (12583)
    assert set(target["Customer ID"].tolist()) == {13047, 12583}


def test_build_target_revenue_positive():
    df = _make_cleaned_df()
    target = build_target(df, PREDICTION_DATE)
    assert (target["future_revenue"] > 0).all()


def test_build_target_log1p_column_present():
    df = _make_cleaned_df()
    target = build_target(df, PREDICTION_DATE)
    assert "log1p_future_revenue" in target.columns
    assert (target["log1p_future_revenue"] >= 0).all()


# ---------------------------------------------------------------------------
# Tests: build_customer_dataset
# ---------------------------------------------------------------------------


def test_build_customer_dataset_shape():
    df = _make_cleaned_df()
    customer_df = build_customer_dataset(df, PREDICTION_DATE)
    # Modellable = pre-cutoff customers: 17850 and 13047
    assert len(customer_df) == 2


def test_build_customer_dataset_group_a_has_zero_revenue():
    df = _make_cleaned_df()
    customer_df = build_customer_dataset(df, PREDICTION_DATE)
    group_a = customer_df[customer_df["Customer ID"] == 17850]
    assert len(group_a) == 1
    assert float(group_a["future_revenue"].iloc[0]) == 0.0


def test_build_customer_dataset_group_c_has_positive_revenue():
    df = _make_cleaned_df()
    customer_df = build_customer_dataset(df, PREDICTION_DATE)
    group_c = customer_df[customer_df["Customer ID"] == 13047]
    assert len(group_c) == 1
    assert float(group_c["future_revenue"].iloc[0]) > 0


def test_build_customer_dataset_log1p_safe():
    """log1p_future_revenue must be >= 0 for all customers (no log(negative))."""
    df = _make_cleaned_df()
    customer_df = build_customer_dataset(df, PREDICTION_DATE)
    assert (customer_df["log1p_future_revenue"] >= 0).all()


def test_build_customer_dataset_post_only_excluded():
    """Group B customers (post-cutoff only) must not appear in the output."""
    df = _make_cleaned_df()
    customer_df = build_customer_dataset(df, PREDICTION_DATE)
    assert 12583 not in customer_df["Customer ID"].tolist()
