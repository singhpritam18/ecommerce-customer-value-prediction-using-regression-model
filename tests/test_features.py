"""
tests/test_features.py — Unit tests for the features sub-package.

Uses synthetic in-memory DataFrames; no real dataset required.

Test groups
-----------
1.  build_rfm_features — basic structural checks
2.  build_rfm_features — exact-value assertions (manually computed)
3.  build_rfm_features — edge cases (tenure=0, single invoice, multi-country)
4.  build_target — structural and value checks
5.  build_customer_dataset — join logic, Group A / B / C, log1p safety
6.  Leakage checklist — verify X and y columns are disjoint, no future data in X
"""

import math

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
# Shared test fixtures
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
        ("INV002", "71053", 3, 3.39, pre_date, 17850, "United Kingdom"),
        # cust_both: 1 pre-cutoff, 1 post-cutoff
        ("INV003", "22771", 12, 1.25, pre_date - pd.Timedelta(days=10), 13047, "Germany"),
        ("INV004", "22772", 5, 4.00, post_date, 13047, "Germany"),
        # cust_post_only: post-cutoff only (Group B — will be excluded from features)
        ("INV005", "85123B", 2, 5.50, post_date, 12583, "France"),
    ]

    df = pd.DataFrame(
        rows,
        columns=["Invoice", "StockCode", "Quantity", "Price", "InvoiceDate", "Customer ID", "Country"],
    )
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Customer ID"] = pd.array(df["Customer ID"].tolist(), dtype="Int64")
    df["line_total"] = df["Quantity"] * df["Price"]
    return df


def _make_exact_df(prediction_date: pd.Timestamp = PREDICTION_DATE) -> pd.DataFrame:
    """
    Minimal synthetic dataset designed so every feature value can be
    computed by hand without a calculator.

    Customer 1001 — two historical invoices, no future transactions (Group A):
        INV-A  (30 days before cutoff): StockCode X, Qty=2, Price=10.00 → line_total=20.00
        INV-B  (10 days before cutoff): StockCode Y, Qty=3, Price=5.00  → line_total=15.00
                                         StockCode Z, Qty=1, Price=5.00  → line_total=5.00

    Customer 2002 — one historical invoice, one future invoice (Group C):
        INV-C  (5 days before cutoff) : StockCode W, Qty=4, Price=8.00  → line_total=32.00
        INV-D  (10 days after cutoff) : StockCode V, Qty=2, Price=6.00  → line_total=12.00

    Customer 3003 — same-day historical invoice (tenure == 0) (Group A):
        INV-E  (20 days before cutoff): StockCode P, Qty=5, Price=4.00  → line_total=20.00
    """
    d = prediction_date
    rows = [
        # Customer 1001 — historical invoices
        ("INV-A", "X", 2, 10.00, d - pd.Timedelta(days=30), 1001, "United Kingdom"),
        ("INV-B", "Y", 3,  5.00, d - pd.Timedelta(days=10), 1001, "United Kingdom"),
        ("INV-B", "Z", 1,  5.00, d - pd.Timedelta(days=10), 1001, "United Kingdom"),
        # Customer 2002 — historical + future
        ("INV-C", "W", 4,  8.00, d - pd.Timedelta(days=5),  2002, "Germany"),
        ("INV-D", "V", 2,  6.00, d + pd.Timedelta(days=10), 2002, "Germany"),
        # Customer 3003 — same-day (tenure == 0)
        ("INV-E", "P", 5,  4.00, d - pd.Timedelta(days=20), 3003, "France"),
    ]
    df = pd.DataFrame(
        rows,
        columns=["Invoice", "StockCode", "Quantity", "Price", "InvoiceDate", "Customer ID", "Country"],
    )
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Customer ID"] = pd.array(df["Customer ID"].tolist(), dtype="Int64")
    df["line_total"] = df["Quantity"] * df["Price"]
    return df


def _make_multi_country_df(prediction_date: pd.Timestamp = PREDICTION_DATE) -> pd.DataFrame:
    """
    Customer 9001 has transactions in two countries.
    Earlier transaction: Spain. Later transaction: Portugal.
    Most-recent-country rule must return Portugal.
    """
    d = prediction_date
    rows = [
        ("INV-MC1", "AAA", 1, 10.0, d - pd.Timedelta(days=50), 9001, "Spain"),
        ("INV-MC2", "BBB", 2, 15.0, d - pd.Timedelta(days=10), 9001, "Portugal"),
    ]
    df = pd.DataFrame(
        rows,
        columns=["Invoice", "StockCode", "Quantity", "Price", "InvoiceDate", "Customer ID", "Country"],
    )
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Customer ID"] = pd.array(df["Customer ID"].tolist(), dtype="Int64")
    df["line_total"] = df["Quantity"] * df["Price"]
    return df


# ---------------------------------------------------------------------------
# GROUP 1 — build_rfm_features: structural checks
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
        "Customer ID",
        "recency_days",
        "frequency",
        "monetary",
        "avg_basket_value",
        "unique_products",
        "tenure_days",
        "purchase_rate",
        "country",
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
# GROUP 2 — build_rfm_features: exact-value assertions
#
# Customer 1001 (from _make_exact_df):
#   INV-A: 30 days before cutoff, line_total=20.00  (StockCode X)
#   INV-B: 10 days before cutoff, line_total=15.00+5.00=20.00  (StockCodes Y, Z)
#
#   recency_days     = cutoff - last_purchase = cutoff - (cutoff - 10d) = 10
#   frequency        = 2  (INV-A and INV-B are distinct invoices)
#   monetary         = 20.00 + 15.00 + 5.00 = 40.00
#   avg_basket_value = (20.00 + 20.00) / 2 = 20.00
#   unique_products  = 2  (StockCode X and Y) ... wait — X, Y, Z → 3
#   tenure_days      = last - first = (cutoff-10) - (cutoff-30) = 20
#   purchase_rate    = 2 / 20 = 0.1
#   country          = "United Kingdom"
#
# Customer 2002 (from _make_exact_df):
#   INV-C: 5 days before cutoff, line_total=32.00  (StockCode W)
#
#   recency_days     = 5
#   frequency        = 1
#   monetary         = 32.00
#   avg_basket_value = 32.00
#   unique_products  = 1  (StockCode W only)
#   tenure_days      = 0  (only one invoice date)
#   purchase_rate    = 0.0  (tenure==0 → assigned 0)
#   country          = "Germany"
#
# Customer 3003 (from _make_exact_df):
#   INV-E: 20 days before cutoff, line_total=20.00
#
#   recency_days     = 20
#   frequency        = 1
#   monetary         = 20.00
#   avg_basket_value = 20.00
#   unique_products  = 1
#   tenure_days      = 0
#   purchase_rate    = 0.0
#   country          = "France"
# ---------------------------------------------------------------------------


def _get_cust(features: pd.DataFrame, cid: int) -> pd.Series:
    """Helper: extract the single row for a given customer ID."""
    rows = features[features["Customer ID"] == cid]
    assert len(rows) == 1, f"Expected 1 row for customer {cid}, got {len(rows)}"
    return rows.iloc[0]


def test_exact_recency_customer_1001():
    df = _make_exact_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    row = _get_cust(features, 1001)
    assert row["recency_days"] == 10


def test_exact_recency_customer_2002():
    df = _make_exact_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    row = _get_cust(features, 2002)
    assert row["recency_days"] == 5


def test_exact_frequency_customer_1001():
    """INV-A and INV-B are two distinct invoices → frequency = 2."""
    df = _make_exact_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    row = _get_cust(features, 1001)
    assert row["frequency"] == 2


def test_exact_frequency_not_line_count():
    """
    INV-B has 2 line items (StockCode Y and Z). Frequency counts invoices,
    not rows → frequency must be 2, not 3.
    """
    df = _make_exact_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    row = _get_cust(features, 1001)
    # 3 pre-cutoff rows but only 2 distinct invoices
    assert row["frequency"] == 2


def test_exact_monetary_customer_1001():
    """20.00 (INV-A) + 15.00 + 5.00 (INV-B) = 40.00"""
    df = _make_exact_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    row = _get_cust(features, 1001)
    assert math.isclose(row["monetary"], 40.00, rel_tol=1e-9)


def test_exact_monetary_customer_2002():
    """4 * 8.00 = 32.00"""
    df = _make_exact_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    row = _get_cust(features, 2002)
    assert math.isclose(row["monetary"], 32.00, rel_tol=1e-9)


def test_exact_avg_basket_value_customer_1001():
    """INV-A spend=20.00, INV-B spend=20.00 → avg = (20+20)/2 = 20.00"""
    df = _make_exact_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    row = _get_cust(features, 1001)
    assert math.isclose(row["avg_basket_value"], 20.00, rel_tol=1e-9)


def test_exact_avg_basket_value_single_invoice():
    """Customer 2002 has exactly one invoice → avg_basket_value == monetary."""
    df = _make_exact_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    row = _get_cust(features, 2002)
    assert math.isclose(row["avg_basket_value"], 32.00, rel_tol=1e-9)


def test_exact_unique_products_customer_1001():
    """StockCodes X, Y, Z → 3 distinct products."""
    df = _make_exact_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    row = _get_cust(features, 1001)
    assert row["unique_products"] == 3


def test_exact_unique_products_customer_2002():
    """Only StockCode W → 1 unique product."""
    df = _make_exact_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    row = _get_cust(features, 2002)
    assert row["unique_products"] == 1


def test_exact_tenure_days_customer_1001():
    """first=(cutoff-30d), last=(cutoff-10d) → tenure = 20 days."""
    df = _make_exact_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    row = _get_cust(features, 1001)
    assert row["tenure_days"] == 20


def test_exact_purchase_rate_customer_1001():
    """frequency=2, tenure=20 → purchase_rate = 0.10."""
    df = _make_exact_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    row = _get_cust(features, 1001)
    assert math.isclose(row["purchase_rate"], 0.10, rel_tol=1e-9)


def test_exact_country_customer_1001():
    df = _make_exact_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    row = _get_cust(features, 1001)
    assert row["country"] == "United Kingdom"


# ---------------------------------------------------------------------------
# GROUP 3 — build_rfm_features: edge cases
# ---------------------------------------------------------------------------


def test_tenure_zero_customers_have_zero_purchase_rate():
    """
    Customers whose all historical transactions fall on a single day have
    tenure_days == 0.  purchase_rate must be 0.0, not NaN or inf.

    Customer 2002 (single invoice date) and Customer 3003 (single invoice)
    both have tenure == 0.
    """
    df = _make_exact_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    for cid in [2002, 3003]:
        row = _get_cust(features, cid)
        assert row["tenure_days"] == 0
        assert row["purchase_rate"] == 0.0, (
            f"Customer {cid}: tenure=0 but purchase_rate={row['purchase_rate']}"
        )


def test_purchase_rate_is_never_nan_or_inf():
    """No customer should ever have NaN or infinite purchase_rate."""
    df = _make_exact_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    assert features["purchase_rate"].isna().sum() == 0
    assert np.isinf(features["purchase_rate"]).sum() == 0


def test_country_uses_most_recent_invoice():
    """
    When a customer has transactions in multiple countries, country must
    reflect the MOST RECENT InvoiceDate, not the first or most frequent.

    Customer 9001: Spain (50 days before cutoff), Portugal (10 days before cutoff).
    Expected country: Portugal.
    """
    df = _make_multi_country_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    row = _get_cust(features, 9001)
    assert row["country"] == "Portugal", (
        f"Expected 'Portugal' (most recent), got '{row['country']}'"
    )


def test_country_not_first_invoice_country():
    """Complement of the previous test: country must NOT be Spain (the earlier one)."""
    df = _make_multi_country_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    row = _get_cust(features, 9001)
    assert row["country"] != "Spain"


def test_single_invoice_customer_tenure_zero():
    """
    A customer with exactly one invoice must have tenure_days == 0 because
    first_purchase == last_purchase.
    """
    df = _make_exact_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    row = _get_cust(features, 3003)
    assert row["tenure_days"] == 0


def test_avg_basket_value_equals_monetary_for_single_invoice_customer():
    """
    When frequency == 1, avg_basket_value == monetary (one basket = total spend).
    """
    df = _make_exact_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    row = _get_cust(features, 2002)
    assert math.isclose(row["avg_basket_value"], row["monetary"], rel_tol=1e-9)


def test_one_row_per_customer_exact_df():
    """The _make_exact_df has 3 pre-cutoff customers → exactly 3 rows."""
    df = _make_exact_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    assert len(features) == 3
    assert features["Customer ID"].nunique() == 3


# ---------------------------------------------------------------------------
# GROUP 4 — build_target: structural and exact-value checks
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


def test_exact_future_revenue_customer_2002():
    """
    Customer 2002 post-cutoff: INV-D, Qty=2, Price=6.00 → future_revenue = 12.00.
    log1p(12.00) = ln(13) ≈ 2.5649.
    """
    df = _make_exact_df()
    target = build_target(df, PREDICTION_DATE)
    row = target[target["Customer ID"] == 2002].iloc[0]
    assert math.isclose(row["future_revenue"], 12.00, rel_tol=1e-9)
    assert math.isclose(row["log1p_future_revenue"], math.log(13), rel_tol=1e-9)


def test_build_target_excludes_historical_only_customers():
    """
    Customers 1001 and 3003 have no future transactions; they must NOT appear
    in the raw target DataFrame (they receive 0 only after the left-join).
    """
    df = _make_exact_df()
    target = build_target(df, PREDICTION_DATE)
    assert 1001 not in target["Customer ID"].tolist()
    assert 3003 not in target["Customer ID"].tolist()


def test_build_target_uses_only_post_cutoff_transactions():
    """
    Modify a historical transaction to have a very large line_total.
    The target must not change because it only reads post-cutoff rows.
    """
    df = _make_exact_df()
    # Inflate a historical transaction
    df.loc[df["Invoice"] == "INV-C", "line_total"] = 999999.0
    # INV-C is pre-cutoff; target must still reflect only INV-D (12.00)
    target = build_target(df, PREDICTION_DATE)
    row = target[target["Customer ID"] == 2002].iloc[0]
    assert math.isclose(row["future_revenue"], 12.00, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# GROUP 5 — build_customer_dataset: join logic and guarantees
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


def test_build_customer_dataset_group_a_future_revenue_is_genuine_zero():
    """
    Customer 1001 has no future transactions → future_revenue must be exactly 0.0,
    not NaN.  A zero is a genuine observation (customer did not return), not missing data.
    """
    df = _make_exact_df()
    customer_df = build_customer_dataset(df, PREDICTION_DATE)
    row_1001 = customer_df[customer_df["Customer ID"] == 1001].iloc[0]
    assert row_1001["future_revenue"] == 0.0
    assert not math.isnan(row_1001["future_revenue"])


def test_build_customer_dataset_group_a_log1p_is_zero():
    """log1p(0) == 0.  Customer with zero future revenue must have log1p == 0."""
    df = _make_exact_df()
    customer_df = build_customer_dataset(df, PREDICTION_DATE)
    row_1001 = customer_df[customer_df["Customer ID"] == 1001].iloc[0]
    assert row_1001["log1p_future_revenue"] == 0.0


def test_build_customer_dataset_exact_future_revenue_customer_2002():
    """
    Customer 2002 future transaction: 2 * 6.00 = 12.00.
    After join, future_revenue must be 12.00, log1p must be ln(13).
    """
    df = _make_exact_df()
    customer_df = build_customer_dataset(df, PREDICTION_DATE)
    row = customer_df[customer_df["Customer ID"] == 2002].iloc[0]
    assert math.isclose(row["future_revenue"], 12.00, rel_tol=1e-9)
    assert math.isclose(row["log1p_future_revenue"], math.log(13), rel_tol=1e-9)


def test_build_customer_dataset_one_row_per_customer():
    """No customer ID must appear more than once."""
    df = _make_exact_df()
    customer_df = build_customer_dataset(df, PREDICTION_DATE)
    assert customer_df["Customer ID"].nunique() == len(customer_df)


def test_build_customer_dataset_all_expected_columns_present():
    """Final dataset must contain all feature and target columns."""
    df = _make_exact_df()
    customer_df = build_customer_dataset(df, PREDICTION_DATE)
    required = {
        "Customer ID",
        "recency_days",
        "frequency",
        "monetary",
        "avg_basket_value",
        "unique_products",
        "tenure_days",
        "purchase_rate",
        "country",
        "future_revenue",
        "log1p_future_revenue",
    }
    assert required.issubset(set(customer_df.columns))


def test_build_customer_dataset_log1p_consistent_with_future_revenue():
    """log1p_future_revenue must equal np.log1p(future_revenue) for every row."""
    df = _make_exact_df()
    customer_df = build_customer_dataset(df, PREDICTION_DATE)
    recomputed = np.log1p(customer_df["future_revenue"])
    pd.testing.assert_series_equal(
        customer_df["log1p_future_revenue"].reset_index(drop=True),
        recomputed.reset_index(drop=True),
        check_names=False,
    )


# ---------------------------------------------------------------------------
# GROUP 6 — Leakage checklist
# ---------------------------------------------------------------------------


def test_leakage_customer_id_not_in_feature_columns():
    """
    Customer ID is the join key, not a predictor.
    The feature columns (all columns except the ID) must not be the ID itself.
    """
    df = _make_exact_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    feature_cols = [c for c in features.columns if c != "Customer ID"]
    assert "Customer ID" not in feature_cols


def test_leakage_x_uses_only_historical_transactions():
    """
    Inflating a post-cutoff transaction must not change any X feature.
    """
    df = _make_exact_df()
    features_before = build_rfm_features(df, PREDICTION_DATE)

    df_modified = df.copy()
    # Add an extra post-cutoff line for customer 1001 (who normally has no future data)
    extra = pd.DataFrame([{
        "Invoice": "INV-FUTURE-LEAK",
        "StockCode": "LEAK",
        "Quantity": 999,
        "Price": 999.0,
        "InvoiceDate": pd.Timestamp(PREDICTION_DATE) + pd.Timedelta(days=5),
        "Customer ID": pd.array([1001], dtype="Int64")[0],
        "Country": "United Kingdom",
        "line_total": 999 * 999.0,
    }])
    df_modified = pd.concat([df_modified, extra], ignore_index=True)

    features_after = build_rfm_features(df_modified, PREDICTION_DATE)

    row_before = _get_cust(features_before, 1001)
    row_after = _get_cust(features_after, 1001)

    for col in ["recency_days", "frequency", "monetary", "avg_basket_value",
                "unique_products", "tenure_days", "purchase_rate", "country"]:
        assert row_before[col] == row_after[col], (
            f"Feature '{col}' changed after adding a post-cutoff transaction "
            f"(before={row_before[col]}, after={row_after[col]}). "
            "This is a data leakage violation."
        )


def test_leakage_y_uses_only_future_transactions():
    """
    Inflating a pre-cutoff transaction must not change future_revenue.
    """
    df = _make_exact_df()
    target_before = build_target(df, PREDICTION_DATE)

    df_modified = df.copy()
    # Inflate the historical transaction for customer 2002
    df_modified.loc[df_modified["Invoice"] == "INV-C", "line_total"] = 999999.0

    target_after = build_target(df_modified, PREDICTION_DATE)

    row_before = target_before[target_before["Customer ID"] == 2002].iloc[0]
    row_after = target_after[target_after["Customer ID"] == 2002].iloc[0]

    assert math.isclose(row_before["future_revenue"], row_after["future_revenue"], rel_tol=1e-9), (
        "future_revenue changed after modifying a pre-cutoff transaction — leakage."
    )


def test_leakage_future_revenue_not_in_features():
    """
    The feature DataFrame returned by build_rfm_features must not contain
    'future_revenue' or 'log1p_future_revenue'.
    """
    df = _make_exact_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    assert "future_revenue" not in features.columns
    assert "log1p_future_revenue" not in features.columns


def test_leakage_no_future_transaction_count_in_features():
    """
    The feature DataFrame must not reflect the count of future transactions.
    Concretely, frequency for customer 2002 must be 1 (one historical invoice),
    not 2 (one historical + one future).
    """
    df = _make_exact_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    row = _get_cust(features, 2002)
    # Only INV-C is historical; INV-D is post-cutoff
    assert row["frequency"] == 1, (
        f"frequency={row['frequency']} but should be 1 (historical only). "
        "If 2, future invoice INV-D was incorrectly included."
    )


def test_leakage_no_future_monetary_in_features():
    """
    monetary for customer 2002 must reflect only historical spend (32.00),
    not historical + future (32.00 + 12.00 = 44.00).
    """
    df = _make_exact_df()
    features = build_rfm_features(df, PREDICTION_DATE)
    row = _get_cust(features, 2002)
    assert math.isclose(row["monetary"], 32.00, rel_tol=1e-9), (
        f"monetary={row['monetary']} but should be 32.00 (historical only). "
        "If 44.00, future transaction INV-D was incorrectly included."
    )
