"""
tests/test_data.py — Unit tests for the data sub-package.

All tests use synthetic in-memory DataFrames so no real dataset is required.
"""

import pandas as pd
import pytest

from ecommerce_customer_value_prediction.data.clean_data import (
    STANDARD_STOCKCODE_RE,
    create_line_total,
    drop_cancellation_invoices,
    drop_full_duplicates,
    drop_missing_customer_id,
    drop_nonstandard_stockcodes,
    enforce_dtypes,
    keep_positive_qty_and_price,
)
from ecommerce_customer_value_prediction.data.validate_data import (
    report_cancellations,
    report_duplicates,
    report_missing_customer_id,
    report_price,
    report_quantity,
    validate_cleaned,
    validate_schema,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_raw_df() -> pd.DataFrame:
    """Minimal synthetic raw DataFrame that mirrors the Online Retail II schema."""
    return pd.DataFrame(
        {
            # Rows 4 and 5 are TRUE full-row duplicates (identical Invoice+all other cols)
            "Invoice": ["536365", "536365", "C536379", "536382", "536383", "536383"],
            "StockCode": ["85123A", "71053", "85123A", "POST", "22771", "22771"],
            "Description": ["WHITE HANGING HEART", "WHITE METAL LANTERN", None, "POSTAGE", "AJANTA", "AJANTA"],
            "Quantity": [6, 6, -1, 1, 12, 12],
            "InvoiceDate": pd.to_datetime(
                [
                    "2010-12-01 08:26:00",
                    "2010-12-01 08:26:00",
                    "2010-12-01 09:00:00",
                    "2010-12-01 10:00:00",
                    "2010-12-02 11:00:00",
                    "2010-12-02 11:00:00",  # true full-row duplicate of row above
                ]
            ),
            "Price": [2.55, 3.39, 2.55, 18.0, 1.25, 1.25],
            "Customer ID": [17850.0, 17850.0, 17850.0, None, 13047.0, 13047.0],
            "Country": ["United Kingdom"] * 6,
        }
    )


def _make_clean_df() -> pd.DataFrame:
    """A clean DataFrame that should pass all validation checks."""
    df = pd.DataFrame(
        {
            "Invoice": ["536365", "536366", "536367"],
            "StockCode": ["85123A", "71053", "22771"],
            "Description": ["Item A", "Item B", "Item C"],
            "Quantity": [6, 3, 12],
            "InvoiceDate": pd.to_datetime(
                ["2010-12-01 08:26:00", "2010-12-02 09:00:00", "2011-01-15 14:30:00"]
            ),
            "Price": [2.55, 3.39, 1.25],
            "Customer ID": pd.array([17850, 17851, 13047], dtype="Int64"),
            "Country": ["United Kingdom", "France", "Germany"],
            "line_total": [15.30, 10.17, 15.00],
        }
    )
    return df


# ---------------------------------------------------------------------------
# Tests: drop_missing_customer_id
# ---------------------------------------------------------------------------


def test_drop_missing_customer_id_removes_correct_rows():
    df = _make_raw_df()
    result = drop_missing_customer_id(df)
    assert result["Customer ID"].isnull().sum() == 0
    assert len(result) == len(df) - 1  # one row had NaN Customer ID


def test_drop_missing_customer_id_pure(tmp_path):
    """Function must not mutate the input DataFrame."""
    df = _make_raw_df()
    original_len = len(df)
    drop_missing_customer_id(df)
    assert len(df) == original_len


# ---------------------------------------------------------------------------
# Tests: drop_full_duplicates
# ---------------------------------------------------------------------------


def test_drop_full_duplicates_removes_extras():
    df = _make_raw_df()
    # Rows 4 and 5 are identical → one should be removed
    result = drop_full_duplicates(df)
    assert len(result) == len(df) - 1


def test_drop_full_duplicates_no_dupes_unchanged():
    df = _make_clean_df().drop(columns=["line_total"])
    result = drop_full_duplicates(df)
    assert len(result) == len(df)


# ---------------------------------------------------------------------------
# Tests: keep_positive_qty_and_price
# ---------------------------------------------------------------------------


def test_keep_positive_qty_and_price_removes_negatives():
    df = _make_raw_df()
    result = keep_positive_qty_and_price(df)
    assert (result["Quantity"] > 0).all()
    assert (result["Price"] > 0).all()


def test_keep_positive_qty_and_price_removes_zero_price():
    df = _make_raw_df().copy()
    df.loc[0, "Price"] = 0.0
    result = keep_positive_qty_and_price(df)
    assert 0.0 not in result["Price"].values


# ---------------------------------------------------------------------------
# Tests: drop_cancellation_invoices
# ---------------------------------------------------------------------------


def test_drop_cancellation_invoices_removes_c_prefix():
    df = _make_raw_df()
    # Invoice column was already cast to str by load_raw; simulate that here
    df["Invoice"] = df["Invoice"].astype(str)
    result = drop_cancellation_invoices(df)
    assert not result["Invoice"].str.startswith("C").any()


def test_drop_cancellation_invoices_keeps_others():
    df = _make_raw_df()
    df["Invoice"] = df["Invoice"].astype(str)
    result = drop_cancellation_invoices(df)
    # Should keep all rows whose invoice does NOT start with C
    expected_len = (~df["Invoice"].str.startswith("C")).sum()
    assert len(result) == expected_len


# ---------------------------------------------------------------------------
# Tests: drop_nonstandard_stockcodes
# ---------------------------------------------------------------------------


def test_drop_nonstandard_stockcodes_removes_post():
    df = _make_raw_df()
    result = drop_nonstandard_stockcodes(df)
    assert "POST" not in result["StockCode"].values


def test_drop_nonstandard_stockcodes_keeps_standard():
    df = _make_raw_df()
    result = drop_nonstandard_stockcodes(df)
    assert result["StockCode"].str.match(STANDARD_STOCKCODE_RE).all()


# ---------------------------------------------------------------------------
# Tests: enforce_dtypes
# ---------------------------------------------------------------------------


def test_enforce_dtypes_invoice_is_str():
    df = _make_raw_df()
    df = drop_missing_customer_id(df)
    result = enforce_dtypes(df)
    # pandas 3.x may return StringDtype instead of object; both are valid string types
    assert pd.api.types.is_string_dtype(result["Invoice"])


def test_enforce_dtypes_invoicedate_is_datetime():
    df = _make_raw_df()
    df = drop_missing_customer_id(df)
    result = enforce_dtypes(df)
    assert pd.api.types.is_datetime64_any_dtype(result["InvoiceDate"])


def test_enforce_dtypes_customer_id_is_int64():
    df = _make_raw_df()
    df = drop_missing_customer_id(df)
    result = enforce_dtypes(df)
    assert str(result["Customer ID"].dtype) == "Int64"


# ---------------------------------------------------------------------------
# Tests: create_line_total
# ---------------------------------------------------------------------------


def test_create_line_total_all_positive():
    df = _make_clean_df().drop(columns=["line_total"])
    df["Customer ID"] = df["Customer ID"].astype("Int64")
    result = create_line_total(df)
    assert (result["line_total"] > 0).all()


def test_create_line_total_correct_value():
    df = _make_clean_df().drop(columns=["line_total"])
    df["Customer ID"] = df["Customer ID"].astype("Int64")
    result = create_line_total(df)
    expected = df["Quantity"] * df["Price"]
    pd.testing.assert_series_equal(result["line_total"], expected, check_names=False)


# ---------------------------------------------------------------------------
# Tests: validate_cleaned
# ---------------------------------------------------------------------------


def test_validate_cleaned_passes_on_good_data():
    df = _make_clean_df()
    # Should not raise
    validate_cleaned(df)


def test_validate_cleaned_raises_on_missing_cid():
    df = _make_clean_df()
    df.loc[0, "Customer ID"] = pd.NA
    with pytest.raises(AssertionError, match="Customer ID"):
        validate_cleaned(df)


def test_validate_cleaned_raises_on_c_invoice():
    df = _make_clean_df()
    df.loc[0, "Invoice"] = "C536365"
    with pytest.raises(AssertionError, match="C-prefix"):
        validate_cleaned(df)


def test_validate_cleaned_raises_on_nonpositive_qty():
    df = _make_clean_df()
    df.loc[0, "Quantity"] = 0
    with pytest.raises(AssertionError, match="Quantity"):
        validate_cleaned(df)


def test_validate_cleaned_raises_on_bad_stockcode():
    df = _make_clean_df()
    df.loc[0, "StockCode"] = "POST"
    with pytest.raises(AssertionError, match="StockCode"):
        validate_cleaned(df)


# ---------------------------------------------------------------------------
# Tests: validate_schema
# ---------------------------------------------------------------------------


def test_validate_schema_passes_for_expected_columns():
    df = _make_clean_df()
    validate_schema(df)  # should not raise


def test_validate_schema_raises_on_missing_column():
    df = _make_clean_df().drop(columns=["Country"])
    with pytest.raises(ValueError, match="Country"):
        validate_schema(df)


# ---------------------------------------------------------------------------
# Tests: diagnostic reporters
# ---------------------------------------------------------------------------


def test_report_missing_customer_id_counts_correctly():
    df = _make_raw_df()
    report = report_missing_customer_id(df)
    assert report["n_rows_missing_cid"] == 1
    assert 0 < report["pct_missing_cid"] < 100


def test_report_duplicates_detects_full_dup():
    df = _make_raw_df()
    report = report_duplicates(df)
    # Rows 4 and 5 are identical
    assert report["full_dup_rows_total"] == 2
    assert report["full_dup_extra_to_remove"] == 1


def test_report_quantity_counts():
    df = _make_raw_df()
    report = report_quantity(df)
    assert report["qty_negative"] == 1
    assert report["qty_positive"] >= 1


def test_report_price_no_negatives():
    df = _make_raw_df()
    report = report_price(df)
    assert report["price_negative"] == 0


def test_report_cancellations():
    df = _make_raw_df()
    df["Invoice"] = df["Invoice"].astype(str)
    report = report_cancellations(df)
    assert report["c_prefix_rows"] == 1
