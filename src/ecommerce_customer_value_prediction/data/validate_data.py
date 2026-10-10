"""
validate_data.py — Schema, quality, and constraint validation.

Responsibility
--------------
* Report diagnostic counts for each quality issue (used by notebooks).
* Assert hard constraints on the cleaned DataFrame (used by pipelines).

No filtering happens here.  Every function either returns a diagnostic
dict/DataFrame or raises ``AssertionError`` / ``ValueError``.
"""

import pandas as pd
from loguru import logger

from ecommerce_customer_value_prediction.data.clean_data import STANDARD_STOCKCODE_RE


# ---------------------------------------------------------------------------
# Diagnostic reporters  (used by notebooks for EDA narrative)
# ---------------------------------------------------------------------------


def report_missing_customer_id(df: pd.DataFrame) -> dict:
    """
    Return diagnostic counts about rows with missing ``Customer ID``.

    Returns
    -------
    dict with keys:
        n_rows_missing_cid, pct_missing_cid,
        revenue_in_missing_cid_rows, revenue_total, revenue_pct_missing
    """
    mask = df["Customer ID"].isnull()
    n_total = len(df)
    n_missing = int(mask.sum())

    df_temp = df.copy()
    df_temp["_lt"] = df_temp["Quantity"] * df_temp["Price"]
    revenue_missing = float(df_temp.loc[mask, "_lt"].sum())
    revenue_total = float(df_temp["_lt"].sum())

    return {
        "n_rows_missing_cid": n_missing,
        "pct_missing_cid": round(n_missing / n_total * 100, 2),
        "revenue_in_missing_cid_rows": round(revenue_missing, 2),
        "revenue_total": round(revenue_total, 2),
        "revenue_pct_missing": round(revenue_missing / revenue_total * 100, 2)
        if revenue_total
        else 0.0,
    }


def report_duplicates(df: pd.DataFrame) -> dict:
    """
    Count full-row duplicates and ``(Invoice, StockCode)`` duplicates.

    Returns
    -------
    dict with keys:
        full_dup_rows_total, full_dup_extra_to_remove,
        inv_sku_dup_rows, ambiguous_inv_sku_dup_rows
    """
    all_cols = list(df.columns)
    full_dup_mask = df.duplicated(subset=all_cols, keep=False)
    full_dup_extra = int(df.duplicated(subset=all_cols, keep="first").sum())
    inv_sku_dup_mask = df.duplicated(subset=["Invoice", "StockCode"], keep=False)
    ambiguous_mask = inv_sku_dup_mask & ~full_dup_mask

    return {
        "full_dup_rows_total": int(full_dup_mask.sum()),
        "full_dup_extra_to_remove": full_dup_extra,
        "inv_sku_dup_rows": int(inv_sku_dup_mask.sum()),
        "ambiguous_inv_sku_dup_rows": int(ambiguous_mask.sum()),
    }


def report_quantity(df: pd.DataFrame) -> dict:
    """Return counts of zero, negative, and positive Quantity rows."""
    return {
        "qty_zero": int((df["Quantity"] == 0).sum()),
        "qty_negative": int((df["Quantity"] < 0).sum()),
        "qty_positive": int((df["Quantity"] > 0).sum()),
    }


def report_price(df: pd.DataFrame) -> dict:
    """Return counts of zero, negative, and positive Price rows."""
    return {
        "price_zero": int((df["Price"] == 0).sum()),
        "price_negative": int((df["Price"] < 0).sum()),
        "price_positive": int((df["Price"] > 0).sum()),
    }


def report_cancellations(df: pd.DataFrame) -> dict:
    """Return count of C-prefix invoice rows."""
    return {"c_prefix_rows": int(df["Invoice"].str.startswith("C").sum())}


def report_nonstandard_stockcodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a frequency table of non-standard ``StockCode`` values.

    Returns
    -------
    pd.DataFrame with columns ``["StockCode", "count"]``, sorted descending.
    """
    std_mask = df["StockCode"].str.match(STANDARD_STOCKCODE_RE, na=False)
    ns = df[~std_mask]["StockCode"].value_counts().reset_index()
    ns.columns = ["StockCode", "count"]
    return ns


def report_missing_description(df: pd.DataFrame) -> dict:
    """
    Count rows with missing ``Description``.

    Note: ``Description`` is **not** required for RFM or revenue target
    construction; rows with missing description are retained by design.
    """
    n_missing = int(df["Description"].isnull().sum())
    n_valid_cid = int(df[df["Description"].isnull()]["Customer ID"].notna().sum())
    return {
        "missing_description": n_missing,
        "missing_desc_but_valid_cid": n_valid_cid,
    }


# ---------------------------------------------------------------------------
# Hard validation  (used by pipelines; raises on failure)
# ---------------------------------------------------------------------------


def validate_cleaned(df: pd.DataFrame) -> None:
    """
    Assert hard quality constraints on the cleaned DataFrame.

    Raises
    ------
    AssertionError
        If one or more constraints are violated. All failures are reported
        together so the caller sees the full picture in one run.

    Constraints checked
    --------------------
    1. No missing ``Customer ID``        — prevents ghost customers in RFM.
    2. No full-row duplicates            — prevents inflated frequency/revenue.
    3. All ``Quantity > 0``              — ensures gross-purchase rows only.
    4. All ``Price > 0``                 — ensures non-zero, log-safe revenue.
    5. ``InvoiceDate`` is datetime64     — required for temporal operations.
    6. All ``line_total > 0`` (if col exists) — end-to-end revenue sanity.
    7. No C-prefix ``Invoice``           — confirms cancellation exclusion.
    8. All ``StockCode`` standard        — confirms system-line exclusion.
    """
    errors: list[str] = []

    if df["Customer ID"].isnull().any():
        n = int(df["Customer ID"].isnull().sum())
        errors.append(f"FAIL: {n:,} missing Customer ID(s) found.")

    if df.duplicated().any():
        n = int(df.duplicated().sum())
        errors.append(f"FAIL: {n:,} full-row duplicate(s) found.")

    if not (df["Quantity"] > 0).all():
        n = int((df["Quantity"] <= 0).sum())
        errors.append(f"FAIL: {n:,} row(s) with Quantity <= 0.")

    if not (df["Price"] > 0).all():
        n = int((df["Price"] <= 0).sum())
        errors.append(f"FAIL: {n:,} row(s) with Price <= 0.")

    if not pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
        errors.append(f"FAIL: InvoiceDate dtype is {df['InvoiceDate'].dtype}, expected datetime64.")

    if "line_total" in df.columns and not (df["line_total"] > 0).all():
        n = int((df["line_total"] <= 0).sum())
        errors.append(f"FAIL: {n:,} row(s) with line_total <= 0.")

    if df["Invoice"].str.startswith("C").any():
        n = int(df["Invoice"].str.startswith("C").sum())
        errors.append(f"FAIL: {n:,} C-prefix invoice row(s) remain.")

    std_mask = df["StockCode"].str.match(STANDARD_STOCKCODE_RE, na=False)
    if not std_mask.all():
        n = int((~std_mask).sum())
        errors.append(f"FAIL: {n:,} non-standard StockCode row(s) remain.")

    if errors:
        msg = f"Validation failed with {len(errors)} error(s):\n" + "\n".join(errors)
        for e in errors:
            logger.error(e)
        raise AssertionError(msg)

    logger.success("All validation checks PASSED.")


def validate_schema(df: pd.DataFrame, expected_columns: list[str] | None = None) -> None:
    """
    Check that ``df`` contains the expected column set.

    Parameters
    ----------
    df : pd.DataFrame
    expected_columns : list[str] | None
        If ``None``, defaults to the standard Online Retail II columns.

    Raises
    ------
    ValueError if any expected column is missing.
    """
    if expected_columns is None:
        expected_columns = [
            "Invoice",
            "StockCode",
            "Description",
            "Quantity",
            "InvoiceDate",
            "Price",
            "Customer ID",
            "Country",
        ]
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing expected column(s): {missing}")
    logger.info(f"[validate_schema] All {len(expected_columns)} expected columns present.")
