"""
clean_data.py — Transaction-level cleaning for the Online Retail II dataset.

Responsibility
--------------
Apply a sequence of row-level filters that convert the combined raw DataFrame
into a reliable, gross-revenue transaction dataset ready for feature engineering.

Every public function follows the contract:
    takes a DataFrame  →  returns a filtered / transformed DataFrame

No feature engineering, no RFM aggregation, no target construction, no modelling.

Data-layer contract
--------------------
data/raw/      ← IMMUTABLE.  Source file is never touched here.
data/interim/  ← Output of this module: ``transactions_cleaned.csv``.
data/processed/ ← Written by the feature pipeline (Notebook 03+).
"""

from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from ecommerce_customer_value_prediction.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from ecommerce_customer_value_prediction.data.load_data import load_raw

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_EXCEL: Path = RAW_DATA_DIR / "online_retail_II.xlsx"
CLEANED_CSV: Path = INTERIM_DATA_DIR / "transactions_cleaned.csv"

# Standard product StockCode: exactly 5 digits, optionally one trailing letter.
STANDARD_STOCKCODE_RE: str = r"^\d{5}[a-zA-Z]?$"


# ---------------------------------------------------------------------------
# Step 2 — Missing Customer ID
# ---------------------------------------------------------------------------


def drop_missing_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where ``Customer ID`` is missing.

    Decision rationale
    ------------------
    * ``Customer ID`` is the join key for all customer-level RFM modelling.
    * Anonymous rows cannot be attributed to any real individual.
    * Imputation (single UNKNOWN, random IDs, most-frequent) all produce
      meaningless or noise-inflated customer features.
    * Excluding them reduces dataset size but introduces no bias — remaining
      customers are real, identifiable individuals.
    """
    before = len(df)
    df = df.dropna(subset=["Customer ID"]).copy()
    after = len(df)
    logger.info(
        f"[drop_missing_customer_id] Removed {before - after:,} rows "
        f"({(before - after) / before * 100:.2f}%). Remaining: {after:,}"
    )
    return df


# ---------------------------------------------------------------------------
# Step 3 — Duplicate rows
# ---------------------------------------------------------------------------


def drop_full_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the first occurrence of each fully-duplicated row; remove extras.

    Decision rationale
    ------------------
    * Full-row duplicates (every column identical) are data-entry/export errors.
    * They inflate transaction frequency and overstate customer revenue.
    * Only exact full-row duplicates are removed; ``(Invoice, StockCode)``
      duplicates with different quantities/prices are intentionally retained.
    """
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    logger.info(
        f"[drop_full_duplicates] Removed {before - after:,} duplicate rows. "
        f"Remaining: {after:,}"
    )
    return df


# ---------------------------------------------------------------------------
# Steps 4 & 5 — Quantity and Price
# ---------------------------------------------------------------------------


def keep_positive_qty_and_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retain only rows where ``Quantity > 0`` AND ``Price > 0``.

    Decision rationale
    ------------------
    * Non-positive Quantity: returns, cancellations, back-office adjustments.
    * Non-positive Price: free samples, giveaways, credit adjustments.
    * The primary gross-revenue experiment uses completed, priced purchases only.
    * Every surviving row has ``line_total > 0``, enabling safe log-transform.
    * C-prefix cancellations (almost always Qty < 0) are implicitly excluded.
      An explicit C-prefix filter follows for completeness.
    """
    before = len(df)
    mask = (df["Quantity"] > 0) & (df["Price"] > 0)
    df = df[mask].reset_index(drop=True)
    after = len(df)
    logger.info(
        f"[keep_positive_qty_and_price] Removed {before - after:,} rows "
        f"(Qty<=0 or Price<=0). Remaining: {after:,}"
    )
    return df


# ---------------------------------------------------------------------------
# Step 6 — Cancellation invoices
# ---------------------------------------------------------------------------


def drop_cancellation_invoices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows whose ``Invoice`` starts with ``'C'``.

    Decision rationale
    ------------------
    * C-prefix invoices denote returns or cancellations — they reverse a prior
      purchase and are not forward purchase events.
    * The positive-Qty filter already removes most C-rows; this step catches
      the rare C-rows that have positive Quantity (data anomalies).
    * Raw data is untouched — C-rows remain available for a net-revenue
      sensitivity analysis.
    """
    before = len(df)
    df = df[~df["Invoice"].str.startswith("C")].reset_index(drop=True)
    after = len(df)
    logger.info(
        f"[drop_cancellation_invoices] Removed {before - after:,} C-prefix rows. "
        f"Remaining: {after:,}"
    )
    return df


# ---------------------------------------------------------------------------
# Step 7 — Non-standard StockCodes
# ---------------------------------------------------------------------------


def drop_nonstandard_stockcodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retain only rows whose ``StockCode`` matches ``^\\d{5}[a-zA-Z]?$``.

    Removes system / non-product lines: POST, DOT, AMAZONFEE, BANK CHARGES,
    M, C2, CRUK, PADS, etc.

    Decision rationale
    ------------------
    * System charges inflate transaction frequency and monetary value.
    * They are not customer purchase behaviour.
    * Including postage would make a frequent buyer look even more engaged —
      but that engagement is just a proxy for order count, not product revenue.
    """
    before = len(df)
    std_mask = df["StockCode"].str.match(STANDARD_STOCKCODE_RE, na=False)
    df = df[std_mask].reset_index(drop=True)
    after = len(df)
    logger.info(
        f"[drop_nonstandard_stockcodes] Removed {before - after:,} system-code rows. "
        f"Remaining: {after:,}"
    )
    return df


# ---------------------------------------------------------------------------
# Step 9 — Data types
# ---------------------------------------------------------------------------


def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce canonical column types.

    +-------------+---------------+--------------------------------------------------+
    | Column      | Type          | Rationale                                        |
    +=============+===============+==================================================+
    | Invoice     | str           | Identifier; C-prefix values exist               |
    | StockCode   | str           | Categorical; non-numeric codes exist            |
    | Description | str / NA      | Free-text label; NaN preserved                  |
    | Quantity    | int           | Positive integer after filtering                |
    | InvoiceDate | datetime64    | Required for recency, tenure, temporal splits   |
    | Price       | float         | Unit price for revenue calculation              |
    | Customer ID | Int64         | Nullable integer (avoids float representation)  |
    | Country     | str           | Categorical country name                        |
    +-------------+---------------+--------------------------------------------------+
    """
    df = df.copy()

    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["Description"] = df["Description"].astype(str).where(
        df["Description"].notna(), other=None
    )
    df["Quantity"] = df["Quantity"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Price"] = df["Price"].astype(float)
    df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors="coerce").astype("Int64")
    df["Country"] = df["Country"].astype(str)

    logger.info("[enforce_dtypes] Data types enforced.")
    return df


# ---------------------------------------------------------------------------
# Step 10 — line_total
# ---------------------------------------------------------------------------


def create_line_total(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``line_total = Quantity × Price``.

    This column is computed **after** all filters so that every surviving row
    has ``Quantity > 0`` and ``Price > 0``, guaranteeing ``line_total > 0``
    and making ``log(line_total)`` always safe and well-defined.
    """
    df = df.copy()
    df["line_total"] = df["Quantity"] * df["Price"]
    assert (df["line_total"] > 0).all(), "line_total must be positive after filtering"
    logger.info(
        f"[create_line_total] Created line_total. "
        f"Range: {df['line_total'].min():.2f} – {df['line_total'].max():,.2f}"
    )
    return df


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_cleaned(df: pd.DataFrame, output_path: Path = CLEANED_CSV) -> None:
    """
    Write the cleaned transaction dataset to ``data/interim/``.

    CSV is used over Parquet for transparency — the interim file is
    human-readable so analysts can inspect it directly without tooling.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Cleaned data saved → {output_path}  ({len(df):,} rows)")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_cleaning_pipeline(
    excel_path: Path = RAW_EXCEL,
    output_path: Path = CLEANED_CSV,
    save: bool = True,
) -> pd.DataFrame:
    """
    Execute every cleaning step in order and return the cleaned DataFrame.

    Steps
    -----
    1. Load raw data from the Excel workbook.
    2. Drop rows with missing Customer ID.
    3. Drop full-row duplicate transactions.
    4. Keep only rows with Quantity > 0 and Price > 0 (gross revenue).
    5. Drop C-prefix cancellation invoices.
    6. Drop non-standard StockCode system lines.
    7. Enforce canonical data types.
    8. Create line_total = Quantity × Price.
    9. Validate the cleaned DataFrame.
    10. Optionally save to ``data/interim/transactions_cleaned.csv``.

    Parameters
    ----------
    excel_path : Path
        Source workbook (default ``data/raw/online_retail_II.xlsx``).
    output_path : Path
        Destination CSV (default ``data/interim/transactions_cleaned.csv``).
    save : bool
        If ``True``, write the result to *output_path*.

    Returns
    -------
    pd.DataFrame
        Cleaned transaction-level DataFrame.
    """
    # Import here to avoid circular import (validate_data imports from clean_data)
    from ecommerce_customer_value_prediction.data.validate_data import validate_cleaned

    steps = [
        ("Load raw data", lambda _: load_raw(excel_path)),
        ("Drop missing Customer ID", drop_missing_customer_id),
        ("Drop full duplicates", drop_full_duplicates),
        ("Keep Qty>0 and Price>0", keep_positive_qty_and_price),
        ("Drop C-prefix invoices", drop_cancellation_invoices),
        ("Drop non-standard StockCode", drop_nonstandard_stockcodes),
        ("Enforce data types", enforce_dtypes),
        ("Create line_total", create_line_total),
    ]

    df = None
    for label, fn in tqdm(steps, desc="Cleaning pipeline"):
        df = fn(df)  # first step (load) ignores its argument
        logger.info(f"  ✓ {label} — {len(df):,} rows")

    validate_cleaned(df)

    if save:
        save_cleaned(df, output_path)

    return df
