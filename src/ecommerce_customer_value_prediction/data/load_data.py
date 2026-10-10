"""
load_data.py — Raw data loading for the Online Retail II dataset.

Responsibility
--------------
Load the two sheets from the raw Excel workbook and return a single combined
DataFrame with normalised column types. This module does NOT clean or filter data.

Data-layer contract
--------------------
data/raw/   ← IMMUTABLE.  online_retail_II.xlsx is NEVER overwritten here.
data/interim/ ← Written by clean_data.py after filtering.
data/processed/ ← Written by the feature pipeline (Notebook 03+).
"""

from pathlib import Path

import pandas as pd
from loguru import logger

from ecommerce_customer_value_prediction.config import RAW_DATA_DIR

# Default location of the source workbook
RAW_EXCEL: Path = RAW_DATA_DIR / "online_retail_II.xlsx"

# Columns added only for Notebook 01 exploration — must not persist downstream
_EXPLORATION_COLS: frozenset[str] = frozenset({"sheet", "ym", "line_total_tmp", "line_total"})


def load_raw(excel_path: Path = RAW_EXCEL) -> pd.DataFrame:
    """
    Load and combine the two sheets from the Online Retail II workbook.

    Parameters
    ----------
    excel_path : Path
        Location of ``online_retail_II.xlsx``.  Defaults to ``data/raw/``.

    Returns
    -------
    pd.DataFrame
        Combined raw transactions with ``Invoice`` normalised to ``str``.
        Any exploration-only columns (``sheet``, ``ym``, ``line_total_tmp``,
        ``line_total``) are dropped if present.

    Notes
    -----
    Why two sheets are combined here
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Both sheets share an identical column schema.  Combining in code keeps
    notebooks DRY and makes re-loading trivial from any downstream module.

    Why the raw file is immutable
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * **Reproducibility** — any analyst can re-run from the original source.
    * **Auditability** — revised cleaning decisions can be compared against
      the original.
    * **Safety** — accidental overwrites are prevented.
    """
    logger.info(f"Loading raw workbook: {excel_path}")

    if not excel_path.exists():
        raise FileNotFoundError(
            f"Raw workbook not found: {excel_path}\n"
            "Download online_retail_II.xlsx from the UCI ML Repository and place "
            "it under data/raw/."
        )

    df_0910 = pd.read_excel(excel_path, sheet_name="Year 2009-2010")
    logger.info(f"  Year 2009-2010 : {df_0910.shape[0]:,} rows")

    df_1011 = pd.read_excel(excel_path, sheet_name="Year 2010-2011")
    logger.info(f"  Year 2010-2011 : {df_1011.shape[0]:,} rows")

    df = pd.concat([df_0910, df_1011], ignore_index=True)
    logger.info(f"  Combined       : {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Drop exploration-only columns left over from Notebook 01
    to_drop = _EXPLORATION_COLS & set(df.columns)
    if to_drop:
        df = df.drop(columns=list(to_drop))
        logger.info(f"  Dropped exploration columns: {sorted(to_drop)}")

    # Normalise Invoice to str so C-prefix checks work reliably downstream
    df["Invoice"] = df["Invoice"].astype(str)

    return df
