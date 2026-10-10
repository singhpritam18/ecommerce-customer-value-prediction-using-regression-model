"""Data sub-package: loading, cleaning, and validation."""

from ecommerce_customer_value_prediction.data.load_data import load_raw
from ecommerce_customer_value_prediction.data.clean_data import (
    drop_missing_customer_id,
    drop_full_duplicates,
    keep_positive_qty_and_price,
    drop_cancellation_invoices,
    drop_nonstandard_stockcodes,
    enforce_dtypes,
    create_line_total,
    save_cleaned,
    run_cleaning_pipeline,
    STANDARD_STOCKCODE_RE,
    RAW_EXCEL,
    CLEANED_CSV,
)
from ecommerce_customer_value_prediction.data.validate_data import (
    validate_cleaned,
    report_missing_customer_id,
    report_duplicates,
    report_quantity,
    report_price,
    report_cancellations,
    report_nonstandard_stockcodes,
    report_missing_description,
)

__all__ = [
    # load
    "load_raw",
    # clean
    "STANDARD_STOCKCODE_RE",
    "RAW_EXCEL",
    "CLEANED_CSV",
    "drop_missing_customer_id",
    "drop_full_duplicates",
    "keep_positive_qty_and_price",
    "drop_cancellation_invoices",
    "drop_nonstandard_stockcodes",
    "enforce_dtypes",
    "create_line_total",
    "save_cleaned",
    "run_cleaning_pipeline",
    # validate / report
    "validate_cleaned",
    "report_missing_customer_id",
    "report_duplicates",
    "report_quantity",
    "report_price",
    "report_cancellations",
    "report_nonstandard_stockcodes",
    "report_missing_description",
]
