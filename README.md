# E-Commerce Customer Value Prediction

Predict future 4-month gross revenue for each customer using RFM-style features and Ridge regression on the [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) dataset (UCI ML Repository).

---

## Project Structure

```
ecommerce-customer-value-prediction/
│
├── data/                                  ← All data (never commit raw data to git)
│   ├── raw/                               ← IMMUTABLE source files; downloaded once
│   │   └── online_retail_II.xlsx          ← UCI Online Retail II dataset
│   ├── interim/                           ← Cleaned transaction-level output
│   │   └── transactions_cleaned.csv       ← Produced by `make clean-data`
│   └── processed/                         ← Customer-level modelling dataset
│       └── customer_dataset.csv           ← Produced by `make train` (feature + target)
│
├── models/                                ← Serialised model artifacts
│   ├── ridge_pipeline.pkl                 ← Fitted sklearn Pipeline (pickle)
│   └── ridge_pipeline_metadata.json       ← Training config & statistics
│
├── notebooks/                             ← Jupyter notebooks for exploration & reporting
│   ├── 01-initial-data-exploration.ipynb  ← Schema, missing values, distributions, time range
│   └── 02-data-cleaning.ipynb             ← Step-by-step cleaning with decision narrative
│
├── reports/
│   ├── figures/                           ← Saved plots (EDA charts, residual plots, etc.)
│   └── metrics/
│       └── evaluation_metrics.json        ← Hold-out test metrics (R², RMSE, MAE in log+GBP)
│
├── src/                                   ← Installable Python package (`pip install -e .`)
│   └── ecommerce_customer_value_prediction/
│       │
│       ├── __init__.py                    ← Package version & top-level re-exports
│       │
│       ├── config/                        ← Centralised project configuration
│       │   ├── __init__.py                ← Re-exports all path constants
│       │   └── settings.py                ← PROJ_ROOT, *_DATA_DIR, MODELS_DIR, METRICS_DIR
│       │
│       ├── data/                          ← Raw → interim data pipeline
│       │   ├── __init__.py                ← Re-exports run_cleaning_pipeline, validate_cleaned
│       │   ├── load_data.py               ← load_raw_data(): reads Excel workbook → DataFrame
│       │   ├── clean_data.py              ← run_cleaning_pipeline(): 6-step filtering + type coercion
│       │   │                              │   • drop_missing_customer_id
│       │   │                              │   • drop_full_duplicates
│       │   │                              │   • keep_positive_qty_and_price
│       │   │                              │   • drop_cancellation_invoices (C-prefix)
│       │   │                              │   • drop_nonstandard_stockcodes (POST, DOT, …)
│       │   │                              │   • enforce_dtypes + create_line_total
│       │   └── validate_data.py           ← Two layers of validation:
│       │                                  │   • Diagnostic reporters (used by notebooks)
│       │                                  │   • validate_cleaned() hard assertions (used by pipelines)
│       │
│       ├── features/                      ← Interim → processed feature engineering
│       │   ├── __init__.py                ← Re-exports build_customer_dataset, PREDICTION_DATE
│       │   └── build_features.py          ← build_rfm_features()  → per-customer RFM features
│       │                                  │   build_target()        → future_revenue + log1p transform
│       │                                  │   build_customer_dataset() → join features + target
│       │                                  │   save_customer_dataset()  → write to data/processed/
│       │
│       ├── models/                        ← Model training, evaluation, and inference
│       │   ├── __init__.py                ← Re-exports train, evaluate, predict, load_model, save_model
│       │   ├── train.py                   ← build_pipeline() → sklearn Pipeline (StandardScaler + OHE + Ridge)
│       │   │                              │   train()      → fit pipeline, return metadata dict
│       │   │                              │   save_model() → pickle pipeline + JSON metadata
│       │   │                              │   load_model() → deserialise from disk
│       │   ├── evaluate.py                ← evaluate() → RMSE/MAE/R² in log-space and GBP-space
│       │   │                              │   saves results to reports/metrics/evaluation_metrics.json
│       │   └── predict.py                 ← predict() → load pipeline + append pred columns to DataFrame
│       │                                  │   pred_log1p_future_revenue, pred_future_revenue (GBP)
│       │
│       └── pipelines/                     ← End-to-end workflow orchestration
│           ├── __init__.py                ← Re-exports run_training_pipeline
│           └── training_pipeline.py       ← run_training_pipeline():
│                                          │   1. Load / generate cleaned transactions
│                                          │   2. Build customer dataset (features + target)
│                                          │   3. Stratified 80/20 train/test split
│                                          │   4. Fit pipeline on train, evaluate on test
│                                          │   5. Refit final pipeline on full data, save model
│
├── tests/                                 ← Pytest test suite (no real dataset required)
│   ├── test_data.py                       ← 27 tests: cleaning functions + diagnostic reporters
│   ├── test_features.py                   ← 16 tests: RFM features, target, Group A/B/C logic
│   └── test_models.py                     ← 18 tests: train / evaluate / predict / save-load
│
├── docs/                                  ← MkDocs documentation source
│   └── docs/
│       ├── index.md
│       ├── getting-started.md
│       └── problem-formulation.md
│
├── references/                            ← Data dictionaries, papers, background reading
├── .gitignore
├── LICENSE                                ← MIT
├── Makefile                               ← make install | clean-data | train | test | lint | format
├── pyproject.toml                         ← Build system (hatchling), dependencies, pytest config
├── requirements.txt                       ← Pinned environment snapshot (pip freeze)
└── README.md
```

---

## Setup

### 1. Prerequisites

- Python ≥ 3.10
- `pip`

### 2. Clone and install

```bash
git clone https://github.com/singhpritam18/ecommerce-customer-value-prediction-using-regression-model.git
cd ecommerce-customer-value-prediction

# Create a virtual environment (optional but recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Unix / macOS
source .venv/bin/activate

# Install the package in editable mode
pip install -e .
```

### 3. Download the raw dataset

Download **online_retail_II.xlsx** from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and place it at:

```
data/raw/online_retail_II.xlsx
```

The raw file is **never modified** by any pipeline step.

---

## Usage

All commands are available via `make`. Run `make help` to see all targets.

### Run the cleaning pipeline

```bash
make clean-data
# Produces: data/interim/transactions_cleaned.csv
```

### Run the full training pipeline

```bash
make train
# Reads:    data/interim/transactions_cleaned.csv
# Produces: data/processed/customer_dataset.csv
#           models/ridge_pipeline.pkl
#           models/ridge_pipeline_metadata.json
#           reports/metrics/evaluation_metrics.json
```

### View evaluation metrics

```bash
make metrics
```

### Run all tests

```bash
make test
# 61 tests across test_data.py, test_features.py, test_models.py
# No real dataset required — all tests use synthetic in-memory data
```

### Lint and format

```bash
make lint      # check only
make format    # auto-fix
```

---

## Modelling Approach

### Problem

Predict each customer's **gross revenue over the next 4 months** (2011-09-01 to 2011-12-31), given their purchase history up to the prediction date.

### Prediction date

`2011-09-01` — separates the feature-construction window (pre-cutoff) from the revenue-measurement window (post-cutoff).

### Customer groups

| Group | Description | Target |
|---|---|---|
| A | Pre-cutoff only | `future_revenue = 0` |
| B | Post-cutoff only | Excluded (no feature history) |
| C | Both periods | `future_revenue = sum of post-cutoff line_total` |

### Features (RFM-style)

| Feature | Description |
|---|---|
| `recency_days` | Days since last pre-cutoff purchase |
| `frequency` | Unique invoices pre-cutoff |
| `monetary` | Total spend pre-cutoff (GBP) |
| `avg_basket_value` | Mean invoice spend pre-cutoff |
| `unique_products` | Distinct StockCodes purchased |
| `tenure_days` | Days between first and last purchase |
| `purchase_rate` | Invoices per day of tenure |
| `country` | Country of most recent purchase |

### Target

`log1p_future_revenue = log(future_revenue + 1)` — log-transform makes the target approximately normal and enables safe regression.

### Model

**Ridge regression** (scikit-learn `Pipeline` with `StandardScaler` + `OneHotEncoder`)

- Handles multicollinearity between RFM features.
- Fast, interpretable baseline.
- `alpha=1.0` default; tunable via `make train --alpha <value>`.

### Evaluation metrics

Both log-space (optimised) and GBP-space (business-interpretable):

```
rmse_log, mae_log, r2_log
rmse_gbp, mae_gbp, median_ae_gbp
```

---

## Data Cleaning Decisions

All cleaning logic lives in `src/.../data/clean_data.py` and is documented in `notebooks/02-data-cleaning.ipynb`.

| Step | What was removed | Why |
|---|---|---|
| Drop missing Customer ID | ~25% of rows | Cannot attribute to any real customer |
| Drop full-row duplicates | Extra copies | Data-entry/export errors |
| Keep Qty > 0 and Price > 0 | Returns, cancellations, free items | Gross revenue only |
| Drop C-prefix invoices | Remaining cancellations | Not forward purchases |
| Drop non-standard StockCodes | POST, DOT, AMAZONFEE, etc. | Not product behaviour |
| Create `line_total` after filtering | (column creation) | Guarantees `line_total > 0` |

Raw data in `data/raw/` is **never overwritten**.

---

## Import Convention

After `pip install -e .`, import from the installed package — no `sys.path` manipulation required:

```python
from ecommerce_customer_value_prediction.data.clean_data import run_cleaning_pipeline
from ecommerce_customer_value_prediction.features.build_features import build_customer_dataset
from ecommerce_customer_value_prediction.models.train import train, save_model
from ecommerce_customer_value_prediction.models.predict import predict
```

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `01-initial-data-exploration.ipynb` | EDA: schema, missing values, distributions, time coverage |
| `02-data-cleaning.ipynb` | Step-by-step cleaning with Problem → Evidence → Decision narrative |

Both notebooks import from the installed `src/` package.

---

## License

MIT
