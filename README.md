# E-Commerce Customer Value Prediction

Predict future 4-month gross revenue for each customer using RFM-style features and Ridge regression on the [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) dataset (UCI ML Repository).

---

## Project Structure

```
ecommerce-customer-value-prediction/
├── data/
│   ├── raw/                     ← IMMUTABLE source file (online_retail_II.xlsx)
│   ├── interim/                 ← Cleaned transactions (transactions_cleaned.csv)
│   └── processed/               ← Customer-level modelling dataset (customer_dataset.csv)
├── models/                      ← Fitted pipeline (ridge_pipeline.pkl) + metadata JSON
├── notebooks/
│   ├── 01-initial-data-exploration.ipynb
│   └── 02-data-cleaning.ipynb
├── reports/
│   ├── figures/
│   └── metrics/                 ← Evaluation metrics JSON
├── src/
│   └── ecommerce_customer_value_prediction/
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py      ← All project paths (PROJ_ROOT, RAW_DATA_DIR, …)
│       ├── data/
│       │   ├── __init__.py
│       │   ├── load_data.py     ← Load raw Excel workbook
│       │   ├── clean_data.py    ← Transaction-level cleaning + pipeline
│       │   └── validate_data.py ← Diagnostic reporters + hard constraints
│       ├── features/
│       │   ├── __init__.py
│       │   └── build_features.py ← RFM features + future_revenue target
│       ├── models/
│       │   ├── __init__.py
│       │   ├── train.py         ← Ridge pipeline: build, fit, save, load
│       │   ├── evaluate.py      ← Evaluation metrics (log-space + GBP)
│       │   └── predict.py       ← Inference: load model + generate predictions
│       └── pipelines/
│           ├── __init__.py
│           └── training_pipeline.py ← End-to-end orchestration
├── tests/
│   ├── test_data.py             ← 27 tests for cleaning + validation
│   ├── test_features.py         ← 16 tests for RFM feature engineering
│   └── test_models.py           ← 18 tests for train/eval/predict/save-load
├── pyproject.toml
├── Makefile
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
