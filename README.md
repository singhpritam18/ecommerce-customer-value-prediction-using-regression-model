# E-Commerce Customer Value Prediction

An educational regression project that predicts customer-level future gross
purchase revenue from historical RFM-style features. It compares ordinary least
squares (OLS), Ridge, Lasso, and ElasticNet, then packages a fitted preprocessing
and regression pipeline for another developer to use **without training**.

**Readiness: EDUCATIONAL / PORTFOLIO PROTOTYPE READY. Not suitable for business
revenue forecasting or production deployment.** The selected model has modest
log-space predictive performance and severe GBP-space outlier errors.
See the [model card](reports/model_card.md).

## Dataset and prediction problem

Source: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii).
The local cleaned extract used in this experiment contains **84,955 transactions**
from **2009-12-01 through 2011-12-09**, yielding **4,374 historical customers**.
This is the observed local extract, not a claim of using every workbook row.

- Cutoff: **2011-09-01**.
- Features use transactions strictly before the cutoff.
- `future_revenue` sums cleaned `line_total` on or after the cutoff.
- Target: `log1p_future_revenue = log(1 + future_revenue)`.
- Intended horizon: approximately four months. Actual observation ends on
  **2011-12-09**, 99 calendar days after the cutoff, not December 31.
  The implementation sums all available post-cutoff rows; it does not enforce
  a four-month endpoint on an extended dataset.
- Customers with historical purchases but no future purchases have target zero
  (64.70% of customer rows). Customers with no pre-cutoff history are excluded.
- Cleaning excludes missing customer IDs, duplicate rows, nonpositive quantity
  or price, cancellations, and nonstandard stock codes. This is **gross purchase
  revenue**, not profit, net revenue, or lifetime value.

Raw data remains immutable in `data/raw/`. Local data is ignored by Git.
SHA-256 fingerprints and cleaned-data/feature-rebuild validation are recorded in
the [selection artifact](reports/metrics/final_experiment/selection.json).

## Features and expected input schema

Provide one row per customer, with these **eight required fields**:

| Field | Type | Definition |
|---|---|---|
| `recency_days` | finite real number | Whole days from last historical purchase to cutoff |
| `frequency` | finite real number | Count of distinct historical invoices |
| `monetary` | finite real number | Historical sum of `line_total`, GBP |
| `avg_basket_value` | finite real number | Mean historical invoice spend, GBP |
| `unique_products` | finite real number | Count of distinct historical StockCodes |
| `tenure_days` | finite real number | Whole days between first and last historical purchases |
| `purchase_rate` | finite real number | `frequency / tenure_days`; zero when tenure is zero |
| `country` | non-empty string | Country of most recent historical transaction |

Integer and floating-point numeric columns are accepted without changing their
input dtypes. Numeric strings, booleans, complex values, nulls, NaN, infinity,
missing columns, duplicate column names, and empty input are rejected.
Numeric CSV text is parsed into numeric columns by pandas; malformed values are
rejected rather than silently coerced.

Extra columns, such as `customer_reference`, are retained but never used as
predictors. Unknown countries are supported through the fitted encoder's
`handle_unknown="ignore"` policy (all-zero country encoding).
The identifier and target columns are not predictors.

Construct features with the same definitions and historical window as training.
The supported prototype interface is **customer-level features**, not arbitrary
raw transactions. The existing `build_rfm_features` correctly excludes
post-cutoff rows when given cleaned transactions, but assumes validated cleaning,
timestamps, complete history, and a supplied cutoff. It is not a safe general
raw-transaction service contract, so no raw-transaction inference endpoint is
exposed.

## Completed experiment and regularization findings

The source pipeline's split is frozen: **3,499 training / 875 test customers**,
80/20, `random_state=42`, stratified by zero versus positive future revenue.
Preprocessing is fitted inside every CV fold: `StandardScaler` for seven numeric
features and dense `OneHotEncoder(handle_unknown="ignore")` for country.

Five-fold shuffled, binary-stratified training CV uses the grids in
[tune.py](src/models/tune.py). Each model's configuration and the overall winner
are locked using **minimum mean CV RMSE in log1p space before test evaluation**.
All four locked pipelines are then evaluated once on the reserved test rows.

| Model | Locked regularization | CV RMSE log (mean +/- std) | Test RMSE log | Test R2 log | Test RMSE GBP | Test MAE GBP |
|---|---|---:|---:|---:|---:|---:|
| OLS | none | 1.728911 +/- 0.073463 | 1.697233 | 0.259848 | 74,634.82 | 2,737.28 |
| **Ridge** | **alpha=10.0** | **1.725261 +/- 0.070007** | **1.690147** | **0.266016** | **52,996.15** | **2,570.97** |
| Lasso | alpha=0.05 | 1.726086 +/- 0.060478 | 1.682392 | 0.272736 | 20,353.49 | 994.88 |
| ElasticNet | alpha=0.1, l1_ratio=0.75 | 1.725480 +/- 0.056377 | 1.682621 | 0.272538 | 11,112.11 | 556.38 |

Ridge's L2 shrinkage retains all features; Lasso's L1 penalty permits sparse
coefficients; ElasticNet combines shrinkage and sparsity. Regularization improves
CV log RMSE only slightly over OLS. The differences between model families are
much smaller than fold variability and do not establish a statistically
significant winner. Some alternatives perform better on the test set, especially
in GBP space; **that inspection does not change the CV-locked selection**.

The authoritative [experiment outputs](reports/metrics/final_experiment/) contain
the candidate grids, all locked hyperparameters, CV results, split/dataset
fingerprints, test metrics, and error analysis. The older unexecuted notebooks
are explanatory research material, not evidence of a completed experiment:
notebook 08 uses a different split and test-based model selection. Do not run it
to recreate or change this locked artifact.

## Final model and artifacts

**Final estimator: Ridge, alpha=10.0, fit_intercept=True**, with the remaining
sklearn constructor parameters recorded in [model_metadata.json](models/model_metadata.json).
The saved object is the complete fitted sklearn `Pipeline`, not just an estimator.

- [customer_value_model.joblib](models/customer_value_model.joblib): inference artifact.
- [model_metadata.json](models/model_metadata.json): features, categorical schema,
  target/transform, horizon, full hyperparameters, training timestamp, dataset
  fingerprint, CV/test metrics, preprocessing, version information, and checksum.
- [runtime-constraints.txt](models/runtime-constraints.txt): exact inference stack.

Post-selection policy: refit the locked selected pipeline on **all 4,374 customer
rows** after evaluation. The reported test metrics belong to the earlier
training-partition fit, **not an independent evaluation of the full-data refit**.
No new tuning, feature changes, target changes, clipping, or bias correction
occurs during packaging.

For Ridge, the final recorded test metrics are RMSE log **1.690147**, MAE log
**1.363400**, R2 log **0.266016**, RMSE GBP **52,996.15**, MAE GBP **2,570.97**,
and median absolute error GBP **3.45**.

## Installation

The source package declares Python >=3.10. This saved artifact's pinned stack
requires **Python >=3.12**; the verified runtime is **Python 3.13** with
scikit-learn **1.9.1**. Use the artifact's constraints, not the older exploratory
`requirements.txt` snapshot.

```powershell
git clone https://github.com/singhpritam18/ecommerce-customer-value-prediction-using-regression-model.git
cd ecommerce-customer-value-prediction-using-regression-model
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -c models\runtime-constraints.txt -e .
```

On macOS/Linux, activate with `source .venv/bin/activate` and use the equivalent
platform paths. Import the installed `ecommerce_customer_value_prediction`
package; no notebook path hacks or `conftest.py` aliases are needed.

Keep the repository's `models/` files together. If moving an artifact elsewhere,
pass its explicit path to `load_model` or the CLI. Only load trusted joblib/pickle
files: deserialization can execute code. The loader checks the packaged artifact's
sklearn version, checksum, and target transformation before loading it.

## Single-customer prediction

No dataset download, feature-building job, training, or tuning is required:

```python
import pandas as pd
from ecommerce_customer_value_prediction.models.predict import load_model, predict

customer = pd.DataFrame([{
    "recency_days": 30,
    "frequency": 10,
    "monetary": 1500.0,
    "avg_basket_value": 150.0,
    "unique_products": 25,
    "tenure_days": 300,
    "purchase_rate": 10 / 300,
    "country": "United Kingdom",
}])
pipeline = load_model()  # models/customer_value_model.joblib
result = predict(customer, pipeline=pipeline)
print(result[["pred_log1p_future_revenue", "pred_future_revenue"]])
```

Verified output, approximately:

```text
pred_log1p_future_revenue  pred_future_revenue
2.488367                  11.041591
```

`pred_log1p_future_revenue` is the raw regression output.
`pred_future_revenue = expm1(pred_log1p_future_revenue)` is the GBP-scale
estimate over the observed prediction horizon, **not a calibrated expected
revenue or an amount guaranteed to be earned**. Input fields, index, and row order
are preserved; these two float columns are appended.

The inverse transform is intentionally unchanged: it can produce negative values
for negative log predictions and very large values for large log predictions.
Non-finite results cause an explicit error. No clipping or smearing correction is
silently applied.

Run the working example with `python examples\predict_one.py`.

## Multiple customers and batch prediction

Pass a multirow DataFrame to the same `predict` function. The two synthetic
customer records in [customer_features.csv](examples/customer_features.csv) are
ready to use:

```powershell
predict-customer-value --input examples\customer_features.csv --output customer_predictions.csv
```

To use a copied artifact:

```powershell
predict-customer-value --input customer_features.csv --output customer_predictions.csv --model models\customer_value_model.joblib
```

These commands perform inference only and fail explicitly for invalid input.
Existing output files are not overwritten; choose a new filename. The
[verified example output](examples/customer_predictions.csv) contains GBP
predictions of approximately **11.04** and **5.99** for the two synthetic rows.

## Reproducing experiments and training

**Skip this section to use the saved model.** Download the raw workbook to
`data/raw/online_retail_II.xlsx` only if preparing training data. `make clean-data`
and `make features` provide the existing cleaning/feature workflows. The raw file
is never modified.

Exact reproduction requires the **same local processed snapshot and row order**
identified by SHA-256 in the selection metadata, plus the pinned runtime.
The private local data snapshot is not distributed in Git; a fresh/full UCI
download may produce a different dataset. This limits exact experiment
reproduction but does not prevent standalone inference from the shipped model.

To execute a deliberately new experiment:

```powershell
python -m ecommerce_customer_value_prediction.pipelines.final_model experiment --experiment-dir reports\metrics\reproduced_experiment
```

Each experiment directory is single-use. All candidate configurations and the
winner are persisted before holdout evaluation. Do not repeat test inspection to
select a new model. Use a new temporal validation design for new research.
The prior `reports/metrics/cv_results.csv` is preserved as training-only evidence.

To refit/package the **already locked** experiment without retuning or evaluating
the test set again:

```powershell
python -m ecommerce_customer_value_prediction.pipelines.final_model package
```

This requires the exact locked dataset fingerprint and sklearn version.
Existing model/metadata files are preserved under timestamped
`models/archive/` directories before replacement; the archive location is logged.
Inference never runs this command automatically. Legacy `make train` and
`make compare-models` are exploratory workflows, not the final packaging command.

## Running tests and lint checks

Verified result: **238 tests passed**. Changed model/packaging/inference code
passes Ruff formatting and lint checks; the Python package wheel builds
successfully.

```powershell
python -m pip install pytest ruff
python -m pytest tests/ -q
python -m pytest tests/test_inference.py -q
python -m pytest tests/test_inference.py::test_save_terminate_reload_matches_and_never_fits -q
ruff format --check src/ tests/
ruff check src/ tests/
```

Tests cover synthetic fit/save/terminated-process reload equivalence at
`rtol=atol=1e-12`, required schema, extra fields, unknown countries, multiple rows,
invalid types, non-finite values, archive preservation, and locking before test
evaluation. A new-user test loads the shipped model and examples in a fresh
Python process outside the repository working directory, forbids `fit`, checks
the output schema, and runs the installed batch CLI. Raw data is not required.

## Limitations and production work

- A single historical customer holdout is not temporal backtesting and does not
  measure performance across future periods, economic conditions, or new cohorts.
- The target is zero-inflated and long-tailed; the overall median error masks
  severe errors for revenue-positive/high-value customers.
- Ridge's held-out mean prediction is **2,541.54 GBP** versus an actual mean of
  **57.14 GBP**. Its largest absolute error is approximately **1.20 million GBP**.
  There are **50 negative predictions** among 875 test customers. These are
  disclosed prototype failures, not business-usable estimates.
- Simple `expm1` is not calibrated to conditional mean revenue. The observed
  result has severe overprediction outliers; a generic upward correction is not
  justified by these diagnostics.
- Unknown countries are handled technically but not validated geographically.
  New customers without historical purchases are outside the modeled cohort.
- The observed horizon is shorter than a complete four months. Currency,
  cleaning policy, cutoff, and historical coverage must match the input contract.
- Exact data-snapshot reproduction is limited by undistributed local data.

Real deployment needs rolling **temporal backtesting**, revenue-space calibration
and error acceptance criteria, validated data contracts/history windows,
monitoring of input quality and predictive performance, drift detection,
retraining triggers with fresh independent validation, versioned releases and
rollback, and operational batch/service deployment with access control and
reliability testing. Inference working alone is not deployment readiness.

## Project structure

- `src/data/`, `src/features/`: cleaning, validation, customer aggregation.
- `src/models/`: estimator/preprocessing factories, CV, metrics, validated inference.
- `src/pipelines/`: legacy workflows and the locked experiment/packaging entry point.
- `models/`, `examples/`: consumer artifacts, runtime constraints, runnable examples.
- `reports/metrics/final_experiment/`, `reports/model_card.md`: experiment evidence
  and intended-use/limitation summary.
- `notebooks/`, `references/`, `docs/`: research narratives and supporting docs.
- `tests/`: data, features, modeling, tuning, packaging, and fresh-process inference.

## License

MIT. Dataset use remains subject to its source terms.
