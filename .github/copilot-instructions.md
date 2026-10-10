# Repository instructions

## Commands

Run commands from the repository root. Python 3.10+ is required. The installed
package name is `ecommerce_customer_value_prediction`, although its source lives
in `src/`; install it before running module entry points.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install pytest ruff
```

The [Makefile](../Makefile) defines the data and training workflows:

- `make clean-data`: produce cleaned transactions.
- `make features`: produce the customer-level dataset.
- `make train`: train the default Ridge model.
- `make train-linear`, `make train-ridge`, `make train-lasso`,
  `make train-elasticnet`: train a specific model family.
- `make compare-models`: compare all model families.

Direct training commands also work without Make:

```powershell
python -m ecommerce_customer_value_prediction.pipelines.training_pipeline --model ridge
python -m ecommerce_customer_value_prediction.pipelines.training_pipeline --experiment all
python -m ecommerce_customer_value_prediction.pipelines.training_pipeline --model ridge --tune
```

Tests use synthetic in-memory data and do not require the raw workbook:

```powershell
python -m pytest tests/ -v
python -m pytest tests/test_features.py -v
python -m pytest tests/test_models.py::test_save_load_roundtrip_predictions_identical -v
```

`make test` runs the suite. `make lint` runs the following checks, and
`make format` applies formatting/import fixes:

```powershell
ruff format --check src/ tests/
ruff check src/ tests/
```

Ruff configuration lives in [pyproject.toml](../pyproject.toml); use its
99-character line length and import-sorting rules.

## Architecture

This project predicts customer-level future gross revenue from historical
transactions in the UCI Online Retail II dataset.

- [settings.py](../src/config/settings.py) centralizes repository-relative data,
  model, and report paths. It optionally loads the root `.env`.
- [load_data.py](../src/data/load_data.py),
  [clean_data.py](../src/data/clean_data.py), and
  [validate_data.py](../src/data/validate_data.py) handle transaction ingestion,
  cleaning, and validation. Cleaning produces
  `data/interim/transactions_cleaned.csv`.
- [build_features.py](../src/features/build_features.py) aggregates transactions
  into customer-level RFM-style predictors and a future-revenue target, producing
  `data/processed/customer_dataset.csv`.
- [training_pipeline.py](../src/pipelines/training_pipeline.py) orchestrates
  feature construction, splitting, cross-validation, training, evaluation, and
  artifact persistence. It can fall back to cleaning when interim data is absent.
- [train.py](../src/models/train.py), [tune.py](../src/models/tune.py),
  [evaluate.py](../src/models/evaluate.py), and [predict.py](../src/models/predict.py)
  implement sklearn preprocessing/model pipelines, hyperparameter search,
  evaluation, and inference from serialized pipelines. Model families are linear
  regression, Ridge, Lasso, and ElasticNet.

The numbered notebooks follow the exploration-to-error-analysis workflow.
[_dev_cv_run.py](../_dev_cv_run.py) and
[_build_nb07_08_09.py](../_build_nb07_08_09.py) are development/notebook helpers,
not replacements for the package training entry point.

## Data and modeling conventions

- The authoritative completed experiment is in
  `reports/metrics/final_experiment/`: `selection.json` locks Ridge (`alpha=10.0`)
  using training CV RMSE before any test evaluation. Do not retune or reselect
  from the test metrics. The older `08-model-selection-and-comparison.ipynb`
  is unexecuted and uses a different split and selection policy; it is not the
  source of the packaged artifact.
- The executed educational notebooks are `06-linear-regression-baseline.ipynb`,
  `07-regularization-analysis.ipynb`, and `08-final-results-and-error-analysis.ipynb`.
  They read artifacts without fitting/tuning or regenerating test predictions.
  Preserve their explicit evidence gaps: only full-data Ridge coefficients and
  aggregate errors were saved, not alternative coefficients or row-level errors.
- Consumer inference uses `models/customer_value_model.joblib` with
  `models/model_metadata.json` and `models/runtime-constraints.txt`.
  `python -m ecommerce_customer_value_prediction.pipelines.final_model package`
  refits the locked configuration on the exact dataset without tuning/testing;
  prior artifacts are archived. Inference never triggers training.
- Run `predict-customer-value --input customer_features.csv --output customer_predictions.csv`
  for batch inference. Output files must not already exist. See the root README
  and `reports/model_card.md` for the finite numeric/non-empty country schema,
  inverse-transform semantics, and severe GBP-space error limitations.

- Keep `data/raw/online_retail_II.xlsx` unchanged; derived datasets belong in the
  interim/processed paths configured by `settings.py`.
- `Customer ID` is the customer-level key. The prediction cutoff is
  `2011-09-01`: historical predictors use transactions strictly before it;
  future revenue sums all available transactions on or after it. The intended
  horizon is approximately four months, but this extract ends on 2011-12-09.
- The regression target is `log1p_future_revenue`, derived from
  `future_revenue`. Neither target column, `Customer ID`, nor future-derived
  transaction fields may enter the predictor matrix. Existing feature/model
  tests enforce the leakage boundary.
- Preserve the deterministic 80/20 split stratified by active versus zero-revenue
  customers. Comparisons must use the same frozen split across model families.
  Cross-validation and tuning use only the training partition; the held-out
  test partition is reserved for final evaluation.
- Preserve preprocessing within the serialized sklearn pipeline, including
  scaling and categorical one-hot encoding, so training and inference agree.
- Evaluation reports both log-space (`rmse_log`, `mae_log`, `r2_log`) and
  GBP-space (`rmse_gbp`, `mae_gbp`, `median_ae_gbp`) metrics. Preserve these
  output names when changing evaluation or report consumers.
- Training persists model pickles/metadata under `models/` and metric/comparison
  artifacts under `reports/metrics/`. Inference appends
  `pred_log1p_future_revenue` to its output.

See the [README](../README.md) for setup and artifact details, and the
[feature/target](../references/feature-target-engineering.md),
[baseline](../references/linear-regression-baseline.md), and
[model comparison](../references/regularization-and-model-comparison.md)
references for modeling rationale.
