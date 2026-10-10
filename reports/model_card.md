# Customer value model card

## Intended use

**EDUCATIONAL / PORTFOLIO PROTOTYPE READY.** Demonstrates regression comparison,
training-only CV selection, complete-pipeline persistence, and schema-validated
customer inference. Not approved for production or financial/business decisions.

## Model and target

- Ridge regression, `alpha=10.0`, `fit_intercept=True`; full constructor parameters
  are in [model metadata](../models/model_metadata.json).
- Pipeline: StandardScaler on numeric features, dense OneHotEncoder with
  `handle_unknown="ignore"` on country, then Ridge.
- Target: `log1p_future_revenue = log(1 + future gross purchase revenue in GBP)`.
- Output: raw log1p prediction and `expm1` GBP prediction; no clipping, smearing,
  or mean-revenue calibration.
- Cutoff: 2011-09-01. Intended horizon approximately four months; the local
  observation ends 2011-12-09 (99 days). Target construction sums all available
  post-cutoff rows, not an explicitly bounded future window.

## Inputs and training context

Seven finite real numeric fields: `recency_days`, `frequency`, `monetary`,
`avg_basket_value`, `unique_products`, `tenure_days`, `purchase_rate`.
Categorical field: non-empty string `country`, from the most recent historical
purchase. Feature definitions and validation rules are in the
[README](../README.md#features-and-expected-input-schema).

UCI Online Retail II local cleaned extract: 84,955 transactions,
2009-12-01 through 2011-12-09; 4,374 customers with pre-cutoff history.
64.70% have zero future revenue. Cancellations, nonpositive quantities/prices,
nonstandard stock codes, duplicate rows, and missing customer IDs are excluded.
This is gross purchase revenue, not net revenue/profit or lifetime value.
The local extract's completeness relative to the full workbook is not established.

The deterministic 80/20 binary-stratified split has 3,499 training and 875 test
customers (`random_state=42`). Five-fold training-only stratified CV selects the
model and hyperparameters before the held-out evaluation. The selected pipeline
is subsequently refit on all 4,374 customer rows; no further selection is made.

## Evaluation results

Training CV RMSE log: **1.725261 +/- 0.070007**.
CV MAE log: **1.399844 +/- 0.052029**.
CV R2 log: **0.251098 +/- 0.051559**.

| Held-out metric | Value |
|---|---:|
| Customers | 875 |
| RMSE log | 1.690147 |
| MAE log | 1.363400 |
| R2 log | 0.266016 |
| RMSE GBP | 52,996.15 |
| MAE GBP | 2,570.97 |
| Median absolute error GBP | 3.45 |

These are metrics of the training-partition fit, not an independent evaluation
of the full-data-refitted artifact. Ridge narrowly wins training CV;
test performance does not drive reselection. Differences between CV model
means are far smaller than fold variability.

## Known limitations and error analysis

- Single historical customer split; no rolling temporal or external validation.
- Zero inflation and heavy tails make median error misleading for active customers.
- Mean predicted versus actual held-out revenue: **2,541.54 versus 57.14 GBP**.
- MAE for positive-revenue customers (309 rows): **7,275.19 GBP**.
  For customers above 1,000 GBP actual revenue (9 rows): **247,742.26 GBP**.
- Largest absolute error: approximately **1,195,100.69 GBP**.
- **50/875 negative predictions**; unchanged inverse transform is intentional.
- Unknown country handling prevents an encoder error, not distributional risk.
- Inconsistent history windows or a longer post-cutoff dataset change the task.
- No uncertainty intervals, calibrated GBP conditional mean, drift monitoring,
  or new-customer capability.
- Dataset SHA-256 and environment are recorded, but the exact local data snapshot
  is not distributed. Exact experiment reproduction requires that snapshot.

See [error_analysis.json](metrics/final_experiment/error_analysis.json) and
[test_metrics.json](metrics/final_experiment/test_metrics.json) for source results.

## Inappropriate uses

Revenue budgeting, financial commitments, customer eligibility/pricing decisions,
automated marketing expenditure, or any claim of production forecasting accuracy.
Do not interpret coefficients as causal effects or infer that unseen cohorts
are validated because the model accepts their features.

## Retraining and deployment considerations

Future work requires temporal backtesting with complete fixed horizons,
revenue-space calibration and explicit error thresholds, data contracts for
cleaning/currency/history/cutoffs, input-quality and performance monitoring,
drift detection, retraining triggers, fresh validation, versioned releases,
rollback, and operational service/batch deployment with access control.
Repeated inspection of this test set is not fresh validation.

## Consumer verification

Verified: **238 tests passed**, changed Python code passes Ruff checks, and the
package wheel builds. The installed batch CLI reproduces the saved example
output exactly.

The repository supplies the full pipeline, metadata/checksum, pinned runtime
constraints, single-customer Python example, two-row CSV example, and batch CLI.
Tests exercise cross-process serialization equality at `rtol=atol=1e-12`,
schema failures, unknown categories, metadata integrity, archive preservation,
and fresh-process inference with fitting forbidden.
