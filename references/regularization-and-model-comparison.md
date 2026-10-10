# Step 8 — Regularisation and Model Comparison

**Status:** Complete  
**Date:** 2026-10-10  
**Author:** Learned from Step 6 (OLS baseline) → Step 8 (Lasso / ElasticNet)

---

## A. What was done

Added Lasso and ElasticNet as first-class estimators alongside the existing OLS and Ridge models.
All four models now:
- Share a common `create_model(name, **hyperparams)` factory
- Use the same `build_pipeline()` preprocessing (StandardScaler + OneHotEncoder)
- Are trained by the same `train()` function
- Can be saved/loaded via `save_model` / `load_model`
- Are accessible via the unified CLI (`--model linear|ridge|lasso|elasticnet`)
- Are compared by `--experiment all` on a single frozen train/test split

---

## B. Supported models

| CLI name    | sklearn class     | Penalty        | Produces zeros | Solver         |
|-------------|-------------------|----------------|----------------|----------------|
| `linear`    | LinearRegression  | none           | No             | closed form    |
| `ridge`     | Ridge             | L2 (sum β²)    | No             | closed form    |
| `lasso`     | Lasso             | L1 (sum |β|)   | Yes            | coordinate descent |
| `elasticnet`| ElasticNet        | L1 + L2        | Yes (partial)  | coordinate descent |

---

## C. Model factory

```python
from ecommerce_customer_value_prediction.models.train import create_model

# Unregularised OLS
est = create_model("linear")

# Ridge with custom alpha
est = create_model("ridge", alpha=10.0)

# Lasso — default max_iter=10_000 applied automatically
est = create_model("lasso", alpha=0.01)

# ElasticNet — l1_ratio controls L1/L2 mix (0=Ridge, 1=Lasso)
est = create_model("elasticnet", alpha=0.1, l1_ratio=0.5)
```

---

## D. CLI usage

```bash
# Train OLS baseline
make train-linear

# Train Ridge
make train-ridge                      # alpha=1.0
python -m ...training_pipeline --model ridge --alpha 10.0

# Train Lasso
make train-lasso                      # alpha=0.01
python -m ...training_pipeline --model lasso --alpha 0.005

# Train ElasticNet
make train-elasticnet                 # alpha=0.1, l1_ratio=0.5
python -m ...training_pipeline --model elasticnet --alpha 0.1 --l1-ratio 0.7

# Train all four models and compare
make compare-models
python -m ...training_pipeline --experiment all

# Show comparison table after compare-models
make report
```

---

## E. Notebook sequence

| Notebook | Title | Purpose |
|---|---|---|
| `07-regularization-experiments.ipynb` | Regularisation Experiments | Explore Ridge/Lasso/ElasticNet behaviour on training CV only. No test-set use. |
| `08-model-selection-and-comparison.ipynb` | Model Selection | GridSearchCV tuning + final test-set evaluation (first and only use of test set). |
| `09-final-error-analysis.ipynb` | Final Error Analysis | Residual diagnostics, segment analysis, limitations. |

---

## F. Key design decisions

### F1. Shared frozen split
All four models in `--experiment all` use the **same** `train_test_split` call with
`random_state=42`, `test_size=0.20`, stratified on a binary `has_revenue` flag.
This ensures a fair head-to-head comparison: the same customers appear in every
model's training and test set.

### F2. Lasso max_iter default
`create_model("lasso")` automatically sets `max_iter=10_000`.
This is because Lasso uses coordinate descent. On standardised data with collinear
features (monetary / frequency / unique_products are correlated), the default
`max_iter=1000` can fail to converge. Setting 10,000 eliminates ConvergenceWarning.

### F3. ElasticNet parameterisation
sklearn's `ElasticNet(alpha, l1_ratio)`:
- `alpha` = overall regularisation strength
- `l1_ratio` = fraction of L1 in the penalty (0 = pure Ridge, 1 = pure Lasso)
The GridSearch in Notebook 08 sweeps `alpha ∈ [0.001, 0.5]` and
`l1_ratio ∈ [0.1, 0.3, 0.5, 0.7, 0.9]`.

### F4. Refit on full dataset
After evaluating on the held-out test set, each model is refit on the **full**
`customer_dataset.csv` before being saved to `models/`. This is correct because:
1. The test-set evaluation already tells us generalisation performance.
2. More data always produces a better final model than training on 80%.

### F5. log1p retransformation
The target is `log1p_future_revenue`. Predictions are returned in both spaces:
- `pred_log1p_future_revenue` — model output (log-space)
- `pred_future_revenue = expm1(pred_log1p_future_revenue)` — £ estimate

For fleet-level revenue forecasting, a **smearing correction factor**
`exp(σ²/2)` should be applied to `pred_future_revenue` to correct for
Jensen's inequality retransformation bias.

---

## G. Ridge: L2 regularisation summary

**Objective:** minimise Σ(y - ŷ)² + α Σβ²

**Effect on correlated RFM features:**
Monetary, frequency, and unique_products are positively correlated (~0.6–0.8).
OLS assigns arbitrary large ± splits across correlated features, producing
unstable coefficients that vary wildly across bootstrap resamples.
Ridge shrinks all coefficients proportionally, distributing credit evenly
across correlated features and stabilising the coefficient estimates.

**What Ridge cannot do:** set any coefficient to exactly zero.
All features remain in the model regardless of alpha.

---

## H. Lasso: L1 regularisation summary

**Objective:** minimise Σ(y - ŷ)² + α Σ|βⱼ|

**Why sparsity occurs:**
The L1 penalty is not differentiable at βⱼ = 0. The sub-gradient condition
at zero is satisfied when the data signal for a feature is smaller than α.
The optimiser sets the coefficient to exactly zero.

**Geometric intuition:** The Lasso constraint region is a diamond (L1 ball).
In high-dimensional problems, the optimal solution tends to land on a corner
of the diamond, where one or more coordinates are zero.

**Correlated features:** Lasso arbitrarily zeros one of a correlated pair.
The zeroed coefficient does NOT mean the feature is unimportant — it means
its signal was already captured by its correlate.

---

## I. ElasticNet summary

**Objective:** minimise Σ(y - ŷ)² + α[(1-ρ)/2 Σβ² + ρ Σ|βⱼ|]

ElasticNet blends L1 (sparsity) and L2 (stability).
With correlated RFM features, ElasticNet tends to produce a sparser model
than Ridge but keeps more features than Lasso, avoiding the arbitrary
one-of-a-correlated-pair zero.

---

## J. Files changed

| File | Change |
|---|---|
| `src/models/train.py` | Added `create_model()` factory, Lasso/ElasticNet imports, per-model path constants, `SUPPORTED_MODELS`, hyperparameter recording in metadata |
| `src/pipelines/training_pipeline.py` | Added `--model` CLI flag, `--experiment all`, `run_experiment_all()`, `_cv_metrics()`, `_make_stratified_split()` |
| `tests/test_models.py` | Full rewrite: `create_model` tests, parametrised tests across all 4 models, save/load roundtrip, evaluate, predict |
| `Makefile` | Added `train-linear`, `train-ridge`, `train-lasso`, `train-elasticnet`, `compare-models`, `report` targets |
| `notebooks/07-regularization-experiments.ipynb` | New: Ridge/Lasso/ElasticNet theory, alpha sweeps, coefficient paths, stability analysis |
| `notebooks/08-model-selection-and-comparison.ipynb` | New: GridSearchCV tuning, final test-set comparison, research question answer, final model save |
| `notebooks/09-final-error-analysis.ipynb` | New: Residual diagnostics, segment analysis, retransformation bias, limitations |

**Files unchanged (correct as written):**
- `src/features/build_features.py` — feature engineering unchanged
- `src/models/evaluate.py` — evaluation unchanged
- `src/models/predict.py` — prediction unchanged
- `src/config/settings.py` — configuration unchanged
- `tests/test_features.py` — feature tests unchanged (43 tests)
- `tests/test_data.py` — data tests unchanged (33 tests)

---

## K. Test coverage

```
138 passed in 4.61s

Breakdown:
  tests/test_data.py      33 tests  (data cleaning + validation)
  tests/test_features.py  43 tests  (RFM features + target + leakage)
  tests/test_models.py    62 tests  (create_model, build_pipeline, train,
                                     save/load, evaluate, predict —
                                     all four models parametrised)
```

---

## L. Next steps (Step 9 onwards)

1. Run Notebook 07 to explore alpha grids and coefficient paths
2. Run Notebook 08 to execute GridSearchCV and final test-set evaluation
3. Run Notebook 09 to diagnose residuals and document limitations
4. Optionally: hyperparameter tuning with RidgeCV / LassoCV for more
   efficient alpha selection without manual grid search
5. Consider: two-stage model (classify churn → predict revenue for non-churned)
   to address the zero-inflation in `future_revenue`
