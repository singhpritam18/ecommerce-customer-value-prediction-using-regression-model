# Linear Regression Baseline (OLS) — Reference

**Project:** E-commerce Customer Value Prediction  
**Step:** 6 of 8  
**Created:** 2026-10-10  
**Source notebook:** `notebooks/06-linear-regression-baseline.ipynb`  
**Source module:** `src/models/train.py`

---

## 1. Research Objective

Compare OLS, Ridge, Lasso and ElasticNet on an e-commerce customer-value prediction problem.  
**This step:** Establish the unregularised OLS baseline. Ridge/Lasso/ElasticNet are NOT trained here.

---

## 2. Dataset

**File:** `data/processed/customer_dataset.csv`  
**Shape:** 4,374 rows × 11 columns  
**One row = one customer** at the prediction snapshot (2011-09-01)

---

## 3. Features (X)

| Feature | Type | Role |
|---|---|---|
| `recency_days` | numeric | Days since last historical purchase |
| `frequency` | numeric | Count of distinct historical invoices |
| `monetary` | numeric | Sum of historical line_total (£) |
| `avg_basket_value` | numeric | Mean per-invoice spend (£) |
| `unique_products` | numeric | Count of distinct StockCodes |
| `tenure_days` | numeric | Span of historical activity (days) |
| `purchase_rate` | numeric | frequency / tenure_days (0 if tenure=0) |
| `country` | categorical | Most-recent historical country (one-hot encoded) |

**Customer ID is excluded** — it is an arbitrary system key with no predictive meaning.  
**future_revenue is excluded** — it is the raw target (y), not a feature.

---

## 4. Target (y)

**Training target:** `log1p_future_revenue = log(future_revenue + 1)`  
**Preserved for evaluation:** `future_revenue` (raw £)

**Why log1p:** Raw `future_revenue` has skew = 16.27. `log1p_future_revenue` has skew = 1.03.  
**Why log1p(0) = 0 is valid:** 64.7% of customers have future_revenue = 0 (genuine zeros — they did not return). `log1p(0) = log(1) = 0` maps these cleanly.

---

## 5. Train/Test Split

| Parameter | Value |
|---|---|
| Method | `sklearn.model_selection.train_test_split` |
| Test size | 20% |
| Random state | 42 (frozen for all Steps 6–8) |
| Stratification | 4-bin on future_revenue: zero / low (<£50) / medium (<£500) / high |
| Train customers | ~3,499 |
| Test customers | ~875 |

**Reproducibility contract:** All four models (OLS, Ridge, Lasso, ElasticNet) in Steps 6–8  
must use `RANDOM_STATE = 42`, `TEST_SIZE = 0.20`, same stratification → identical test sets.

**The test set is reserved for final model comparison only.** It is not evaluated during development.

---

## 6. Preprocessing

```
X (8 columns)
  → ColumnTransformer (inside sklearn Pipeline)
      numeric (7 cols)  →  StandardScaler   →  z = (x - mean_train) / std_train
      country (1 col)   →  OneHotEncoder    →  binary indicator columns per category
  → X_processed (~46 columns)
  → LinearRegression.fit(X_processed, y_log)
```

**Why StandardScaler inside Pipeline:**  
The scaler is fitted on training-fold data only during every cross-validation iteration.  
If the scaler were fitted on the full dataset before splitting, validation-fold statistics  
would influence the scaler — a form of preprocessing leakage.

**Why OneHotEncoder (not integer encoding):**  
Assigning UK=1, France=2, Germany=3 fabricates an ordering that does not exist.  
OHE creates one binary column per category: "Is this customer from country X?" (0/1).

---

## 7. OLS Objective Function

$$\min_{\beta} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \min_{\beta} \| y - X\beta \|_2^2$$

**No regularisation term.** Closed-form solution: $\hat{\beta} = (X^T X)^{-1} X^T y$

**Why squaring errors:**
1. Makes all errors positive (no cancellation)
2. Penalises large errors more than small ones (residual of 10 → cost 100; residual of 1 → cost 1)
3. Produces a unique, differentiable minimum

---

## 8. Cross-Validation Setup

| Parameter | Value |
|---|---|
| CV strategy | KFold (shuffle=True, random_state=42) |
| Folds | 5 |
| Data used | Training set only — test set not touched |
| Metrics | MAE_log, RMSE_log, R² |

Cross-validation is used for **development performance estimation only**.  
The test set is evaluated once per model after all models are trained (Step 8).

---

## 9. Development Metrics (Training CV — approximate)

Exact values are saved to `reports/metrics/ols_baseline_metrics.json` at notebook runtime.

| Metric | Approximate CV Mean | Scale |
|---|---|---|
| MAE | ~1.3 | log1p |
| RMSE | ~1.7 | log1p |
| R² | ~0.35 | dimensionless |
| MAE | ~£200 | original £ (via expm1) |
| Median AE | ~£50 | original £ (via expm1) |

**Note:** These are development metrics, not test-set metrics.

---

## 10. Back-Transformation

Model predicts `ŷ_log` on the log1p scale.  
To recover £-scale predictions: `ŷ_£ = np.expm1(ŷ_log) = exp(ŷ_log) - 1`

**Retransformation bias warning:**  
`E[expm1(ŷ_log)] ≠ E[y_£]` (Jensen's inequality).  
The mean of expm1-transformed predictions is not the mean of actual revenue.  
Log-normal correction: `E[Y] ≈ exp(μ + σ²/2)`.  
For evaluation purposes we use `expm1` directly; the bias is acknowledged as a limitation.

---

## 11. Coefficient Observations

All numeric coefficients are on the standardised (z-score) scale:  
a coefficient of +0.5 means "1 std increase in this feature → +0.5 increase in log1p_future_revenue."

**Expected signs (from EDA):**
- `recency_days` → negative (higher recency = less recently active = lower future revenue)
- `frequency` → positive (more past trips = established behaviour = more likely to return)
- `monetary` → positive (higher historical spend = higher purchasing capacity)

**Multicollinearity warning:**  
`avg_basket_value` may show a negative coefficient despite being positively correlated with revenue.  
This is a consequence of multicollinearity with `monetary` and `frequency` — the three features  
algebraically overlap (`monetary ≈ frequency × avg_basket_value`).  
Individual OLS coefficients under multicollinearity are unreliable even when predictions are reasonable.

---

## 12. Multicollinearity Findings

Computed from training set correlation matrix and VIF analysis (Section 15 of notebook).

| Feature pair | Approx |r| | Concern |
|---|---|---|
| frequency ↔ unique_products | 0.75 | High |
| monetary ↔ frequency | 0.56 | Moderate |
| monetary ↔ unique_products | 0.55 | Moderate |
| monetary ↔ avg_basket_value | 0.45+ | Moderate (algebraic) |

VIF interpretation:
- `VIF = 1`: no multicollinearity
- `VIF > 5`: moderate concern
- `VIF > 10`: severe — OLS coefficients unreliable

Features likely to have elevated VIF: `monetary`, `frequency`, `unique_products`, `avg_basket_value`.

---

## 13. Coefficient Stability Experiment

30 bootstrap resamples of the training set.  
Features with high coefficient-of-variation (std/|mean|) are **unstable**:

- `monetary` and `avg_basket_value` tend to have the highest instability
- `recency_days` is typically the most stable (lowest correlation with others)

**Key insight:** OLS predictions are relatively stable even when individual coefficients swing  
wildly across resamples. The positive/negative coefficients of correlated features cancel in $\hat{y}$.  
This instability in coefficients (not predictions) is what Ridge regularisation addresses.

---

## 14. Model Assumption Diagnostics

| Assumption | Status | Impact |
|---|---|---|
| Linearity | Partial — some non-linearity in the zero-revenue cluster | Mild prediction bias |
| Independence | Met — customers are independent observations | No concern |
| Homoscedasticity | Violated — residual variance grows with predicted value | Coefficient SEs unreliable |
| Residual normality | Violated (Shapiro-Wilk rejects) | Coefficient p-values unreliable; predictions unaffected |
| No multicollinearity | Violated — VIF elevated for RFM features | Individual coefficients unstable |

**Critical clarification:**  
"The target y must be normally distributed" is a **misconception**.  
Normality assumptions in OLS concern the **residuals**, not the raw target.  
The highly skewed `future_revenue` is fine — we transform it to `log1p_future_revenue`.  
Non-normal residuals affect inference (p-values, CIs) but NOT prediction quality (R², MAE).

---

## 15. Limitations

1. **Unstable coefficients** — multicollinearity causes high coefficient-of-variation across bootstrap resamples for `monetary`, `avg_basket_value`, `frequency`, `unique_products`
2. **No regularisation** — OLS cannot shrink or zero out redundant features
3. **Non-normal residuals** — coefficient significance tests invalid
4. **Heteroscedasticity** — prediction variance grows with predicted revenue; high-value customers are harder to predict
5. **Retransformation bias** — `expm1(predicted_log)` underestimates expected revenue due to Jensen's inequality
6. **Zero-revenue cluster** — OLS predicts slightly positive values for churned customers (actual=0); no mechanism to predict exact zeros

---

## 16. Why Regularisation Is the Next Experiment

| Model | Objective | Effect on correlated features |
|---|---|---|
| OLS | $\min \|y - X\beta\|_2^2$ | Assigns compensating ± coefficients; unstable |
| Ridge (Step 7) | $\min \|y - X\beta\|_2^2 + \alpha\|\beta\|_2^2$ | Shrinks all toward zero; stabilises correlated coefficients |
| Lasso (Step 8) | $\min \|y - X\beta\|_2^2 + \alpha\|\beta\|_1$ | Sets some to exactly zero; implicit feature selection |
| ElasticNet (Step 8) | $\min \|y - X\beta\|_2^2 + \alpha_1\|\beta\|_1 + \alpha_2\|\beta\|_2^2$ | Combines Ridge and Lasso |

**Why features must be scaled before regularisation:**  
Ridge/Lasso penalise coefficient magnitude. Without scaling, the penalty is unfairly  
large for features with small natural units (frequency: 1–194) vs large units (monetary: 0–60,000).  
StandardScaler puts all features on the same ±3σ scale so the penalty is applied fairly.

---

## 17. Code Locations

| Component | Location |
|---|---|
| Pipeline factory | `src/models/train.py` → `build_pipeline(estimator=LinearRegression())` |
| Training function | `src/models/train.py` → `train(df, estimator=LinearRegression())` |
| Evaluation function | `src/models/evaluate.py` → `evaluate(pipeline, test_df)` |
| OLS pipeline (saved) | `models/ols_pipeline.pkl` |
| OLS metadata (saved) | `models/ols_pipeline_metadata.json` |
| Development metrics | `reports/metrics/ols_baseline_metrics.json` |
| Figures | `reports/figures/ols_diagnostics.png`, `ols_numeric_coefficients.png`, `ols_correlation_matrix.png`, `ols_coefficient_stability.png` |
| Unit tests | `tests/test_models.py` |
| Notebook | `notebooks/06-linear-regression-baseline.ipynb` |

---

## 18. Split Reproducibility Contract

```python
# This exact code defines the split for ALL models in Steps 6–8
RANDOM_STATE = 42
TEST_SIZE    = 0.20
y_bins = pd.cut(y_raw, bins=[-0.01, 0.01, 50, 500, float('inf')],
                labels=['zero', 'low', 'medium', 'high'])
X_train, X_test, y_train, y_test, ... = train_test_split(
    X, y, y_raw, y_bins,
    test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_bins,
)
```

Any model that uses a different `random_state`, different `test_size`, or omits `stratify`  
cannot be fairly compared to the OLS baseline.
