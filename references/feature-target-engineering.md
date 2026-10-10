# Feature Engineering & Target Construction — Reference

**Project:** E-commerce Customer Value Prediction  
**Step:** 5 of 6  
**Created:** 2026-10-10  
**Source notebook:** `notebooks/05-feature-target-engineering.ipynb`  
**Source module:** `src/features/build_features.py`

---

## 1. Prediction Date (Cutoff)

```
PREDICTION_DATE = 2011-09-01
```

Configured in `src/features/build_features.py` as `PREDICTION_DATE`.

---

## 2. Historical Window (Source of X)

| Property | Value |
|---|---|
| Start date | 2009-12-01 |
| End date | 2011-08-31 |
| Length | 638 days (~21.3 months) |
| Transaction rows | 73,306 |
| Unique customers | 4,374 |

All feature calculations use **only** rows where `InvoiceDate < 2011-09-01`.

---

## 3. Future Target Window (Source of y)

| Property | Value |
|---|---|
| Start date | 2011-09-01 |
| End date | 2011-12-09 |
| Length | **99 days (~3.3 months)** |
| Transaction rows | 11,649 |
| Unique customers | 2,016 |

All target calculations use **only** rows where `InvoiceDate >= 2011-09-01`.

---

## 4. Unit of Analysis

**One row = one customer** in the modelling dataset.

The cleaned transaction dataset (84,955 rows) is aggregated to 4,374 rows.

---

## 5. Feature Definitions

### 5.1 recency_days

**Business meaning:** Days elapsed since the customer's most recent historical purchase.  
**Lower = more recently active = more likely to return.**

**Formula:**
```
recency_days = prediction_date − max(InvoiceDate)
               where InvoiceDate < prediction_date
```

**Pandas:**
```python
recency = (PREDICTION_DATE - historical_df.groupby('Customer ID')['InvoiceDate'].max()).dt.days
```

---

### 5.2 frequency

**Business meaning:** Number of distinct shopping trips a customer made historically.  
**NOT the number of transaction rows** — one trip = one Invoice number regardless of how many products.

**Formula:**
```
frequency = COUNT(DISTINCT Invoice)   [historical window only]
```

**Pandas:**
```python
frequency = historical_df.groupby('Customer ID')['Invoice'].nunique()
```

---

### 5.3 monetary

**Business meaning:** Total historical spend.  
**Not the same as `future_revenue`** — same variable type, different time window.

**Formula:**
```
monetary = SUM(line_total)   [historical window only]
```

**Pandas:**
```python
monetary = historical_df.groupby('Customer ID')['line_total'].sum()
```

---

### 5.4 avg_basket_value

**Business meaning:** Mean spend per shopping trip. Distinguishes habitual small-basket buyers from occasional large-order buyers even when total monetary spend is identical.

**Formula:**
```
avg_basket_value = MEAN( SUM(line_total per invoice) )   [historical window only]
```

Two-step aggregation: first sum within each (Customer, Invoice) pair, then average across invoices.

**Pandas:**
```python
invoice_spend = historical_df.groupby(['Customer ID', 'Invoice'])['line_total'].sum()
avg_basket_value = invoice_spend.groupby('Customer ID').mean()
```

**Note:** For a single-invoice customer, `avg_basket_value == monetary` (exactly).

---

### 5.5 unique_products

**Business meaning:** Number of distinct product types purchased historically. Measures breadth of engagement and purchase diversity.

**Formula:**
```
unique_products = COUNT(DISTINCT StockCode)   [historical window only]
```

**Pandas:**
```python
unique_products = historical_df.groupby('Customer ID')['StockCode'].nunique()
```

---

### 5.6 tenure_days

**Business meaning:** Span of customer's historical activity — gap between first and last purchase.

**Formula:**
```
tenure_days = max(InvoiceDate) − min(InvoiceDate)   [historical window only]
```

**Pandas:**
```python
tenure = (
    historical_df.groupby('Customer ID')['InvoiceDate'].max()
    - historical_df.groupby('Customer ID')['InvoiceDate'].min()
).dt.days
```

---

### 5.7 purchase_rate

**Business meaning:** Purchasing cadence measured in orders per day. See Section 8 for the edge-case decision.

**Formula:**
```
purchase_rate = frequency / tenure_days   if tenure_days > 0
purchase_rate = 0.0                       if tenure_days == 0
```

**Pandas:**
```python
purchase_rate = np.where(tenure > 0, frequency / tenure, 0.0)
```

---

### 5.8 country

**Business meaning:** Customer's most recently observed country of transaction. Categorical — not encoded yet.

**Formula:**
```
country = Country of the transaction with max(InvoiceDate)   [historical window only]
```

**Pandas:**
```python
country = (
    historical_df
    .sort_values('InvoiceDate')        # ← sort required before .last()
    .groupby('Customer ID')['Country']
    .last()
)
```

---

## 6. Mathematical Formulas Summary

| Feature | Formula |
|---|---|
| `recency_days` | `cutoff − max(InvoiceDate_hist)` in days |
| `frequency` | `|{Invoice : InvoiceDate < cutoff}|` (set cardinality) |
| `monetary` | `Σ line_total` for historical rows |
| `avg_basket_value` | `(1/frequency) × Σ_invoice (Σ_items line_total)` |
| `unique_products` | `|{StockCode : InvoiceDate < cutoff}|` (set cardinality) |
| `tenure_days` | `max(InvoiceDate_hist) − min(InvoiceDate_hist)` in days |
| `purchase_rate` | `frequency / tenure_days` if `tenure_days > 0` else `0` |
| `country` | `Country` of row with `argmax(InvoiceDate_hist)` |

---

## 7. Feature Business Meanings

| Feature | Predicts future value because... |
|---|---|
| `recency_days` | Recent purchasers are still engaged; confirmed by Step 4 EDA |
| `frequency` | Repeat buyers have an established relationship; strongest predictor |
| `monetary` | Historical spend signals purchasing capacity and intent |
| `avg_basket_value` | Distinguishes habitual from occasional buyers at same total spend |
| `unique_products` | Breadth of engagement suggests deeper product affinity |
| `tenure_days` | Long-standing customers have shown sustained interest |
| `purchase_rate` | Cadence predicts whether a customer is in an active cycle |
| `country` | Geographic segment captures different market behaviours |

---

## 8. Edge-Case Decisions

### tenure_days == 0 (37.9% of customers)

**Cause:** Customer made all purchases on a single calendar day (one-time buyers or same-day orders).

**Problem:** `purchase_rate = frequency / 0` → division by zero → `NaN` or `Inf`.

**Options considered:**

| Option | Formula | Issue |
|---|---|---|
| A | `frequency / max(tenure, 1)` | Customer with `freq=5, tenure=0` gets `rate=5.0` — appears more active than `freq=30, tenure=300 → rate=0.1`. Inflates rate for one-day shoppers. |
| B | `0.0` when `tenure=0` | Neutral value. Frequency already captures order count. Cannot measure cadence from one day. |

**Decision: Option B** (`purchase_rate = 0.0` when `tenure_days == 0`).

### Multi-country customers (6 customers)

**Decision: most-recent-country** (Country of chronologically last historical transaction).

Rationale: RFM framework prioritises recency signals consistently. Only 6 customers affected — choice has negligible model impact.

**Critical implementation note:** `historical_df` must be sorted by `InvoiceDate` before calling `.groupby().last()`. Without sorting, `.last()` returns the last row in DataFrame insertion order, which is not guaranteed to be chronological.

### Group B customers (future-only, 472 customers)

**Decision: exclude from modelling population.**

These customers appear only in the future window. They have a target value `y` but no historical feature vector `X`. No prediction can be made for them at the time of deployment.

### future_revenue == 0 (64.7% of customers, 2,830 customers)

**Decision: retain as genuine zero.**

These customers were active historically and were observed for 99 days after the cutoff. Zero is the true answer to "how much will they spend?". It represents lapsed customers, seasonal buyers, or churned customers. Removing or imputing these zeros would bias the model.

---

## 9. Purchase Rate Definition

**Chosen formula:**
```
purchase_rate = frequency / tenure_days   if tenure_days > 0
              = 0.0                       if tenure_days == 0
```

**Rejected formula:** `frequency / max(tenure_days, 1)`  
**Reason rejected:** Assigns artificially high rates to same-day customers (e.g. `rate=5.0` for a customer with 5 orders on one day) inflating their apparent activity beyond what can be measured.

**Documented in:** `build_rfm_features()` docstring and `tests/test_features.py::test_tenure_zero_customers_have_zero_purchase_rate`.

---

## 10. Country Definition

**Rule:** Customer-level country = Country of the transaction with the **most recent InvoiceDate** in the historical window.

**Alternative considered:** Most-frequent country per customer.  
**Why not chosen:** Only 6 customers have multi-country history; most-recent is more consistent with the recency principle used everywhere else.

**Documented in:** `build_rfm_features()` with explicit `pre.sort_values('InvoiceDate')` before aggregation.

---

## 11. Target Definition

```
future_revenue          = SUM(line_total)   for InvoiceDate >= 2011-09-01
log1p_future_revenue    = log(future_revenue + 1)
```

**Primary regression target:** `log1p_future_revenue`  
**Preserved for evaluation:** `future_revenue` (raw £)

---

## 12. Target Transformation Decision

| Metric | `future_revenue` (raw) | `log1p_future_revenue` |
|---|---|---|
| Skew | 16.27 (extreme) | 1.03 (near-symmetric) |
| Zero handling | Valid zero | `log1p(0) = 0` (safe) |
| Error unit | Absolute £ | Proportional log-£ |
| OLS residual normality | Violated | Better satisfied |
| Outlier influence on loss | Dominated by large spenders | Evenly spread |

**Decision: use `log1p_future_revenue` as the primary training target.**

**Why `log1p` not `log`:** `log(0)` is undefined (`-∞`). `log1p(0) = log(1) = 0`. Safe for the 64.7% of customers with zero future revenue.

**Why raw `future_revenue` is preserved:** Final evaluation metrics (RMSE, MAE) must be reported in original £ units. The raw column enables `expm1(prediction)` back-transformation for business-interpretable results.

**Jensen's inequality warning:** `E[expm1(ŷ_log)] ≠ E[y]`. When back-transforming to produce mean revenue predictions, apply the log-normal correction: `E[Y] ≈ expm1(μ + σ²/2)`.

---

## 13. Leakage Rules

| Rule | Enforcement |
|---|---|
| All X features use only `InvoiceDate < 2011-09-01` | Enforced by `pre = df[df['InvoiceDate'] < prediction_date]` in `build_rfm_features` |
| Target uses only `InvoiceDate >= 2011-09-01` | Enforced by `post = df[df['InvoiceDate'] >= prediction_date]` in `build_target` |
| `future_revenue` not in feature DataFrame | Verified by `test_leakage_future_revenue_not_in_features` |
| Future transaction count not in features | Verified by `test_leakage_no_future_transaction_count_in_features` |
| Future monetary not in features | Verified by `test_leakage_no_future_monetary_in_features` |
| Adding post-cutoff rows does not change X | Verified by `test_leakage_x_uses_only_historical_transactions` |
| Modifying pre-cutoff rows does not change y | Verified by `test_leakage_y_uses_only_future_transactions` |
| Customer ID not a model predictor | Verified by `test_leakage_customer_id_not_in_feature_columns` |
| `returned_in_future` never created as feature | Feature does not exist in dataset |

---

## 14. Final Modelling Dataset Schema

**File:** `data/processed/customer_dataset.csv`  
**Shape:** 4,374 rows × 11 columns  
**Production function:** `build_customer_dataset(df, prediction_date)`

| Column | Type | Role | Source window |
|---|---|---|---|
| `Customer ID` | Int64 | Join key (NOT a predictor) | — |
| `recency_days` | float | Feature X | Historical |
| `frequency` | int | Feature X | Historical |
| `monetary` | float | Feature X | Historical |
| `avg_basket_value` | float | Feature X | Historical |
| `unique_products` | int | Feature X | Historical |
| `tenure_days` | int | Feature X | Historical |
| `purchase_rate` | float | Feature X | Historical |
| `country` | str | Feature X (categorical, not yet encoded) | Historical |
| `future_revenue` | float | Target y (raw £, preserved for evaluation) | Future |
| `log1p_future_revenue` | float | Target y (primary training target) | Future |

**Dataset statistics:**

| Statistic | Value |
|---|---|
| Total customers | 4,374 |
| Group A (future_revenue = 0) | 2,830 (64.7%) |
| Group C (future_revenue > 0) | 1,544 (35.3%) |
| future_revenue skew | 16.27 |
| log1p_future_revenue skew | 1.03 |
| Forecast horizon | 99 days |

---

## 15. Code Locations

| Component | Location |
|---|---|
| Feature construction | `src/features/build_features.py` → `build_rfm_features()` |
| Target construction | `src/features/build_features.py` → `build_target()` |
| Dataset assembly | `src/features/build_features.py` → `build_customer_dataset()` |
| Save | `src/features/build_features.py` → `save_customer_dataset()` |
| Unit tests | `tests/test_features.py` (51 tests) |
| Educational notebook | `notebooks/05-feature-target-engineering.ipynb` |
| Output dataset | `data/processed/customer_dataset.csv` |
| Prediction date config | `src/features/build_features.py` → `PREDICTION_DATE` |

---

## 16. Manual Verification Example — Customer 12374

Use this to verify any implementation produces correct values.

**Historical transactions (pre-2011-09-01):**

| Invoice | InvoiceDate | StockCode | Qty | Price | line_total |
|---|---|---|---|---|---|
| 506579 | 2010-04-30 | 85099B | 40 | 1.95 | 78.00 |
| 506579 | 2010-04-30 | 84997D | 18 | 3.75 | 67.50 |
| 506579 | 2010-04-30 | 84997C | 18 | 3.75 | 67.50 |
| 506579 | 2010-04-30 | 84997B | 18 | 3.75 | 67.50 |
| 526941 | 2010-10-14 | 85099C | 20 | 1.95 | 39.00 |
| 526941 | 2010-10-14 | 85099B | 20 | 1.95 | 39.00 |
| 526941 | 2010-10-14 | 85232B | 9 | 4.95 | 44.55 |

**Expected features:**

| Feature | Calculation | Value |
|---|---|---|
| `recency_days` | 2011-09-01 − 2010-10-14 | 321 |
| `frequency` | {506579, 526941} = 2 distinct invoices | 2 |
| `monetary` | 78+67.5+67.5+67.5+39+39+44.55 | £403.05 |
| `avg_basket_value` | mean([280.50, 122.55]) | £201.525 |
| `unique_products` | {85099B, 84997D, 84997C, 84997B, 85099C, 85232B} | 6 |
| `tenure_days` | 2010-10-14 − 2010-04-30 | 166 |
| `purchase_rate` | 2 / 166 | 0.012048 |
| `country` | Most recent (526941 on 2010-10-14) | Austria |

**Future transaction (post-2011-09-01):**

| Invoice | InvoiceDate | StockCode | Qty | Price | line_total |
|---|---|---|---|---|---|
| 576345 | 2011-11-14 | 46775D | 2 | 16.95 | 33.90 |

**Expected targets:**

| Target | Calculation | Value |
|---|---|---|
| `future_revenue` | 2 × 16.95 | £33.90 |
| `log1p_future_revenue` | log(33.90 + 1) = log(34.90) | 3.5525 |
