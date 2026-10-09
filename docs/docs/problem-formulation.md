# Problem Formulation: E-Commerce Customer Value Prediction

> **Teaching Document v2** — Read this in full before writing any data cleaning or feature engineering code.
> This document supersedes v1 and covers the complete reasoning behind every design decision.

---

## Part 0: Why This Document Exists

The previous formulation was identified as:

> *"Technically executable, but misleading. The regression is not predictive — it is reconstructive."*

That formulation predicted **total historical customer spending** — a number that already exists in the dataset. The model would be learning to reconstruct something it already knows, not forecast something unknown.

This document redesigns the problem from scratch as a **genuine future-value prediction problem**, and explains every decision so you can defend it in an interview or code review.

---

## Part 1: Understanding the Raw Data

### What one row represents

In the Online Retail II dataset, **one row is a single line item on an invoice**.

```
Invoice | StockCode | Description           | Qty | InvoiceDate         | Price | Customer ID | Country
489434  | 85048     | 15CM XMAS GLASS BALL  | 12  | 2009-12-01 07:45:00 | 6.95  | 13085       | United Kingdom
489434  | 79323P    | PINK CHERRY LIGHTS    | 12  | 2009-12-01 07:45:00 | 6.75  | 13085       | United Kingdom
489435  | 22041     | RECORD FRAME 7"       | 48  | 2009-12-01 09:00:00 | 2.10  | 14085       | United Kingdom
```

One row is not one customer. One row is not one purchase. It is one product within one purchase.

There is no `revenue` column. You must compute: `line_total = Quantity × Price`.
There is no `customer_value` column. You must engineer it.

### Dataset date range and transaction density

The Online Retail II dataset spans:

```
Start:  2009-12-01
End:    2011-12-09

Total duration: ~24 months (2 years)
```

This is important because the temporal windows you choose must be **proportionate to this range**. If you choose a historical window of only 1 month and a future window of 1 month, you have 22 months of unused data and your features will be very sparse. If you choose a future window that is too long, you leave almost no historical data for building features.

A reasonable design rule:

```
Historical window: ~75-80% of total time span  =>  ~18-21 months
Future window:     ~20-25% of total time span  =>  ~3-6 months
```

This gives enough historical depth for features like purchase rate, tenure, and recency to be meaningful, while leaving enough future data for the target to have real signal.

**Proposed time split for this project:**

```
Historical window:   2009-12-01  ->  2011-08-31   (~21 months)
Prediction date:     2011-09-01
Future window:       2011-09-01  ->  2011-12-09   (~3.3 months)
```

This is not pulled from thin air. The reasoning:

1. September 2011 leaves exactly 3 full months of future data. That is enough for meaningful purchase signal.
2. 21 months of history gives every customer enough time to establish a behavioral pattern (recency, frequency, monetary).
3. The split is close to 87%/13% by time. This is intentionally asymmetric — features need more history than targets need future observations.

---

## Part 2: Three Formulations Compared

### Formulation A — Historical Total Spending Reconstruction

**What it asks:**
"Can I predict how much a customer has spent in total, using aggregated summaries of their own transactions?"

**What X would contain:**

```
avg_order_value  (mean spend per transaction)
num_orders       (count of transactions)
num_products     (distinct product categories)
days_active      (days between first and last purchase)
```

All computed from **the entire dataset**.

**What y would contain:**

```
total_historical_spend = sum(Quantity x Price) across all transactions, entire dataset
```

**Is it truly predictive?**
No. It is **reconstructive**. The target is a direct function of the features. For example:

```
total_spend ~= avg_order_value * num_orders
```

The model is learning an algebraic identity, not a behavioral pattern. Both the feature and the target are computed from the same time window, so the model can trivially reconstruct y from X.

**Is target leakage possible?**
Yes, and it is essentially guaranteed. `total_spend` is a linear combination of `avg_order_value` and `num_orders`. The model will appear to have near-perfect R², but it has learned nothing.

**How difficult is it?**
Very easy to implement. But that ease is a warning sign, not a virtue.

**How interpretable is it?**
The coefficients tell you "how much does avg_order_value explain total_spend." That is a mathematical truism, not a business insight.

**How suitable for studying regularization?**
Poor. When the model is fitting an algebraic identity, Ridge, Lasso, and Elastic Net will all converge to nearly identical coefficients. Regularization has no meaningful work to do when the signal is perfect. You will incorrectly conclude that "regularization has no effect."

**Verdict:** Do not use this formulation. It produces misleading results that teach you nothing about regularization or prediction.

---

### Formulation B — Future Customer Revenue Prediction (Recommended)

**What it asks:**
"Given what we know about a customer up to a fixed date, how much revenue will they generate in the following 3 months?"

**What X would contain:**

```
recency          - days since last purchase BEFORE prediction date
frequency        - number of invoices BEFORE prediction date
monetary         - total spend BEFORE prediction date
avg_order_value  - monetary / frequency
num_products     - distinct StockCodes BEFORE prediction date
tenure           - days since first purchase BEFORE prediction date
purchase_rate    - frequency / tenure
country_encoded  - one-hot encoding of Country
```

All features computed **exclusively from transactions before 2011-09-01**.

**What y would contain:**

```
future_revenue = sum(Quantity x Price)
                 for InvoiceDate >= 2011-09-01
                 for each Customer ID
```

Target computed **exclusively from transactions on or after 2011-09-01**.

**Is it truly predictive?**
Yes. The features describe what a customer has done in the past. The target describes what they will do in the future. The model must learn a generalizable relationship — not a mathematical identity.

**Is target leakage possible?**
Only if you are careless with the time split. With a strict cutoff, leakage is structurally impossible: X is built from the left side of the cutoff, y is built from the right side.

**How difficult is it?**
Moderate. Requires careful temporal splitting and handling of customers who appear in only one window.

**How interpretable is it?**
High business interpretability: "Customers who bought more frequently and more recently tend to generate more revenue in the next quarter."

**How suitable for studying regularization?**
Excellent. RFM features (recency, frequency, monetary) are naturally correlated with each other. The target is noisy and right-skewed (many zeros, a few high-value customers). This is precisely the condition where:

- Ridge keeps all correlated features but shrinks their coefficients together.
- Lasso zeroes out some correlated features and keeps others.
- Elastic Net finds a compromise.
- OLS overfits on correlated features when the sample is small.

You will observe genuine, explainable differences between the four models.

**Verdict:** This is the correct formulation for this project.

---

### Formulation C — Customer Lifetime Value (CLV) Prediction

**What it asks:**
"What is the total revenue I expect from this customer over their entire remaining relationship with the store?"

**What X would contain:**
Same historical RFM features as Formulation B.

**What y would contain:**
An estimate of CLV. This is not a number you can read directly from the dataset. You must estimate it using:

- BG/NBD model to estimate future purchase rate and churn probability.
- Gamma-Gamma model to estimate average future transaction value.
- A time discount rate (e.g., 10% annual) to compute present value of expected future spend.

```
CLV ~= (expected_transactions * expected_avg_order_value) / (1 + discount_rate)
```

This is a probabilistic estimate, not an observation. You are predicting against a label that is itself a model output.

**Is it truly predictive?**
Yes, in principle. But you are predicting a derived quantity, not an observed one.

**Is target leakage possible?**
If CLV is estimated using all available data, yes. The CLV estimate may incorporate future transactions, making it an information leak.

**How difficult is it?**
Very high. CLV computation alone is a separate modeling problem. You would implement at minimum two probabilistic models (BG/NBD + Gamma-Gamma) before fitting a single regression.

**How interpretable is it?**
The CLV label is already a model output. Your regression becomes a model of a model, which is difficult to interpret and validate.

**How suitable for studying regularization?**
The label engineering complexity dominates the project. You would spend 80% of your time building the CLV estimator and 20% running regressions. The regularization comparison would be buried in implementation detail.

**Verdict:** Correct concept for a business problem. Wrong scope for a beginner regression + regularization study.

---

### Side-by-side comparison

| Criterion | A: Historical Reconstruction | B: Future Revenue | C: CLV |
|---|---|---|---|
| Genuinely predictive? | No | **Yes** | Yes |
| Avoids leakage by design? | No | **Yes** | Requires care |
| Features and target from same window? | Yes — problem | **No — correct** | No |
| Reveals regularization behavior? | No | **Yes** | Yes, but obscured |
| Difficulty | Low | **Moderate** | Very high |
| Beginner-appropriate? | Misleading | **Yes** | No |
| Business-interpretable? | No | **Yes** | Yes |
| Can be defended in an interview? | No | **Yes** | Yes, with caveats |

---

## Part 3: The Concept of a Temporal Prediction Problem

This section teaches the conceptual framework that makes prediction genuine rather than reconstructive.

### 3.1 The Observation Date / Prediction Date

The **prediction date** (also called the **cutoff date** or **snapshot date**) is the single most important design decision in any time-aware prediction problem.

It represents the moment in time at which the model would, in a real business scenario, be deployed to make a decision:

> "It is September 1, 2011. We want to know which customers will generate the most revenue in the next quarter, so we can decide where to spend our marketing budget."

**Everything before this date is observable.** You have it. You can use it to build features.
**Everything after this date is unknown.** You do not have it. It is what you are predicting.

```
prediction_date = 2011-09-01

Known to the model:   [2009-12-01 -------------------- 2011-08-31]
Unknown to the model:                                              [2011-09-01 ---- 2011-12-09]
                                                                         ^
                                                                  you stand here
```

### 3.2 The Historical Window

The **historical window** is the time period from which you build features. It begins at the earliest available data point and ends exactly at `prediction_date - 1 day`.

```
historical_window = [dataset_start, prediction_date)
                  = [2009-12-01, 2011-09-01)
                  = all transactions with InvoiceDate < 2011-09-01
```

From this window, you compute all customer behavioral features: how recently they bought, how often, how much, what they bought, for how long they have been a customer.

### 3.3 The Future Window

The **future window** is the time period from which you compute the target. It begins at `prediction_date` and ends at the last available data point.

```
future_window = [prediction_date, dataset_end]
              = [2011-09-01, 2011-12-09]
              = all transactions with InvoiceDate >= 2011-09-01
```

From this window, you aggregate the total spend per customer. That aggregated value becomes `y`.

### 3.4 Feature Period vs. Target Period

These two concepts are often conflated by beginners. They must never overlap.

| | Feature Period | Target Period |
|---|---|---|
| Also called | Historical window | Future window |
| Time range | Before prediction date | After prediction date |
| What you compute | X (features) | y (target) |
| Can they share data? | **Never** | **Never** |

If a single transaction is used both to compute a feature and to compute the target, you have introduced leakage.

### 3.5 Temporal Leakage

**Temporal leakage** is when information from the future accidentally enters the feature set.

It is fundamentally different from classic leakage (where the target variable itself is included as a feature). Temporal leakage is sneakier because the features look legitimate — they are the correct columns, the correct aggregations — but they are computed over the wrong time window.

**Classic leakage example:**

```python
# Including the target directly as a feature
X['future_spend'] = df.groupby('Customer ID')['revenue'].sum()  # WRONG
```

**Temporal leakage example:**

```python
# Feature looks correct (total spend), but is computed from ALL time
X['total_spend'] = df.groupby('Customer ID')['revenue'].sum()
# WRONG: df includes future transactions, so total_spend leaks future info
```

**Clean example:**

```python
prediction_date = pd.Timestamp('2011-09-01')
hist   = df[df['InvoiceDate'] <  prediction_date]
future = df[df['InvoiceDate'] >= prediction_date]

X = hist.groupby('Customer ID').agg(total_spend=('revenue', 'sum'), ...)
y = future.groupby('Customer ID').agg(future_revenue=('revenue', 'sum'))
```

### 3.6 Why Random Splitting Can Be Problematic in Time-Series Problems

Most introductory ML courses teach `train_test_split(X, y, test_size=0.2)`. For this problem, that is **not sufficient on its own**, and can be actively misleading if applied to the wrong level.

**Random split on customers (acceptable here):**
After the temporal split creates your feature matrix X and target y (one row per customer), you can safely do a random train/test split by customer ID. Each row already respects the temporal boundary — X was built from the past, y from the future. Randomly assigning customers to train/test is fine.

**Random split on transactions (dangerous):**
If you were to randomly split the raw transaction data into train and test sets without respecting time, a customer's January transactions might inform their August features in the training set, while their March transactions might appear in the test set. The temporal structure is destroyed.

**The general rule:** Always split time first, then split customers. Never split customers before splitting time.

### 3.7 Why Future Information Must Not Enter X

This is worth stating explicitly, because it is the single most common mistake in customer analytics projects.

The model you train will eventually be deployed in a real environment. When it runs in production, it will stand at "today" and look backward. It will never be able to look forward. If it was trained on features that included forward-looking information, it learned patterns that literally cannot be reproduced at inference time.

Concretely: if your feature `total_spend` was computed using the entire two-year dataset including September–December 2011, then when you run this model in September 2011 in production, the feature you feed it will be different (lower, because the future hasn't happened yet). The model will make predictions based on a distribution it was never trained on.

This is why the temporal split is not just a methodological nicety — it is a requirement for the model to be deployable.

### 3.8 Why the Target Must Be Calculated Separately From the Feature Window

Even if you are meticulous about not including future transactions in your features, computing `y` as "total minus historical" is still wrong:

```python
# Incorrect — even if you think you're separating them
y = total_spend - historical_spend
# y still depends on the same feature-period transactions
```

The target must be computed from scratch, independently, using only transactions in the future window. There should be no mathematical relationship between the feature computation and the target computation. The only shared variable is `Customer ID`.

---

## Part 4: Designing the Final Prediction Problem

### 4.1 Formal Business Problem

> A UK-based e-commerce company wants to identify which customers are most likely to generate significant revenue in the next quarter, so the marketing team can prioritize outreach and allocate campaign budget effectively.
>
> **The question is:** Given a customer's purchase history up to a fixed date, how much revenue will that customer generate in the next 3 months?

### 4.2 Formal Machine Learning Problem

> **Supervised regression.** Given a feature matrix X of shape `(n_customers, p_features)` — where each row describes one customer's behavioral history before the prediction date — fit a regression model that minimizes the mean squared error between predicted and actual future revenue.
>
> Study how regularization penalties (L1, L2, L1+L2) affect coefficient estimation, model sparsity, and generalization performance as the regularization parameter lambda varies.

### 4.3 Definition of X — The Feature Matrix

```
Shape:  (n_customers, p_features)
Rows:   One unique customer (Customer ID)
Cols:   Behavioral features computed from InvoiceDate < 2011-09-01
```

| Column name | Type | Definition |
|---|---|---|
| `recency` | float | Days between the customer's last purchase and 2011-09-01 |
| `frequency` | int | Number of distinct invoices in historical window |
| `monetary` | float | Sum of (Quantity x Price) in historical window |
| `avg_order_value` | float | monetary / frequency |
| `num_unique_products` | int | Distinct StockCodes purchased in historical window |
| `tenure` | int | Days between first and last purchase in historical window |
| `purchase_rate` | float | frequency / tenure (avg invoices per day) |
| `country` | encoded | One-hot or frequency-encoded Country |

**All columns must be computed exclusively from `InvoiceDate < 2011-09-01`.**

### 4.4 Definition of y — The Target Vector

```
Shape:   (n_customers,)
Element: Total revenue generated by customer i during [2011-09-01, 2011-12-09]
```

```
y[i] = sum(Quantity x Price)
       where Customer ID == i
         and InvoiceDate >= 2011-09-01
```

If customer `i` made no purchases after 2011-09-01, then `y[i] = 0`.

**Because y is right-skewed** (many customers with y=0, a few with very high values), the working target for model training is the log-transformed version:

```
y_model = log(y + 1)        # numpy: np.log1p(y)
```

After prediction, reverse with:

```
y_hat = exp(y_model_pred) - 1    # numpy: np.expm1(y_model_pred)
```

### 4.5 Prediction Date

```
prediction_date = 2011-09-01
```

**Why this date:**

1. The dataset ends 2011-12-09. Choosing 2011-09-01 leaves exactly 99 days of future data — enough for meaningful purchase behavior to accumulate.
2. 21 months of history (Dec 2009 – Aug 2011) gives customers enough time to establish stable behavioral patterns. A customer with 21 months of history has a well-defined recency, frequency, and monetary profile.
3. Quarters align naturally with business planning cycles. September–December is the holiday shopping quarter for UK retail, which contains the highest transaction density in the dataset. Predicting behavior in the busiest quarter is the most business-relevant question.

### 4.6 Historical Feature Window

```
Start:    2009-12-01  (earliest InvoiceDate in dataset)
End:      2011-08-31  (last day before prediction_date)
Filter:   InvoiceDate < pd.Timestamp('2011-09-01')
Duration: ~21 months
```

### 4.7 Future Target Window

```
Start:    2011-09-01  (prediction_date, inclusive)
End:      2011-12-09  (last InvoiceDate in dataset, inclusive)
Filter:   InvoiceDate >= pd.Timestamp('2011-09-01')
Duration: ~3.3 months
```

### 4.8 What One Row of the Final Modeling Dataset Represents

One row represents **a single customer** at the moment of prediction (2011-09-01), described by their historical behavior, with an associated future revenue figure.

```
Customer ID | recency | frequency | monetary | avg_order_value | ... | future_revenue | y_log
13085       | 59      | 4         | 466.65   | 116.66          | ... | 243.50         | 5.497
14085       | 120     | 2         | 89.20    | 44.60           | ... | 0.00           | 0.000
17908       | 14      | 9         | 1204.00  | 133.78          | ... | 890.25         | 6.793
```

Each row is an independent observation. The ML model sees each row as a separate customer, with no temporal ordering between rows. The temporal ordering was already encoded into the feature values during the feature engineering step.

### 4.9 Why This Is a Genuine Prediction Problem

A prediction problem has a specific property: **the answer is unknown at the time of prediction**.

At 2011-09-01, the future revenue of each customer is genuinely unknown. It has not happened yet. No amount of looking at the historical data can tell you exactly what will happen next quarter — some customers will return, some will not; some will spend more than before, some will spend less.

The model must learn a generalizable relationship: "customers with high frequency and low recency in their history tend to generate more future revenue." This is a real statistical pattern that holds across customers and across time, but it is not a mathematical certainty.

Compare with Formulation A:

- In Formulation A, `total_spend = avg_order_value * num_orders` is a **tautology**. It is always true. There is nothing to predict.
- In Formulation B, `future_spend = f(recency, frequency, monetary, ...)` is a **hypothesis**. It may or may not hold. The model must earn its accuracy.

This is precisely what you need to be able to explain in an interview:

> *"I defined the target as future revenue rather than historical spend because prediction requires genuine uncertainty. If the target is a deterministic function of the features, there is nothing for the model to learn — and more importantly, nothing for regularization to do. Regularization is designed to prevent overfitting on noisy, uncertain relationships. It can only demonstrate its value when the prediction problem is genuinely uncertain."*

---

## Part 5: Timeline Diagram

```
DATASET TIMELINE  (Online Retail II)
=============================================================================

2009-12-01                                  2011-09-01        2011-12-09
    |                                            |                 |
    v                                            v                 v
----*--------------------------------------------*-----------------*--------
    |                                            |                 |
    |<------------ HISTORICAL WINDOW ----------->|<- FUTURE WINDOW>|
    |               (~21 months)                 |   (~3 months)   |
    |                                            |                 |
    |  Source of X (features)                    |  Source of y    |
    |                                            |  (target)       |
    |  recency, frequency, monetary,             |                 |
    |  avg_order_value, tenure,                  |  future_revenue |
    |  purchase_rate, num_products,              |  = sum(Qty*Price)|
    |  country_encoded                           |  per customer   |
    |                                            |                 |
    |                           ^ PREDICTION DATE = 2011-09-01     |
    |                           |                                  |
    |                    "You stand here."                         |
    |                    Past is observable.                       |
    |                    Future is unknown.                        |
    |                    This is what prediction means.            |

=============================================================================

                        LEAKAGE BOUNDARY
                        ================
    Features (X)               |               Target (y)
    ---------------------------|---------------------------
    ONLY from left side        |    ONLY from right side
    InvoiceDate < 2011-09-01   |    InvoiceDate >= 2011-09-01
                               |
    If any transaction crosses |
    this line into X,          |
    it is LEAKAGE.             |

=============================================================================
```

---

## Part 6: Three Concrete Customer Examples

### Customer A: Loyal high-value customer

**Historical transactions (before 2011-09-01):**

```
Date        | Invoice | Revenue
2010-01-15  | 550231  | 124.50
2010-04-02  | 553892  |  87.20
2010-08-11  | 561034  | 198.00
2010-11-22  | 568901  | 156.75
2011-03-08  | 578234  | 211.40
2011-07-14  | 589012  | 183.60
```

**Computed features (X row):**

```
recency          = days(2011-09-01 - 2011-07-14) = 49 days
frequency        = 6 invoices
monetary         = 961.45
avg_order_value  = 961.45 / 6 = 160.24
tenure           = days(2011-07-14 - 2010-01-15) = 544 days
purchase_rate    = 6 / 544 = 0.011 orders/day
num_products     = 14 unique products (illustrative)
```

**Future transactions (on/after 2011-09-01):**

```
Date        | Invoice | Revenue
2011-09-23  | 594411  | 201.30
2011-11-07  | 601234  | 289.50
2011-12-01  | 608901  | 174.20
```

**Target (y):**

```
future_revenue = 201.30 + 289.50 + 174.20 = 665.00
y_log          = log(665.00 + 1) = 6.501
```

**Interpretation:** Long tenure, high frequency, low recency (bought 49 days before cutoff). A regularized regression should assign strong positive weight to frequency and monetary for predicting high future revenue.

---

### Customer B: Occasional buyer who churned

**Historical transactions (before 2011-09-01):**

```
Date        | Invoice | Revenue
2010-02-08  | 551100  | 42.80
2010-05-19  | 556023  | 67.40
```

**Computed features (X row):**

```
recency          = days(2011-09-01 - 2010-05-19) = 471 days
frequency        = 2 invoices
monetary         = 110.20
avg_order_value  = 110.20 / 2 = 55.10
tenure           = days(2010-05-19 - 2010-02-08) = 100 days
purchase_rate    = 2 / 100 = 0.020 orders/day
num_products     = 4 unique products (illustrative)
```

**Future transactions (on/after 2011-09-01):**

```
(none)
```

**Target (y):**

```
future_revenue = 0.00
y_log          = log(0 + 1) = 0.000
```

**Interpretation:** Despite a non-trivial purchase rate in their brief history, a recency of 471 days is a near-definitive churn signal. This is exactly the kind of uncertain, noisy case where a regularized model is more reliable than OLS — because OLS might over-weight the purchase_rate feature while ignoring the recency signal.

---

### Customer C: New customer with growing engagement

**Historical transactions (before 2011-09-01):**

```
Date        | Invoice | Revenue
2011-04-14  | 581234  |  89.20
2011-06-30  | 586011  | 134.60
2011-08-12  | 591234  | 178.90
```

**Computed features (X row):**

```
recency          = days(2011-09-01 - 2011-08-12) = 20 days
frequency        = 3 invoices
monetary         = 402.70
avg_order_value  = 402.70 / 3 = 134.23
tenure           = days(2011-08-12 - 2011-04-14) = 120 days
purchase_rate    = 3 / 120 = 0.025 orders/day
num_products     = 8 unique products (illustrative)
```

**Future transactions (on/after 2011-09-01):**

```
Date        | Invoice | Revenue
2011-09-28  | 594800  | 156.40
2011-11-15  | 602111  | 210.00
```

**Target (y):**

```
future_revenue = 156.40 + 210.00 = 366.40
y_log          = log(366.40 + 1) = 5.905
```

**Interpretation:** Short tenure but very low recency (bought 20 days before cutoff), increasing order values, and the highest purchase_rate of the three examples. Note that this customer has a shorter history than Customer A but generates substantial future revenue. No single feature tells the full story — which is exactly why multi-feature regression with regularization is the appropriate model.

---

## Part 7: How to Defend This Design in an Interview

**Q: "Why did you define customer value as future revenue rather than historical spend?"**

> *"Historical spend is already known — predicting it is reconstructive, not predictive. The model would fit an algebraic identity (total_spend ≈ avg_order_value × num_orders) and learn nothing generalizable. Future revenue, computed from transactions after a fixed cutoff date, is genuinely unknown at prediction time. The model must learn behavioral patterns — how recency, frequency, and monetary history relate to future purchase behavior — which is the actual business question."*

**Q: "How did you ensure the regression was genuinely predictive rather than reconstructive?"**

> *"I enforced a strict temporal boundary. The feature matrix X was computed exclusively from transactions before September 1, 2011. The target y was computed exclusively from transactions on or after that date. These two computations share no data — the only connection between them is the Customer ID used to join the two aggregations. This means the model cannot memorize the answer. It must learn a generalizable relationship across customers."*

**Q: "Why is this formulation appropriate for studying regularization?"**

> *"Regularization is most informative when: (1) the features are correlated with each other, and (2) the prediction is genuinely uncertain. RFM features — recency, frequency, monetary — are naturally correlated. The future revenue target is noisy: some customers return and spend heavily, others churn. In this setting, Ridge shrinks all correlated coefficients together, Lasso zeros out some and keeps others, and Elastic Net balances both. OLS overfits on the correlated features. These differences are only observable in a genuine prediction problem. A reconstructive problem masks all of them — every model looks identical."*

**Q: "Why did you choose September 2011 as the cutoff?"**

> *"The dataset spans December 2009 to December 2011 — 24 months. I chose September 2011 to leave 3 months of future data for the target window, which is proportionate and meaningful. It also aligns with the start of the holiday quarter, the busiest period in UK e-commerce. Using 21 months of history gives customers enough time to develop observable behavioral patterns. The 87/13 time split is intentionally asymmetric — you need more history to build good features than you need future window to observe purchase behavior."*

---

## Part 8: Summary of Final Design Decisions

| Design decision | Choice | Reason |
|---|---|---|
| Unit of analysis | Customer | Prediction at customer level, not transaction level |
| Target variable | Future revenue (next ~3 months) | Genuinely unknown at prediction time |
| Prediction date | 2011-09-01 | Leaves 3 months of future data; 21 months of history |
| Historical window | 2009-12-01 – 2011-08-31 | All available history before cutoff |
| Future window | 2011-09-01 – 2011-12-09 | Last available data; covers holiday quarter |
| Target transform | log(y + 1) | Right-skewed distribution; log normalizes it for regression |
| Feature basis | RFM + derived features | Proven customer analytics framework; natural correlations reveal regularization behavior |
| Train/test split | Random split by Customer ID | Safe after temporal split; each row already respects the time boundary |
| Models to compare | OLS, Ridge, Lasso, Elastic Net | Covers unpenalized baseline and all three standard regularization families |

---

## Part 9: Pre-Code Checklist

Use this as a mental gate before writing a single line of data cleaning or modeling code.

```
[ ] 1. Have I defined prediction_date explicitly?
        prediction_date = pd.Timestamp('2011-09-01')

[ ] 2. Have I split the raw data into historical and future DataFrames
        using prediction_date, with NO overlap?
        hist   = df[df['InvoiceDate'] <  prediction_date]
        future = df[df['InvoiceDate'] >= prediction_date]

[ ] 3. Are ALL features computed from hist only?
        X = hist.groupby('Customer ID').agg(...)

[ ] 4. Is y computed from future only?
        y = future.groupby('Customer ID').agg(
                future_revenue=('revenue', 'sum')
            )

[ ] 5. Does X contain any column derived from future?
        If yes -> leakage. Remove that column.

[ ] 6. Are X and y joined on Customer ID only?
        modeling_df = X.join(y, how='left').fillna({'future_revenue': 0})

[ ] 7. Does one row of the final modeling_df represent exactly one customer?
        assert modeling_df.index.is_unique

[ ] 8. Have I applied log(y + 1) before training?
        modeling_df['y_log'] = np.log1p(modeling_df['future_revenue'])

[ ] 9. Is the train/test split done by Customer ID (rows), not by date?
        X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
        (This is safe — temporal split was already done in step 2.)

[ ] 10. Can I explain to a reviewer why this is genuinely predictive?
         "Features are from the past. Target is from the future.
          They share no transactions. The model must generalize."
```

---

*This document defines the complete conceptual foundation for the project. No data cleaning, feature engineering, or modeling code should be written until the reasoning in every section above is understood and agreed upon.*
