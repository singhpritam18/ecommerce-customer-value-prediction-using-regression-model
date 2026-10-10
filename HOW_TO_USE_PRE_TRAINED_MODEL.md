## Using the Pretrained Model

A trained customer-value prediction model is included with this repository.

If you only want to generate predictions, you **do not need to retrain the model, run the notebooks, or download the original training dataset**.

The packaged model already contains the fitted preprocessing pipeline and the selected regression model.

### 1. Clone the Repository

```bash
git clone https://github.com/singhpritam18/ecommerce-customer-value-prediction-using-regression-model.git
cd ecommerce-customer-value-prediction-using-regression-model
```

### 2. Create a Virtual Environment

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the Package

```bash
python -m pip install .
```

For development, use editable mode instead:

```bash
python -m pip install -e .
```

The pretrained model is included in the repository, so model training is not required for inference.

---

### 4. Expected Input

The model expects **customer-level historical features**, not raw transaction rows.

Required features:

| Feature | Description |
|---|---|
| `recency_days` | Days since the customer's most recent historical purchase |
| `frequency` | Number of distinct historical orders/invoices |
| `monetary` | Total historical customer revenue |
| `avg_basket_value` | Average historical revenue per order |
| `unique_products` | Number of distinct products purchased |
| `tenure_days` | Days between the customer's first and last historical purchase |
| `purchase_rate` | Historical purchase-rate feature |
| `country` | Customer country |

Example:

```python
import pandas as pd

customer = pd.DataFrame(
    [
        {
            "recency_days": 14,
            "frequency": 8,
            "monetary": 1450.0,
            "avg_basket_value": 181.25,
            "unique_products": 35,
            "tenure_days": 320,
            "purchase_rate": 0.025,
            "country": "United Kingdom",
        }
    ]
)
```

> The input values must be calculated using the same feature definitions used during model development.

---

### 5. Predict From Python

```python
import pandas as pd

from ecommerce_customer_value_prediction.models.predict import predict


customers = pd.DataFrame(
    [
        {
            "recency_days": 14,
            "frequency": 8,
            "monetary": 1450.0,
            "avg_basket_value": 181.25,
            "unique_products": 35,
            "tenure_days": 320,
            "purchase_rate": 0.025,
            "country": "United Kingdom",
        }
    ]
)

predictions = predict(customers)

print(
    predictions[
        [
            "pred_log1p_future_revenue",
            "pred_future_revenue",
        ]
    ]
)
```

The prediction API automatically:

1. validates the input,
2. loads the pretrained model,
3. applies the fitted numerical scaling,
4. applies the fitted categorical encoding,
5. generates the regression prediction,
6. converts the prediction back to the original revenue scale.

**No training occurs during prediction.**

---

### 6. Prediction Output

The prediction result contains two model outputs:

| Output | Meaning |
|---|---|
| `pred_log1p_future_revenue` | Prediction on the model's transformed `log1p` target scale |
| `pred_future_revenue` | Prediction transformed back to the original revenue scale |

For most applications, use:

```python
predicted_revenue = predictions.iloc[0]["pred_future_revenue"]

print(f"Predicted future revenue: £{predicted_revenue:,.2f}")
```

Example output:

```text
Predicted future revenue: £11.81
```

The value above is only an example. Actual predictions depend on the supplied customer features.

---

### 7. Predict Multiple Customers

Pass multiple rows in the same DataFrame:

```python
customers = pd.DataFrame(
    [
        {
            "recency_days": 14,
            "frequency": 8,
            "monetary": 1450.0,
            "avg_basket_value": 181.25,
            "unique_products": 35,
            "tenure_days": 320,
            "purchase_rate": 0.025,
            "country": "United Kingdom",
        },
        {
            "recency_days": 90,
            "frequency": 3,
            "monetary": 420.0,
            "avg_basket_value": 140.0,
            "unique_products": 12,
            "tenure_days": 250,
            "purchase_rate": 0.012,
            "country": "France",
        },
    ]
)

predictions = predict(customers)

print(
    predictions[
        [
            "country",
            "pred_future_revenue",
        ]
    ]
)
```

One prediction is returned for each input customer.

---

### 8. Batch Prediction From CSV

Predictions can also be generated without writing Python code.

Prepare a CSV such as:

```csv
recency_days,frequency,monetary,avg_basket_value,unique_products,tenure_days,purchase_rate,country
14,8,1450.0,181.25,35,320,0.025,United Kingdom
90,3,420.0,140.0,12,250,0.012,France
```

Run:

```bash
predict-customer-value --input customers.csv --output predictions.csv
```

The generated `predictions.csv` contains the original input columns together with:

```text
pred_log1p_future_revenue
pred_future_revenue
```

---

### 9. Preprocessing Is Already Included

Do **not** manually scale numerical variables or encode `country`.

The saved model contains the complete fitted pipeline:

```text
Customer Features
        ↓
StandardScaler
        +
OneHotEncoder
        ↓
Fitted Regression Model
        ↓
Revenue Prediction
```

Therefore, inference code should call:

```python
predict(customers)
```

and should **not** call:

```python
scaler.fit(...)
encoder.fit(...)
model.fit(...)
```

---

### 10. Raw Transactions Are Not Accepted Directly

The current prediction interface expects customer-level features.

It does not accept raw transaction fields such as:

```text
Invoice
StockCode
Quantity
Price
InvoiceDate
```

Raw transaction data must first be transformed into the required customer-level features using definitions consistent with model training.

---

### 11. Verify the Installation

To verify that the package and pretrained model work correctly:

```bash
python -m pytest tests/ -v
```

You can also create a small Python script using the example above and run:

```bash
python your_prediction_script.py
```

---

### 12. Important Model Limitations

This model is intended as an **educational/portfolio machine-learning prototype**.

The prediction should be interpreted as:

> An estimate of future customer revenue over the forecast horizon used during model development.

It should **not** be interpreted as guaranteed revenue or unlimited customer lifetime value.

The evaluation identified important limitations, including large errors for some high-value customers and negative revenue predictions for some observations.

The model should therefore not be used as the sole basis for consequential financial or customer decisions without further validation.

---

### Quick Start

If the repository is already cloned:

```bash
# Create environment
python -m venv .venv

# Activate — Windows
.\.venv\Scripts\Activate.ps1

# Install
python -m pip install .

# Predict from CSV
predict-customer-value --input customers.csv --output predictions.csv
```

Or from Python:

```python
from ecommerce_customer_value_prediction.models.predict import predict

predictions = predict(customers)
```

**The pretrained model is loaded automatically. No retraining is required.**