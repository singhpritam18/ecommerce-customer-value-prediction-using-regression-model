"""Run from the repository root after installation; no training or dataset needed."""

from pathlib import Path

import pandas as pd

from ecommerce_customer_value_prediction.models.predict import load_model, predict

root = Path(__file__).resolve().parents[1]
pipeline = load_model(root / "models" / "customer_value_model.joblib")
customers = pd.read_csv(root / "examples" / "customer_features.csv")
result = predict(customers.iloc[[0]], pipeline=pipeline)
print(result[["customer_reference", "pred_log1p_future_revenue", "pred_future_revenue"]])
