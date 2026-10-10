"""
tests/test_models.py — Unit tests for the models sub-package.

Tests cover training, evaluation, prediction, and save/load consistency.
All tests use synthetic in-memory data; no real dataset is required.
"""

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from ecommerce_customer_value_prediction.features.build_features import (
    PREDICTION_DATE,
    build_customer_dataset,
)
from ecommerce_customer_value_prediction.models.evaluate import evaluate
from ecommerce_customer_value_prediction.models.predict import predict
from ecommerce_customer_value_prediction.models.train import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COL,
    build_pipeline,
    load_model,
    save_model,
    train,
)


# ---------------------------------------------------------------------------
# Synthetic customer dataset fixture
# ---------------------------------------------------------------------------


def _make_customer_df(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic customer-level dataset large enough for Ridge to fit.
    Mirrors the structure produced by ``build_customer_dataset``.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "Customer ID": range(1, n + 1),
            "recency_days": rng.integers(1, 365, size=n).astype(float),
            "frequency": rng.integers(1, 50, size=n).astype(float),
            "monetary": rng.uniform(10, 5000, size=n),
            "avg_basket_value": rng.uniform(5, 200, size=n),
            "unique_products": rng.integers(1, 30, size=n).astype(float),
            "tenure_days": rng.integers(0, 700, size=n).astype(float),
            "purchase_rate": rng.uniform(0, 0.5, size=n),
            "country": rng.choice(["United Kingdom", "Germany", "France"], size=n),
            "future_revenue": rng.uniform(0, 3000, size=n),
        }
    )
    df["log1p_future_revenue"] = np.log1p(df["future_revenue"])
    return df


# ---------------------------------------------------------------------------
# Tests: build_pipeline
# ---------------------------------------------------------------------------


def test_build_pipeline_returns_sklearn_pipeline():
    pipeline = build_pipeline()
    assert isinstance(pipeline, Pipeline)


def test_build_pipeline_has_preprocessor_and_regressor():
    pipeline = build_pipeline()
    step_names = [name for name, _ in pipeline.steps]
    assert "preprocessor" in step_names
    assert "regressor" in step_names


# ---------------------------------------------------------------------------
# Tests: train
# ---------------------------------------------------------------------------


def test_train_returns_fitted_pipeline_and_metadata():
    df = _make_customer_df()
    pipeline, metadata = train(df)
    assert isinstance(pipeline, Pipeline)
    assert isinstance(metadata, dict)


def test_train_metadata_keys():
    df = _make_customer_df()
    _, metadata = train(df)
    for key in ("n_customers", "numeric_features", "categorical_features", "target_col", "alpha"):
        assert key in metadata


def test_train_metadata_n_customers():
    df = _make_customer_df(n=30)
    _, metadata = train(df)
    assert metadata["n_customers"] == 30


def test_train_raises_on_missing_columns():
    df = _make_customer_df().drop(columns=["recency_days"])
    with pytest.raises(ValueError, match="recency_days"):
        train(df)


def test_train_pipeline_can_predict():
    df = _make_customer_df()
    pipeline, _ = train(df)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    preds = pipeline.predict(X)
    assert len(preds) == len(df)


# ---------------------------------------------------------------------------
# Tests: save_model / load_model
# ---------------------------------------------------------------------------


def test_save_and_load_model_roundtrip():
    df = _make_customer_df()
    pipeline, metadata = train(df)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.pkl"
        metadata_path = Path(tmpdir) / "metadata.json"

        save_model(pipeline, metadata, model_path=model_path, metadata_path=metadata_path)

        assert model_path.exists()
        assert metadata_path.exists()

        loaded = load_model(model_path)
        assert isinstance(loaded, Pipeline)

        # Predictions must be identical before and after round-trip
        X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        preds_before = pipeline.predict(X)
        preds_after = loaded.predict(X)
        np.testing.assert_array_almost_equal(preds_before, preds_after)


def test_save_model_metadata_json_valid():
    df = _make_customer_df()
    pipeline, metadata = train(df)

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_path = Path(tmpdir) / "metadata.json"
        save_model(pipeline, metadata, model_path=Path(tmpdir) / "m.pkl",
                   metadata_path=metadata_path)
        with open(metadata_path) as f:
            loaded_meta = json.load(f)
        assert loaded_meta["n_customers"] == len(df)


def test_load_model_raises_if_not_found():
    with pytest.raises(FileNotFoundError):
        load_model(Path("nonexistent_model.pkl"))


# ---------------------------------------------------------------------------
# Tests: evaluate
# ---------------------------------------------------------------------------


def test_evaluate_returns_all_metric_keys():
    df = _make_customer_df(n=60)
    train_df = df.iloc[:40]
    test_df = df.iloc[40:]
    pipeline, _ = train(train_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        metrics = evaluate(pipeline, test_df,
                           metrics_path=Path(tmpdir) / "metrics.json", save=True)

    for key in ("n_test_customers", "rmse_log", "mae_log", "r2_log",
                "rmse_gbp", "mae_gbp", "median_ae_gbp"):
        assert key in metrics


def test_evaluate_n_test_customers_correct():
    df = _make_customer_df(n=60)
    train_df, test_df = df.iloc[:40], df.iloc[40:]
    pipeline, _ = train(train_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        metrics = evaluate(pipeline, test_df,
                           metrics_path=Path(tmpdir) / "metrics.json", save=False)

    assert metrics["n_test_customers"] == 20


def test_evaluate_saves_json():
    df = _make_customer_df(n=60)
    pipeline, _ = train(df.iloc[:40])

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "metrics.json"
        evaluate(pipeline, df.iloc[40:], metrics_path=out, save=True)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "r2_log" in data


def test_evaluate_raises_on_missing_column():
    df = _make_customer_df(n=30)
    pipeline, _ = train(df)
    with pytest.raises(ValueError, match="recency_days"):
        evaluate(pipeline, df.drop(columns=["recency_days"]))


# ---------------------------------------------------------------------------
# Tests: predict
# ---------------------------------------------------------------------------


def test_predict_adds_prediction_columns():
    df = _make_customer_df()
    pipeline, _ = train(df)
    result = predict(df, pipeline=pipeline)
    assert "pred_log1p_future_revenue" in result.columns
    assert "pred_future_revenue" in result.columns


def test_predict_output_length():
    df = _make_customer_df(n=20)
    pipeline, _ = train(df)
    result = predict(df, pipeline=pipeline)
    assert len(result) == 20


def test_predict_revenue_non_negative():
    """expm1 of any real number is >= -1; for reasonable inputs it should be > 0."""
    df = _make_customer_df()
    pipeline, _ = train(df)
    result = predict(df, pipeline=pipeline)
    # Predictions on training data with Ridge should be reasonable positive values
    assert result["pred_future_revenue"].median() > 0


def test_predict_raises_on_missing_feature():
    df = _make_customer_df()
    pipeline, _ = train(df)
    with pytest.raises(ValueError, match="recency_days"):
        predict(df.drop(columns=["recency_days"]), pipeline=pipeline)
