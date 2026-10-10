"""
tests/test_models.py — Unit tests for the model construction checkpoint.

Checkpoint objective
--------------------
Verify that the reusable source code can correctly construct and train
LinearRegression, Ridge, Lasso and ElasticNet under the same
preprocessing architecture.

No real dataset required — all tests use small synthetic DataFrames.

Test groups
-----------
1.  create_model — correct estimator type returned
2.  create_model — hyperparameters forwarded correctly
3.  create_model — invalid name raises ValueError
4.  build_pipeline — shared preprocessing architecture
5.  build_pipeline — all four estimators seat inside the same pipeline
6.  train — all four estimators fit on synthetic data
7.  train — prediction shape is correct for every estimator
8.  train — metadata records correct estimator name and hyperparameters
9.  train — target safety (Customer ID, future_revenue not in X)
10. save_model / load_model — round-trip (predictions identical)
11. evaluate — all metric keys present
12. predict — output columns and shape
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ecommerce_customer_value_prediction.models.evaluate import evaluate
from ecommerce_customer_value_prediction.models.predict import predict
from ecommerce_customer_value_prediction.models.train import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SUPPORTED_MODELS,
    TARGET_COL,
    build_pipeline,
    create_model,
    load_model,
    save_model,
    train,
)


# ---------------------------------------------------------------------------
# Shared synthetic fixture
# ---------------------------------------------------------------------------


def _make_df(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """
    Small synthetic customer-level DataFrame that mirrors the schema of
    ``build_customer_dataset`` output.  No real data required.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "Customer ID": range(1, n + 1),
            "recency_days":    rng.integers(1, 365, size=n).astype(float),
            "frequency":       rng.integers(1, 50,  size=n).astype(float),
            "monetary":        rng.uniform(10, 5000, size=n),
            "avg_basket_value":rng.uniform(5,  200,  size=n),
            "unique_products": rng.integers(1, 30,  size=n).astype(float),
            "tenure_days":     rng.integers(0, 700,  size=n).astype(float),
            "purchase_rate":   rng.uniform(0, 0.5,  size=n),
            "country":         rng.choice(["United Kingdom", "Germany", "France"], size=n),
            "future_revenue":  rng.uniform(0, 3000, size=n),
        }
    )
    df["log1p_future_revenue"] = np.log1p(df["future_revenue"])
    return df


# ===========================================================================
# GROUP 1 — create_model: correct estimator type
# ===========================================================================


def test_create_model_linear_returns_linear_regression():
    assert isinstance(create_model("linear"), LinearRegression)


def test_create_model_ridge_returns_ridge():
    assert isinstance(create_model("ridge"), Ridge)


def test_create_model_lasso_returns_lasso():
    assert isinstance(create_model("lasso"), Lasso)


def test_create_model_elasticnet_returns_elasticnet():
    assert isinstance(create_model("elasticnet"), ElasticNet)


# ===========================================================================
# GROUP 2 — create_model: hyperparameters forwarded correctly
# ===========================================================================


def test_create_model_ridge_alpha():
    m = create_model("ridge", alpha=10.0)
    assert m.alpha == 10.0


def test_create_model_lasso_alpha():
    m = create_model("lasso", alpha=0.05)
    assert m.alpha == 0.05


def test_create_model_lasso_default_max_iter():
    """Lasso needs max_iter=10_000 to avoid ConvergenceWarning on collinear features."""
    m = create_model("lasso")
    assert m.max_iter == 10_000


def test_create_model_elasticnet_alpha_and_l1_ratio():
    m = create_model("elasticnet", alpha=0.2, l1_ratio=0.7)
    assert m.alpha == 0.2
    assert m.l1_ratio == 0.7


def test_create_model_elasticnet_default_max_iter():
    """ElasticNet also uses coordinate descent — same max_iter default applies."""
    m = create_model("elasticnet")
    assert m.max_iter == 10_000


def test_create_model_linear_has_no_alpha():
    """LinearRegression has no alpha attribute — no regularisation."""
    m = create_model("linear")
    assert not hasattr(m, "alpha")


def test_create_model_case_insensitive():
    assert isinstance(create_model("RIDGE"),      Ridge)
    assert isinstance(create_model("Linear"),     LinearRegression)
    assert isinstance(create_model("LASSO"),      Lasso)
    assert isinstance(create_model("ElasticNet"), ElasticNet)


# ===========================================================================
# GROUP 3 — create_model: invalid name raises
# ===========================================================================


def test_create_model_raises_on_unknown_name():
    with pytest.raises(ValueError, match="Unknown model_name"):
        create_model("xgboost")


def test_create_model_raises_on_empty_string():
    with pytest.raises(ValueError):
        create_model("")


def test_supported_models_contains_exactly_four():
    assert set(SUPPORTED_MODELS) == {"linear", "ridge", "lasso", "elasticnet"}


# ===========================================================================
# GROUP 4 — build_pipeline: shared preprocessing architecture
# ===========================================================================


def test_build_pipeline_returns_sklearn_pipeline():
    assert isinstance(build_pipeline(), Pipeline)


def test_build_pipeline_has_two_steps_preprocessor_and_regressor():
    step_names = [name for name, _ in build_pipeline().steps]
    assert step_names == ["preprocessor", "regressor"]


def test_build_pipeline_preprocessor_is_column_transformer():
    p = build_pipeline()
    assert isinstance(p.named_steps["preprocessor"], ColumnTransformer)


def test_build_pipeline_numeric_transformer_is_standard_scaler():
    """
    StandardScaler is mandatory for Ridge, Lasso and ElasticNet.
    Without it, features on different scales receive unequal regularisation.
    """
    p = build_pipeline()
    ct = p.named_steps["preprocessor"]
    # ct.transformers is a list of (name, transformer, columns) 3-tuples
    tf_map = {name: tf for name, tf, _ in ct.transformers}
    assert isinstance(tf_map["num"], StandardScaler)


def test_build_pipeline_categorical_transformer_is_one_hot_encoder():
    p = build_pipeline()
    ct = p.named_steps["preprocessor"]
    tf_map = {name: tf for name, tf, _ in ct.transformers}
    assert isinstance(tf_map["cat"], OneHotEncoder)


def test_build_pipeline_numeric_features_are_project_features():
    p = build_pipeline()
    ct = p.named_steps["preprocessor"]
    _, _, num_cols = ct.transformers[0]  # ("num", scaler, cols)
    assert list(num_cols) == NUMERIC_FEATURES


def test_build_pipeline_categorical_features_are_project_features():
    p = build_pipeline()
    ct = p.named_steps["preprocessor"]
    _, _, cat_cols = ct.transformers[1]
    assert list(cat_cols) == CATEGORICAL_FEATURES


def test_build_pipeline_default_estimator_is_ridge():
    p = build_pipeline()
    assert isinstance(p.named_steps["regressor"], Ridge)


# ===========================================================================
# GROUP 5 — build_pipeline: all four estimators seat in the same pipeline
# ===========================================================================


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_build_pipeline_accepts_any_estimator(model_name):
    """Every estimator produced by create_model fits inside build_pipeline."""
    estimator = create_model(model_name)
    p = build_pipeline(estimator=estimator)
    assert isinstance(p, Pipeline)
    assert type(p.named_steps["regressor"]) is type(estimator)


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_build_pipeline_shares_same_preprocessor_for_all_estimators(model_name):
    """
    The preprocessor definition must be identical regardless of which estimator
    is used.  This ensures no model has private preprocessing logic.
    """
    p = build_pipeline(estimator=create_model(model_name))
    ct = p.named_steps["preprocessor"]
    # ct.transformers is a list of (name, transformer, columns) 3-tuples
    tf_map = {name: tf for name, tf, _ in ct.transformers}
    assert isinstance(tf_map["num"], StandardScaler)
    assert isinstance(tf_map["cat"], OneHotEncoder)


# ===========================================================================
# GROUP 6 — train: all four estimators fit on synthetic data
# ===========================================================================


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_train_fits_without_error(model_name):
    df = _make_df()
    pipeline, metadata = train(df, estimator=create_model(model_name))
    assert isinstance(pipeline, Pipeline)
    assert isinstance(metadata, dict)


# ===========================================================================
# GROUP 7 — train: prediction shape correct for every estimator
# ===========================================================================


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_train_prediction_shape_matches_input(model_name):
    df = _make_df(n=30)
    pipeline, _ = train(df, estimator=create_model(model_name))
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    preds = pipeline.predict(X)
    assert preds.shape == (30,)


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_train_predictions_are_finite(model_name):
    df = _make_df()
    pipeline, _ = train(df, estimator=create_model(model_name))
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    assert np.all(np.isfinite(pipeline.predict(X)))


# ===========================================================================
# GROUP 8 — train: metadata records estimator name and hyperparameters
# ===========================================================================


def test_train_linear_metadata_estimator_name():
    _, metadata = train(_make_df(), estimator=create_model("linear"))
    assert metadata["estimator"] == "LinearRegression"


def test_train_ridge_metadata_estimator_name():
    _, metadata = train(_make_df(), estimator=create_model("ridge", alpha=5.0))
    assert metadata["estimator"] == "Ridge"
    assert metadata["hyperparameters"]["alpha"] == 5.0


def test_train_lasso_metadata_records_alpha():
    _, metadata = train(_make_df(), estimator=create_model("lasso", alpha=0.05))
    assert metadata["estimator"] == "Lasso"
    assert metadata["hyperparameters"]["alpha"] == 0.05


def test_train_elasticnet_metadata_records_alpha_and_l1_ratio():
    _, metadata = train(_make_df(), estimator=create_model("elasticnet", alpha=0.1, l1_ratio=0.7))
    assert metadata["estimator"] == "ElasticNet"
    assert metadata["hyperparameters"]["alpha"] == 0.1
    assert metadata["hyperparameters"]["l1_ratio"] == 0.7


def test_train_metadata_contains_required_keys():
    _, metadata = train(_make_df())
    for key in ("n_customers", "numeric_features", "categorical_features",
                "target_col", "estimator", "hyperparameters"):
        assert key in metadata


def test_train_metadata_n_customers_correct():
    _, metadata = train(_make_df(n=25))
    assert metadata["n_customers"] == 25


def test_train_metadata_feature_lists_match_constants():
    _, metadata = train(_make_df())
    assert metadata["numeric_features"] == NUMERIC_FEATURES
    assert metadata["categorical_features"] == CATEGORICAL_FEATURES
    assert metadata["target_col"] == TARGET_COL


# ===========================================================================
# GROUP 9 — target safety: predictors exclude leakage columns
# ===========================================================================


def test_customer_id_not_in_numeric_features():
    assert "Customer ID" not in NUMERIC_FEATURES


def test_customer_id_not_in_categorical_features():
    assert "Customer ID" not in CATEGORICAL_FEATURES


def test_future_revenue_not_in_numeric_features():
    assert "future_revenue" not in NUMERIC_FEATURES


def test_future_revenue_not_in_categorical_features():
    assert "future_revenue" not in CATEGORICAL_FEATURES


def test_log1p_future_revenue_not_used_as_predictor():
    """log1p_future_revenue is the target, not a predictor."""
    assert "log1p_future_revenue" not in NUMERIC_FEATURES
    assert "log1p_future_revenue" not in CATEGORICAL_FEATURES


def test_train_raises_on_missing_feature_column():
    df = _make_df().drop(columns=["recency_days"])
    with pytest.raises(ValueError, match="recency_days"):
        train(df)


def test_target_col_is_log1p_future_revenue():
    assert TARGET_COL == "log1p_future_revenue"


# ===========================================================================
# GROUP 10 — save_model / load_model: round-trip
# ===========================================================================


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_save_load_roundtrip_predictions_identical(model_name):
    df = _make_df()
    pipeline, metadata = train(df, estimator=create_model(model_name))
    with tempfile.TemporaryDirectory() as tmpdir:
        mp = Path(tmpdir) / f"{model_name}.pkl"
        mep = Path(tmpdir) / f"{model_name}_meta.json"
        save_model(pipeline, metadata, model_path=mp, metadata_path=mep)
        loaded = load_model(mp)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    np.testing.assert_array_almost_equal(pipeline.predict(X), loaded.predict(X))


def test_load_model_raises_if_file_missing():
    with pytest.raises(FileNotFoundError):
        load_model(Path("nonexistent.pkl"))


def test_save_model_writes_valid_json_metadata():
    df = _make_df()
    pipeline, metadata = train(df)
    with tempfile.TemporaryDirectory() as tmpdir:
        mp = Path(tmpdir) / "m.pkl"
        mep = Path(tmpdir) / "m.json"
        save_model(pipeline, metadata, model_path=mp, metadata_path=mep)
        loaded_meta = json.loads(mep.read_text())
    assert loaded_meta["estimator"] == "Ridge"
    assert loaded_meta["n_customers"] == len(df)


# ===========================================================================
# GROUP 11 — evaluate: all metric keys present
# ===========================================================================


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_evaluate_returns_required_metric_keys(model_name):
    df = _make_df(n=60)
    pipeline, _ = train(df.iloc[:40], estimator=create_model(model_name))
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics = evaluate(pipeline, df.iloc[40:],
                           metrics_path=Path(tmpdir) / "m.json", save=False)
    for key in ("n_test_customers", "rmse_log", "mae_log", "r2_log",
                "rmse_gbp", "mae_gbp", "median_ae_gbp"):
        assert key in metrics, f"Missing metric key '{key}' for {model_name}"


# ===========================================================================
# GROUP 12 — predict: output columns and shape
# ===========================================================================


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_predict_returns_correct_columns_and_shape(model_name):
    df = _make_df(n=20)
    pipeline, _ = train(df, estimator=create_model(model_name))
    result = predict(df, pipeline=pipeline)
    assert "pred_log1p_future_revenue" in result.columns
    assert "pred_future_revenue" in result.columns
    assert len(result) == 20
