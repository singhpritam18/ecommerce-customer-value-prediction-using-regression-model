"""Customer contracts, artifact preservation, and independent-process inference."""

import json
import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from ecommerce_customer_value_prediction.models.predict import (
    OUTPUT_COLUMNS,
    load_model,
    predict,
    predict_csv,
)
from ecommerce_customer_value_prediction.models.train import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COL,
    create_model,
    train,
)
from ecommerce_customer_value_prediction.models.tune import PARAM_GRIDS
from ecommerce_customer_value_prediction.pipelines import final_model


@pytest.fixture
def customers():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "Customer ID": np.arange(60),
            "recency_days": rng.integers(1, 100, 60),
            "frequency": rng.integers(1, 20, 60),
            "monetary": rng.uniform(100, 2000, 60),
            "avg_basket_value": rng.uniform(10, 100, 60),
            "unique_products": rng.integers(1, 20, 60),
            "tenure_days": rng.integers(1, 500, 60),
            "purchase_rate": rng.uniform(0, 1, 60),
            "country": rng.choice(["United Kingdom", "Germany"], 60),
            "future_revenue": np.where(np.arange(60) % 2, rng.uniform(50, 2000, 60), 0),
        }
    )
    df[TARGET_COL] = np.log1p(df["future_revenue"])
    return df


@pytest.fixture
def fitted(customers):
    return train(customers, estimator=create_model("ridge", alpha=10.0))[0]


def test_one_and_multiple_rows_preserve_input(customers, fitted):
    for rows in (customers.iloc[[0]], customers.iloc[[0, 1, 2]]):
        result = predict(rows, pipeline=fitted)
        pd.testing.assert_frame_equal(result[rows.columns], rows)
        assert list(result.columns[-2:]) == OUTPUT_COLUMNS
        np.testing.assert_allclose(
            result["pred_future_revenue"], np.expm1(result["pred_log1p_future_revenue"])
        )


@pytest.mark.parametrize("feature", NUMERIC_FEATURES + CATEGORICAL_FEATURES)
def test_missing_feature_rejected(customers, fitted, feature):
    with pytest.raises(ValueError, match="missing feature"):
        predict(customers.drop(columns=feature), pipeline=fitted)


def test_extra_feature_ignored_unknown_country_supported(customers, fitted):
    baseline = predict(customers, pipeline=fitted)
    extra = customers.assign(unused_feature="not a predictor")
    result = predict(extra, pipeline=fitted)
    np.testing.assert_allclose(result[OUTPUT_COLUMNS], baseline[OUTPUT_COLUMNS])
    unknown = customers.copy()
    unknown["country"] = "Unseen Country"
    assert np.isfinite(predict(unknown, pipeline=fitted)[OUTPUT_COLUMNS]).all().all()


@pytest.mark.parametrize("value", ["30", "not numeric", True, 1 + 2j])
def test_incorrect_numeric_types_rejected(customers, fitted, value):
    invalid = customers.copy()
    invalid["recency_days"] = value
    with pytest.raises(TypeError, match="real numeric"):
        predict(invalid, pipeline=fitted)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_numeric_rejected(customers, fitted, value):
    invalid = customers.copy()
    invalid["monetary"] = value
    with pytest.raises(ValueError, match="non-finite"):
        predict(invalid, pipeline=fitted)


@pytest.mark.parametrize("value", [None, "", "   ", 123])
def test_invalid_country_rejected(customers, fitted, value):
    invalid = customers.copy()
    invalid["country"] = value
    with pytest.raises((ValueError, TypeError), match="country"):
        predict(invalid, pipeline=fitted)


def test_empty_duplicate_columns_and_nullable_numbers(customers, fitted):
    with pytest.raises(ValueError, match="at least one"):
        predict(customers.iloc[:0], pipeline=fitted)
    duplicate = pd.concat([customers, customers[["country"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate"):
        predict(duplicate, pipeline=fitted)
    nullable = customers.copy()
    nullable["frequency"] = nullable["frequency"].astype("Int64")
    pd.testing.assert_series_equal(
        predict(nullable, pipeline=fitted)["frequency"], nullable["frequency"]
    )
    nullable.loc[0, "frequency"] = pd.NA
    with pytest.raises(ValueError, match="missing"):
        predict(nullable, pipeline=fitted)


def test_batch_csv_roundtrip_and_no_overwrite(tmp_path, customers, fitted):
    model = tmp_path / "model.joblib"
    joblib.dump(fitted, model)
    source = tmp_path / "input.csv"
    output = tmp_path / "output.csv"
    customers.to_csv(source, index=False)
    result = predict_csv(source, output, model)
    np.testing.assert_allclose(
        pd.read_csv(output)[OUTPUT_COLUMNS], result[OUTPUT_COLUMNS], rtol=1e-12
    )
    with pytest.raises(FileExistsError):
        predict_csv(source, output, model)


def test_save_terminate_reload_matches_and_never_fits(tmp_path, customers):
    source = tmp_path / "customers.csv"
    model = tmp_path / "model.joblib"
    before = tmp_path / "before.npy"
    after = tmp_path / "after.npy"
    customers.to_csv(source, index=False)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    fit_script = """
import joblib, numpy as np, pandas as pd, sys
from ecommerce_customer_value_prediction.models.train import train, create_model
from ecommerce_customer_value_prediction.models.predict import predict, OUTPUT_COLUMNS
df = pd.read_csv(sys.argv[1])
pipeline, _ = train(df, estimator=create_model('ridge', alpha=10.0))
np.save(sys.argv[3], predict(df, pipeline=pipeline)[OUTPUT_COLUMNS].to_numpy())
joblib.dump(pipeline, sys.argv[2])
"""
    reload_script = """
import numpy as np, pandas as pd, sys
import ecommerce_customer_value_prediction
from sklearn.pipeline import Pipeline
from ecommerce_customer_value_prediction.models.predict import load_model, predict, OUTPUT_COLUMNS
def forbidden(*args, **kwargs):
    raise AssertionError('Inference triggered training')
Pipeline.fit = forbidden
df = pd.read_csv(sys.argv[1])
result = predict(df, pipeline=load_model(sys.argv[2]))
assert list(result.columns[-2:]) == OUTPUT_COLUMNS
assert len(result) == len(df)
np.save(sys.argv[3], result[OUTPUT_COLUMNS].to_numpy())
"""
    for script, result in ((fit_script, before), (reload_script, after)):
        completed = subprocess.run(
            [sys.executable, "-c", script, str(source), str(model), str(result)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
    np.testing.assert_allclose(np.load(before), np.load(after), rtol=1e-12, atol=1e-12)


def test_experiment_locks_before_holdout_and_package_never_retunes(
    tmp_path, customers, monkeypatch
):
    for name in ("ridge", "lasso", "elasticnet"):
        grid = {"regressor__alpha": [0.1, 1.0]}
        if name == "elasticnet":
            grid["regressor__l1_ratio"] = [0.5]
        monkeypatch.setitem(PARAM_GRIDS, name, grid)
    source = tmp_path / "customer_dataset.csv"
    customers.to_csv(source, index=False)
    directory = tmp_path / "experiment"
    original_evaluate = final_model.evaluate
    calls = []

    def evaluate_once(pipeline, test_df, save):
        assert (directory / "selection.json").exists()
        name = type(pipeline.named_steps["regressor"]).__name__
        assert name not in calls
        calls.append(name)
        return original_evaluate(pipeline, test_df, save=save)

    monkeypatch.setattr(final_model, "evaluate", evaluate_once)
    lock = final_model.run_experiment(source, directory)
    assert len(calls) == 4
    assert lock["selected_model"] == min(
        lock["candidates"], key=lambda name: lock["candidates"][name]["selection_cv_rmse_log"]
    )
    with pytest.raises(FileExistsError):
        final_model.run_experiment(source, directory)

    def forbidden(*args, **kwargs):
        raise AssertionError("Packaging must not tune or evaluate a test partition")

    monkeypatch.setattr(final_model, "evaluate", forbidden)
    monkeypatch.setattr(final_model, "run_cv", forbidden)
    monkeypatch.setattr(final_model, "run_grid_search", forbidden)
    monkeypatch.setattr(final_model, "MODELS_DIR", tmp_path)
    model = tmp_path / "customer_value_model.joblib"
    metadata_path = tmp_path / "model_metadata.json"
    metadata = final_model.package_selected_model(source, directory, model, metadata_path)
    pipeline = load_model(model)
    assert pipeline.named_steps["regressor"].get_params() == metadata["hyperparameters"]
    assert metadata["n_training_customers"] == len(customers)
    original_hash = final_model.file_hash(model)
    final_model.package_selected_model(source, directory, model, metadata_path)
    archived_models = list((tmp_path / "archive").glob("*/*.joblib"))
    assert len(archived_models) == 1
    assert final_model.file_hash(archived_models[0]) == original_hash
    assert list((tmp_path / "archive").glob("*/model_metadata.json"))
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="Dataset differs"):
        final_model.package_selected_model(source, directory, model, metadata_path)
    stored = json.loads(metadata_path.read_text())
    assert stored["model_sha256"] == final_model.file_hash(model)


def test_shipped_model_new_user_and_batch_command(tmp_path):
    root = Path(__file__).resolve().parents[1]
    model = root / "models" / "customer_value_model.joblib"
    metadata = root / "models" / "model_metadata.json"
    example = root / "examples" / "customer_features.csv"
    assert model.exists() and metadata.exists()
    original_hashes = [final_model.file_hash(path) for path in (model, metadata)]
    output = tmp_path / "predictions.csv"
    script = """
import json, numpy as np, pandas as pd, sys
import ecommerce_customer_value_prediction
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from ecommerce_customer_value_prediction.models.predict import load_model, predict, OUTPUT_COLUMNS
def forbidden(*args, **kwargs):
    raise AssertionError('Training is forbidden in a consumer process')
Pipeline.fit = forbidden
Ridge.fit = forbidden
pipeline = load_model()
df = pd.read_csv(sys.argv[1])
result = predict(df, pipeline=pipeline)
assert list(result.columns) == list(df.columns) + OUTPUT_COLUMNS
assert np.isfinite(result[OUTPUT_COLUMNS]).all().all()
assert len(result) == 2
expected = pd.read_csv(sys.argv[2])
np.testing.assert_allclose(result[OUTPUT_COLUMNS], expected[OUTPUT_COLUMNS], rtol=1e-12, atol=1e-12)
print('NEW USER: import/load/predict/schema/no-training PASS')
"""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(example),
            str(root / "examples" / "customer_predictions.csv"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    completed = subprocess.run(
        [
            "predict-customer-value",
            "--input",
            str(example),
            "--output",
            str(output),
            "--model",
            str(model),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(pd.read_csv(output).columns[-2:]) == OUTPUT_COLUMNS
    assert original_hashes == [final_model.file_hash(path) for path in (model, metadata)]


def test_packaged_checksum_and_version_rejected(tmp_path):
    root = Path(__file__).resolve().parents[1]
    model = tmp_path / "customer_value_model.joblib"
    model.write_bytes((root / "models" / model.name).read_bytes())
    metadata_path = tmp_path / "model_metadata.json"
    metadata = json.loads((root / "models" / metadata_path.name).read_text())
    metadata["sklearn_version"] = "wrong-version"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="sklearn version"):
        load_model(model)
    metadata["sklearn_version"] = final_model.sklearn.__version__
    metadata["model_sha256"] = "bad-checksum"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="checksum"):
        load_model(model)
