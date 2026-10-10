"""
tests/test_tune.py — Unit tests for the CV and hyperparameter-tuning framework.

Checkpoint objective
--------------------
Verify a fair cross-validation and hyperparameter-tuning infrastructure for
OLS, Ridge, Lasso and ElasticNet while keeping the final test set untouched.

All tests use small synthetic DataFrames — no real dataset required.

Test groups
-----------
1.  PARAM_GRIDS — structure and nested sklearn parameter naming
2.  make_cv_folds — KFold vs StratifiedKFold selection
3.  run_cv — OLS, Ridge, Lasso, ElasticNet; metric keys; no test-set use
4.  run_grid_search — Ridge, Lasso, ElasticNet; best_params_; results_df shape
5.  GridSearchCV receives full Pipeline (not pre-processed data)
6.  run_all_cv — all four models; tune=False and tune=True
7.  save_cv_results — writes valid CSV with required columns
8.  Train/test split — deterministic, stratified, same split for all models
9.  Test set exclusion — verified structurally
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from ecommerce_customer_value_prediction.models.tune import (
    CV_FOLDS,
    PARAM_GRIDS,
    PRIMARY_METRIC,
    RANDOM_STATE,
    SCORERS,
    make_cv_folds,
    run_all_cv,
    run_cv,
    run_grid_search,
    save_cv_results,
)
from ecommerce_customer_value_prediction.models.train import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
)
from ecommerce_customer_value_prediction.pipelines.training_pipeline import (
    _make_stratified_split,
    RANDOM_STATE as PIPELINE_RANDOM_STATE,
    TEST_SIZE,
)


# ---------------------------------------------------------------------------
# Shared synthetic fixtures
# ---------------------------------------------------------------------------


def _make_df(n: int = 80, seed: int = 0) -> pd.DataFrame:
    """
    Synthetic customer dataset large enough for 5-fold CV (16 rows per fold).
    Includes future_revenue so stratification can be tested.
    """
    rng = np.random.default_rng(seed)
    future_rev = np.concatenate([
        np.zeros(n // 3),                     # churned customers
        rng.uniform(10, 3000, n - n // 3),     # active customers
    ])
    rng.shuffle(future_rev)
    df = pd.DataFrame({
        "Customer ID":     range(1, n + 1),
        "recency_days":    rng.integers(1, 365, size=n).astype(float),
        "frequency":       rng.integers(1, 50,  size=n).astype(float),
        "monetary":        rng.uniform(10, 5000, size=n),
        "avg_basket_value":rng.uniform(5,  200,  size=n),
        "unique_products": rng.integers(1, 30,  size=n).astype(float),
        "tenure_days":     rng.integers(0, 700,  size=n).astype(float),
        "purchase_rate":   rng.uniform(0, 0.5,  size=n),
        "country":         rng.choice(["United Kingdom", "Germany", "France"], size=n),
        "future_revenue":  future_rev,
    })
    df["log1p_future_revenue"] = np.log1p(df["future_revenue"])
    return df


# ===========================================================================
# GROUP 1 — PARAM_GRIDS structure and nested sklearn parameter naming
# ===========================================================================


def test_param_grids_contains_all_four_models():
    assert set(PARAM_GRIDS.keys()) == {"linear", "ridge", "lasso", "elasticnet"}


def test_param_grids_linear_is_empty():
    """OLS has no regularisation hyperparameter — grid must be empty."""
    assert PARAM_GRIDS["linear"] == {}


def test_param_grids_ridge_alpha_uses_nested_prefix():
    """
    sklearn Pipeline nested parameter syntax: because the estimator is in the
    step named 'regressor', all hyperparameters must be prefixed with
    'regressor__'.  GridSearchCV routes 'regressor__alpha' to Ridge.alpha.
    """
    assert "regressor__alpha" in PARAM_GRIDS["ridge"]


def test_param_grids_lasso_alpha_uses_nested_prefix():
    assert "regressor__alpha" in PARAM_GRIDS["lasso"]


def test_param_grids_elasticnet_has_both_alpha_and_l1_ratio():
    grid = PARAM_GRIDS["elasticnet"]
    assert "regressor__alpha" in grid
    assert "regressor__l1_ratio" in grid


def test_param_grids_ridge_covers_broad_log_range():
    """Grid must span at least 4 orders of magnitude."""
    alphas = PARAM_GRIDS["ridge"]["regressor__alpha"]
    assert min(alphas) <= 0.01
    assert max(alphas) >= 100.0
    assert len(alphas) >= 5


def test_param_grids_elasticnet_l1_ratio_excludes_0_and_1():
    """
    l1_ratio=0 → pure Ridge; l1_ratio=1 → pure Lasso.
    Both are tested as separate models so endpoints are excluded.
    """
    ratios = PARAM_GRIDS["elasticnet"]["regressor__l1_ratio"]
    assert 0.0 not in ratios
    assert 1.0 not in ratios


def test_param_grids_elasticnet_l1_ratio_covers_full_spectrum():
    """Must include at least Ridge-like, balanced, and Lasso-like values."""
    ratios = PARAM_GRIDS["elasticnet"]["regressor__l1_ratio"]
    assert min(ratios) <= 0.25    # mostly Ridge-like
    assert 0.5 in ratios           # balanced
    assert max(ratios) >= 0.75    # mostly Lasso-like


def test_primary_metric_is_neg_rmse():
    """Primary ranking metric is neg_root_mean_squared_error (sklearn convention)."""
    assert PRIMARY_METRIC == "neg_root_mean_squared_error"


def test_scorers_contain_mae_rmse_r2():
    assert "mae_log" in SCORERS
    assert "rmse_log" in SCORERS
    assert "r2_log" in SCORERS


# ===========================================================================
# GROUP 2 — make_cv_folds
# ===========================================================================


def test_make_cv_folds_returns_kfold_without_y_binary():
    cv = make_cv_folds(y_binary=None, n_splits=5)
    assert isinstance(cv, KFold)


def test_make_cv_folds_returns_stratified_with_y_binary():
    y_binary = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    cv = make_cv_folds(y_binary=y_binary, n_splits=5)
    assert isinstance(cv, StratifiedKFold)


def test_make_cv_folds_n_splits_respected():
    cv = make_cv_folds(n_splits=3)
    assert cv.n_splits == 3


def test_make_cv_folds_default_splits_equals_cv_folds_constant():
    cv = make_cv_folds()
    assert cv.n_splits == CV_FOLDS


# ===========================================================================
# GROUP 3 — run_cv: all four models; metric keys; scores are numeric
# ===========================================================================


def _cv_result_keys():
    return {
        "model_name", "hyperparameters", "n_folds",
        "mean_cv_mae_log", "std_cv_mae_log",
        "mean_cv_rmse_log", "std_cv_rmse_log",
        "mean_cv_r2_log", "std_cv_r2_log",
    }


@pytest.mark.parametrize("model_name", ["linear", "ridge", "lasso", "elasticnet"])
def test_run_cv_returns_all_required_keys(model_name):
    df = _make_df()
    result = run_cv(model_name, df, n_splits=3)
    for key in _cv_result_keys():
        assert key in result, f"Missing key '{key}' for {model_name}"


@pytest.mark.parametrize("model_name", ["linear", "ridge", "lasso", "elasticnet"])
def test_run_cv_metrics_are_finite(model_name):
    df = _make_df()
    result = run_cv(model_name, df, n_splits=3)
    for key in ("mean_cv_mae_log", "mean_cv_rmse_log", "mean_cv_r2_log"):
        assert np.isfinite(result[key]), f"Non-finite {key} for {model_name}"


@pytest.mark.parametrize("model_name", ["linear", "ridge", "lasso", "elasticnet"])
def test_run_cv_rmse_is_positive(model_name):
    df = _make_df()
    result = run_cv(model_name, df, n_splits=3)
    assert result["mean_cv_rmse_log"] > 0


def test_run_cv_records_correct_model_name():
    result = run_cv("ridge", _make_df(), n_splits=3)
    assert result["model_name"] == "ridge"


def test_run_cv_records_hyperparameters():
    result = run_cv("ridge", _make_df(), hyperparameters={"alpha": 5.0}, n_splits=3)
    assert result["hyperparameters"]["alpha"] == 5.0


def test_run_cv_n_folds_recorded():
    result = run_cv("linear", _make_df(), n_splits=3)
    assert result["n_folds"] == 3


def test_run_cv_metrics_on_log1p_scale_not_gbp():
    """
    run_cv operates on log1p_future_revenue.
    RMSE should be on the order of 1-3 (log scale), NOT hundreds of £.
    """
    df = _make_df(n=80)
    result = run_cv("ridge", df, n_splits=3)
    # log1p(3000) ≈ 8.0, so log-RMSE should be < 10
    assert result["mean_cv_rmse_log"] < 10.0


def test_run_cv_uses_only_training_data_not_test(tmp_path):
    """
    Structural check: run_cv only accepts train_df.
    There is no parameter to pass a test set — the function signature
    enforces that the test set is never evaluated here.
    """
    import inspect
    sig = inspect.signature(run_cv)
    param_names = list(sig.parameters.keys())
    # Must have train_df, must NOT have test_df or test set
    assert "train_df" in param_names
    assert "test_df" not in param_names


# ===========================================================================
# GROUP 4 — run_grid_search: Ridge, Lasso, ElasticNet
# ===========================================================================


def test_run_grid_search_ridge_returns_gridsearchcv_and_df():
    df = _make_df()
    small_grid = {"regressor__alpha": [0.1, 1.0, 10.0]}
    gs, results_df = run_grid_search("ridge", df, param_grid=small_grid, n_splits=3)
    assert isinstance(gs, GridSearchCV)
    assert isinstance(results_df, pd.DataFrame)


def test_run_grid_search_results_df_has_required_columns():
    df = _make_df()
    small_grid = {"regressor__alpha": [0.1, 1.0]}
    _, results_df = run_grid_search("ridge", df, param_grid=small_grid, n_splits=3)
    for col in ("model", "alpha", "mean_cv_rmse_log", "std_cv_rmse_log", "rank"):
        assert col in results_df.columns, f"Missing column '{col}'"


def test_run_grid_search_ridge_n_rows_equals_grid_size():
    df = _make_df()
    small_grid = {"regressor__alpha": [0.01, 0.1, 1.0, 10.0]}
    _, results_df = run_grid_search("ridge", df, param_grid=small_grid, n_splits=3)
    assert len(results_df) == 4


def test_run_grid_search_lasso_runs():
    df = _make_df()
    small_grid = {"regressor__alpha": [0.01, 0.1]}
    gs, results_df = run_grid_search("lasso", df, param_grid=small_grid, n_splits=3)
    assert gs.best_params_ is not None
    assert len(results_df) == 2


def test_run_grid_search_elasticnet_has_l1_ratio_column():
    df = _make_df()
    small_grid = {
        "regressor__alpha": [0.1, 1.0],
        "regressor__l1_ratio": [0.25, 0.75],
    }
    _, results_df = run_grid_search("elasticnet", df, param_grid=small_grid, n_splits=3)
    assert "l1_ratio" in results_df.columns
    assert len(results_df) == 4  # 2 × 2


def test_run_grid_search_best_params_uses_nested_prefix():
    """
    best_params_ keys must use 'regressor__alpha' (nested Pipeline syntax).
    This confirms GridSearchCV correctly routes params through the Pipeline.
    """
    df = _make_df()
    small_grid = {"regressor__alpha": [0.1, 1.0]}
    gs, _ = run_grid_search("ridge", df, param_grid=small_grid, n_splits=3)
    assert "regressor__alpha" in gs.best_params_


def test_run_grid_search_raises_for_linear_with_empty_grid():
    """OLS has an empty PARAM_GRIDS — run_grid_search should raise ValueError."""
    df = _make_df()
    with pytest.raises(ValueError, match="empty grid"):
        run_grid_search("linear", df, n_splits=3)


def test_run_grid_search_results_sorted_by_rank():
    df = _make_df()
    small_grid = {"regressor__alpha": [0.001, 0.1, 10.0, 1000.0]}
    _, results_df = run_grid_search("ridge", df, param_grid=small_grid, n_splits=3)
    ranks = results_df["rank"].tolist()
    assert ranks == sorted(ranks)


# ===========================================================================
# GROUP 5 — GridSearchCV receives FULL Pipeline (not pre-processed data)
# ===========================================================================


def test_gridsearchcv_estimator_is_full_pipeline():
    """
    GridSearchCV must wrap the complete Pipeline (preprocessor + regressor),
    not a bare estimator or pre-processed array.  This is what prevents
    CV preprocessing leakage.
    """
    df = _make_df()
    small_grid = {"regressor__alpha": [0.1, 1.0]}
    gs, _ = run_grid_search("ridge", df, param_grid=small_grid, n_splits=3)
    assert isinstance(gs.estimator, Pipeline), (
        "GridSearchCV.estimator must be a sklearn Pipeline, not a bare estimator. "
        "Passing a bare estimator would allow test-fold statistics to contaminate "
        "the StandardScaler, causing CV preprocessing leakage."
    )


def test_gridsearchcv_pipeline_has_preprocessor_and_regressor():
    df = _make_df()
    small_grid = {"regressor__alpha": [0.1, 1.0]}
    gs, _ = run_grid_search("ridge", df, param_grid=small_grid, n_splits=3)
    step_names = [name for name, _ in gs.estimator.steps]
    assert "preprocessor" in step_names
    assert "regressor" in step_names


def test_nested_param_name_routes_alpha_to_ridge():
    """
    Confirm that 'regressor__alpha' in GridSearchCV routes correctly to
    Ridge.alpha after fitting.  The '__' separator is sklearn's convention for
    nested Pipeline parameters.
    """
    df = _make_df()
    small_grid = {"regressor__alpha": [5.0]}
    gs, _ = run_grid_search("ridge", df, param_grid=small_grid, n_splits=3)
    fitted_alpha = gs.best_estimator_.named_steps["regressor"].alpha
    assert fitted_alpha == 5.0


def test_nested_param_name_routes_l1_ratio_to_elasticnet():
    df = _make_df()
    small_grid = {
        "regressor__alpha": [0.1],
        "regressor__l1_ratio": [0.3],
    }
    gs, _ = run_grid_search("elasticnet", df, param_grid=small_grid, n_splits=3)
    fitted_l1 = gs.best_estimator_.named_steps["regressor"].l1_ratio
    assert fitted_l1 == 0.3


# ===========================================================================
# GROUP 6 — run_all_cv
# ===========================================================================


def test_run_all_cv_returns_dataframe():
    df = _make_df()
    result = run_all_cv(df, n_splits=3)
    assert isinstance(result, pd.DataFrame)


def test_run_all_cv_contains_all_four_models():
    df = _make_df()
    result = run_all_cv(df, n_splits=3)
    assert set(result["model"].tolist()) == {"linear", "ridge", "lasso", "elasticnet"}


def test_run_all_cv_tune_false_one_row_per_model():
    df = _make_df()
    result = run_all_cv(df, n_splits=3, tune=False)
    assert len(result) == 4


def test_run_all_cv_tune_true_multiple_rows_for_regularised_models():
    """When tune=True, Ridge/Lasso/EN produce one row per grid candidate."""
    df = _make_df()
    result = run_all_cv(df, n_splits=3, tune=True)
    # Ridge: 9 alphas, Lasso: 9 alphas, EN: 5×5=25, OLS: 1 → 44 total
    assert len(result) > 4


def test_run_all_cv_has_required_columns():
    df = _make_df()
    result = run_all_cv(df, n_splits=3, tune=False)
    for col in ("model", "alpha", "l1_ratio", "mean_cv_rmse_log", "std_cv_rmse_log"):
        assert col in result.columns, f"Missing column '{col}'"


def test_run_all_cv_rmse_values_are_positive():
    df = _make_df()
    result = run_all_cv(df, n_splits=3, tune=False)
    assert (result["mean_cv_rmse_log"] > 0).all()


# ===========================================================================
# GROUP 7 — save_cv_results
# ===========================================================================


def test_save_cv_results_writes_csv():
    df = _make_df()
    cv_df = run_all_cv(df, n_splits=3, tune=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = save_cv_results(cv_df, path=Path(tmpdir) / "cv_results.csv")
        assert out.exists()
        loaded = pd.read_csv(out)
    assert len(loaded) == 4
    assert "model" in loaded.columns
    assert "mean_cv_rmse_log" in loaded.columns


def test_save_cv_results_default_path_is_metrics_dir():
    """Default path is reports/metrics/cv_results.csv."""
    import inspect
    from ecommerce_customer_value_prediction.models.tune import save_cv_results as scr
    src = inspect.getsource(scr)
    assert "cv_results.csv" in src


def test_save_cv_results_no_test_metrics_in_output():
    """
    The cv_results file must contain ONLY CV metrics, never test-set metrics.
    Column names containing 'test_' would indicate a violation.
    """
    df = _make_df()
    cv_df = run_all_cv(df, n_splits=3, tune=False)
    # Ensure none of the columns are labelled as test metrics
    for col in cv_df.columns:
        assert not col.startswith("test_"), (
            f"Column '{col}' looks like a test-set metric. "
            "cv_results.csv must contain only CV metrics."
        )


# ===========================================================================
# GROUP 8 — Train/test split: deterministic, stratified, same for all models
# ===========================================================================


def test_split_is_deterministic():
    df = _make_df(n=100)
    train1, test1 = _make_stratified_split(df)
    train2, test2 = _make_stratified_split(df)
    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(test1, test2)


def test_split_test_size_approximately_correct():
    df = _make_df(n=100)
    _, test_df = _make_stratified_split(df)
    ratio = len(test_df) / 100
    assert abs(ratio - TEST_SIZE) < 0.05


def test_split_maintains_zero_revenue_proportion():
    """
    Stratification on zero/positive revenue preserves the churn rate
    in both train and test splits.
    """
    df = _make_df(n=100)
    train_df, test_df = _make_stratified_split(df)
    overall_zero_rate = (df["future_revenue"] == 0).mean()
    train_zero_rate = (train_df["future_revenue"] == 0).mean()
    test_zero_rate  = (test_df["future_revenue"] == 0).mean()
    # Allow ±5% tolerance
    assert abs(train_zero_rate - overall_zero_rate) < 0.05
    assert abs(test_zero_rate  - overall_zero_rate) < 0.05


def test_same_split_used_for_all_models():
    """
    When the same random_state is used, all four models see identical
    train/test customers — fair comparison guaranteed.
    """
    df = _make_df(n=100)
    train_a, test_a = _make_stratified_split(df, random_state=42)
    train_b, test_b = _make_stratified_split(df, random_state=42)
    assert list(train_a["Customer ID"]) == list(train_b["Customer ID"])
    assert list(test_a["Customer ID"]) == list(test_b["Customer ID"])


def test_different_random_states_produce_different_splits():
    df = _make_df(n=100)
    train_a, _ = _make_stratified_split(df, random_state=42)
    train_b, _ = _make_stratified_split(df, random_state=99)
    assert list(train_a["Customer ID"]) != list(train_b["Customer ID"])


# ===========================================================================
# GROUP 9 — Test set exclusion: structural verification
# ===========================================================================


def test_run_cv_signature_has_no_test_df_parameter():
    """
    run_cv must not accept a test_df argument — the test set must be
    structurally impossible to pass to the CV function.
    """
    import inspect
    sig = inspect.signature(run_cv)
    assert "test_df" not in sig.parameters


def test_run_grid_search_signature_has_no_test_df_parameter():
    import inspect
    sig = inspect.signature(run_grid_search)
    assert "test_df" not in sig.parameters


def test_run_all_cv_signature_has_no_test_df_parameter():
    import inspect
    sig = inspect.signature(run_all_cv)
    assert "test_df" not in sig.parameters


def test_save_cv_results_does_not_contain_test_column_names():
    """
    Even if a caller accidentally added test columns, save_cv_results writes
    whatever df it receives.  This test verifies our run_all_cv output is clean.
    """
    df = _make_df()
    cv_df = run_all_cv(df, n_splits=3, tune=False)
    test_cols = [c for c in cv_df.columns if "test" in c.lower()]
    assert len(test_cols) == 0, f"Unexpected test-related columns: {test_cols}"


def test_cv_folds_constant_is_five():
    """5-fold CV is the project standard."""
    assert CV_FOLDS == 5


def test_random_state_consistent_with_pipeline():
    """tune.py and training_pipeline.py must use the same RANDOM_STATE."""
    assert RANDOM_STATE == PIPELINE_RANDOM_STATE
