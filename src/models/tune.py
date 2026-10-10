"""
tune.py — Cross-validation and hyperparameter-tuning framework.

Responsibility
--------------
Provide reusable, model-agnostic cross-validation and GridSearchCV utilities
for OLS, Ridge, Lasso, and ElasticNet.

Design principles
-----------------
* The FINAL TEST SET is never touched here.  All search and evaluation in
  this module operates exclusively on the training subset passed by the caller.
* GridSearchCV always receives the COMPLETE Pipeline (preprocessing + estimator),
  not pre-processed data.  This guarantees StandardScaler and OneHotEncoder are
  re-fitted independently inside every training fold — no CV preprocessing leakage.
* Hyperparameter grids are defined as module-level constants so they can be
  imported by pipelines, notebooks, and tests without duplication.
* Metrics are always on log1p_future_revenue (the model target).  The docstrings
  are explicit about this so callers do not mislabel log-RMSE as £ RMSE.

Public API
----------
PARAM_GRIDS          : dict — centralised search grids for all models
run_cv               : cross-validate a pipeline on training data
run_grid_search      : GridSearchCV wrapper for a named model
run_all_cv           : cross-validate all four models and return a summary DataFrame
save_cv_results      : persist the summary DataFrame to reports/metrics/cv_results.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_validate

from ecommerce_customer_value_prediction.config import METRICS_DIR
from ecommerce_customer_value_prediction.models.train import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SUPPORTED_MODELS,
    TARGET_COL,
    build_pipeline,
    create_model,
)

# ---------------------------------------------------------------------------
# CV constants
# ---------------------------------------------------------------------------

CV_FOLDS: int = 5
RANDOM_STATE: int = 42

# Primary metric used to rank models and select best hyperparameters.
# Measured on log1p_future_revenue — the modelled target.
# Do NOT interpret this as £ RMSE.
PRIMARY_METRIC: str = "neg_root_mean_squared_error"  # sklearn convention

# ---------------------------------------------------------------------------
# Centralised hyperparameter grids
# ---------------------------------------------------------------------------

# NOTE: All grids are expressed using sklearn nested parameter syntax.
# Because the estimator lives in the Pipeline step named "regressor",
# hyperparameters are addressed as  regressor__alpha,  regressor__l1_ratio.
# GridSearchCV automatically routes these through Pipeline.set_params().

PARAM_GRIDS: dict[str, dict[str, list]] = {
    # OLS has no regularisation hyperparameter.
    # An empty grid causes GridSearchCV to evaluate a single candidate
    # (the untuned model), which equals a plain CV run.
    "linear": {},
    # Ridge: search alpha over a broad log scale.
    # 10^-4 → 10^4 covers everything from nearly-OLS to very strong shrinkage.
    "ridge": {
        "regressor__alpha": [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 10_000.0],
    },
    # Lasso: coordinate-descent solver; restrict grid to moderate alphas.
    # Very large alpha (>10) zeros all coefficients on this dataset.
    # Very small alpha (<1e-5) behaves like OLS and may not converge.
    "lasso": {
        "regressor__alpha": [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5, 1.0],
    },
    # ElasticNet: search alpha AND l1_ratio.
    # l1_ratio=0 → pure Ridge; l1_ratio=1 → pure Lasso.
    # We exclude exact endpoints to avoid duplicating Ridge and Lasso experiments.
    "elasticnet": {
        "regressor__alpha": [1e-4, 1e-3, 0.01, 0.1, 1.0],
        "regressor__l1_ratio": [0.1, 0.25, 0.5, 0.75, 0.9],
    },
}

# ---------------------------------------------------------------------------
# Scorer definitions
# NOTE: All scorers operate on log1p_future_revenue predictions.
#       Label them clearly — do not call the result "£ RMSE".
# ---------------------------------------------------------------------------


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


SCORERS: dict[str, Any] = {
    "mae_log": make_scorer(mean_absolute_error),
    "rmse_log": make_scorer(_rmse),
    "r2_log": make_scorer(r2_score),
}

# ---------------------------------------------------------------------------
# Fold strategy
# ---------------------------------------------------------------------------


def make_cv_folds(
    y_binary: pd.Series | None = None,
    n_splits: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> KFold | StratifiedKFold:
    """
    Return a cross-validation fold object.

    If ``y_binary`` is supplied (a Series of 0/1 indicating zero vs positive
    future revenue), StratifiedKFold is used to keep the churn ratio
    consistent across folds.  ``y_binary`` is used ONLY for fold construction —
    it is never used as a model predictor.

    If ``y_binary`` is None, plain KFold is returned.

    Parameters
    ----------
    y_binary : pd.Series or None
        Binary 0/1 label: 1 = customer had future revenue > 0.
        Used for stratified fold construction only.
    n_splits : int
    random_state : int

    Returns
    -------
    StratifiedKFold if y_binary is provided, else KFold.
    """
    if y_binary is not None:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


# ---------------------------------------------------------------------------
# Core CV function
# ---------------------------------------------------------------------------


def run_cv(
    model_name: str,
    train_df: pd.DataFrame,
    hyperparameters: dict | None = None,
    n_splits: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
    use_stratified: bool = True,
) -> dict[str, float]:
    """
    Cross-validate a model on the training set.

    The test set is NOT passed to this function and is never used.

    All metrics are on log1p_future_revenue (the model target).
    Do NOT label results as £ metrics.

    Parameters
    ----------
    model_name : str
        One of "linear", "ridge", "lasso", "elasticnet".
    train_df : pd.DataFrame
        Training customers only.  Must include all NUMERIC_FEATURES,
        CATEGORICAL_FEATURES, and TARGET_COL.
    hyperparameters : dict or None
        Passed to ``create_model``.  If None, uses estimator defaults.
    n_splits : int
        Number of CV folds.
    random_state : int
    use_stratified : bool
        If True, use StratifiedKFold stratified on zero/positive revenue.

    Returns
    -------
    dict with keys:
        mean_cv_mae_log, std_cv_mae_log,
        mean_cv_rmse_log, std_cv_rmse_log,
        mean_cv_r2_log, std_cv_r2_log,
        n_folds, model_name, hyperparameters
    """
    hparams = hyperparameters or {}
    estimator = create_model(model_name, **hparams)
    pipeline = build_pipeline(estimator=estimator)

    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df[TARGET_COL]

    # Build stratification label (zero vs positive revenue)
    y_binary = None
    if use_stratified and "future_revenue" in train_df.columns:
        y_binary = (train_df["future_revenue"] > 0).astype(int)

    cv = make_cv_folds(y_binary=y_binary, n_splits=n_splits, random_state=random_state)

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv if y_binary is None else list(cv.split(X_train, y_binary)),
        scoring=SCORERS,
        error_score="raise",
    )

    result = {
        "model_name": model_name,
        "hyperparameters": hparams,
        "n_folds": n_splits,
        # All metrics are on log1p_future_revenue
        "mean_cv_mae_log": round(float(scores["test_mae_log"].mean()), 6),
        "std_cv_mae_log": round(float(scores["test_mae_log"].std()), 6),
        "mean_cv_rmse_log": round(float(scores["test_rmse_log"].mean()), 6),
        "std_cv_rmse_log": round(float(scores["test_rmse_log"].std()), 6),
        "mean_cv_r2_log": round(float(scores["test_r2_log"].mean()), 6),
        "std_cv_r2_log": round(float(scores["test_r2_log"].std()), 6),
    }

    logger.info(
        f"[run_cv] {model_name} | "
        f"CV R²={result['mean_cv_r2_log']:.4f}±{result['std_cv_r2_log']:.4f} | "
        f"CV RMSE_log={result['mean_cv_rmse_log']:.4f}  "
        f"(log1p scale — not £)"
    )
    return result


# ---------------------------------------------------------------------------
# GridSearchCV wrapper
# ---------------------------------------------------------------------------


def run_grid_search(
    model_name: str,
    train_df: pd.DataFrame,
    param_grid: dict | None = None,
    n_splits: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
    use_stratified: bool = True,
    verbose: int = 0,
    n_jobs: int = 1,
) -> tuple[GridSearchCV, pd.DataFrame]:
    """
    Run GridSearchCV for a named model on the training set only.

    The test set is never passed to this function.

    GridSearchCV receives the COMPLETE Pipeline (preprocessing + estimator),
    not pre-processed data.  This means StandardScaler and OneHotEncoder are
    fitted fresh on each fold's training data and applied (without refitting)
    to each fold's validation data.  Preprocessing leakage is structurally
    impossible.

    Why passing the full pipeline matters
    --------------------------------------
    If you scaled data before passing it to GridSearchCV, the scaler would have
    seen ALL training folds during its fit — including the current validation fold.
    The validation fold's statistics (mean, std) would contaminate the scaler,
    biasing CV scores optimistically.  Passing the pipeline to GridSearchCV
    ensures each fold re-fits the scaler from scratch on that fold's training data.

    Parameters
    ----------
    model_name : str
        One of "ridge", "lasso", "elasticnet".  For "linear", use run_cv().
    train_df : pd.DataFrame
        Training customers only.
    param_grid : dict or None
        sklearn nested param grid using "regressor__" prefix.
        If None, falls back to PARAM_GRIDS[model_name].
    n_splits : int
    random_state : int
    use_stratified : bool
    verbose : int
    n_jobs : int
        Parallel workers; defaults to one to bound resource use.

    Returns
    -------
    gs : fitted GridSearchCV object
    results_df : pd.DataFrame with one row per hyperparameter combination,
                 columns: model, alpha, l1_ratio,
                 mean_cv_mae_log, std_cv_mae_log,
                 mean_cv_rmse_log, std_cv_rmse_log,
                 mean_cv_r2_log, std_cv_r2_log, rank
    """
    grid = param_grid if param_grid is not None else PARAM_GRIDS.get(model_name, {})

    if not grid:
        raise ValueError(
            f"run_grid_search called for '{model_name}' which has an empty grid. "
            "Use run_cv() for OLS."
        )

    # Build base pipeline with default hyperparameters
    base_pipeline = build_pipeline(estimator=create_model(model_name))

    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df[TARGET_COL]

    y_binary = None
    if use_stratified and "future_revenue" in train_df.columns:
        y_binary = (train_df["future_revenue"] > 0).astype(int)

    cv = make_cv_folds(y_binary=y_binary, n_splits=n_splits, random_state=random_state)
    cv_obj = list(cv.split(X_train, y_binary)) if y_binary is not None else cv

    gs = GridSearchCV(
        estimator=base_pipeline,
        param_grid=grid,
        scoring=PRIMARY_METRIC,  # used to select best_params_
        cv=cv_obj,
        refit=True,  # refit best estimator on full training set
        n_jobs=n_jobs,
        verbose=verbose,
        error_score="raise",
    )

    gs.fit(X_train, y_train)

    logger.success(
        f"[run_grid_search] {model_name} | "
        f"Best params: {gs.best_params_} | "
        f"Best CV RMSE_log={-gs.best_score_:.4f}  (log1p scale — not £)"
    )

    # Build results DataFrame
    cv_results = gs.cv_results_
    rows = []
    n_candidates = len(cv_results["params"])
    for i in range(n_candidates):
        p = cv_results["params"][i]
        row = {
            "model": model_name,
            "alpha": p.get("regressor__alpha", None),
            "l1_ratio": p.get("regressor__l1_ratio", None),
            # mean_test_score for neg_root_mean_squared_error is negative
            "mean_cv_rmse_log": round(-float(cv_results["mean_test_score"][i]), 6),
            "std_cv_rmse_log": round(float(cv_results["std_test_score"][i]), 6),
            "rank": int(cv_results["rank_test_score"][i]),
        }
        rows.append(row)

    results_df = pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)
    return gs, results_df


# ---------------------------------------------------------------------------
# Run all four models and collect CV summary
# ---------------------------------------------------------------------------


def run_all_cv(
    train_df: pd.DataFrame,
    n_splits: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
    use_stratified: bool = True,
    tune: bool = False,
) -> pd.DataFrame:
    """
    Cross-validate (and optionally tune) all four models on training data.

    The test set is never used.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training customers only.
    n_splits, random_state, use_stratified : as above
    tune : bool
        If True, run GridSearchCV for Ridge, Lasso, ElasticNet.
        If False, run plain CV with default hyperparameters.

    Returns
    -------
    pd.DataFrame with columns:
        model, alpha, l1_ratio,
        mean_cv_mae_log, std_cv_mae_log,
        mean_cv_rmse_log, std_cv_rmse_log,
        mean_cv_r2_log, std_cv_r2_log, rank
    All metrics are on log1p_future_revenue.
    """
    all_rows: list[dict] = []

    for model_name in SUPPORTED_MODELS:
        if model_name == "linear" or not tune:
            # Plain CV — no grid search
            default_hparams: dict = {
                "ridge": {"alpha": 1.0},
                "lasso": {"alpha": 0.01},
                "elasticnet": {"alpha": 0.1, "l1_ratio": 0.5},
            }.get(model_name, {})

            cv_result = run_cv(
                model_name,
                train_df,
                hyperparameters=default_hparams,
                n_splits=n_splits,
                random_state=random_state,
                use_stratified=use_stratified,
            )
            hparams = cv_result["hyperparameters"]
            row = {
                "model": model_name,
                "alpha": hparams.get("alpha", None),
                "l1_ratio": hparams.get("l1_ratio", None),
                "mean_cv_mae_log": cv_result["mean_cv_mae_log"],
                "std_cv_mae_log": cv_result["std_cv_mae_log"],
                "mean_cv_rmse_log": cv_result["mean_cv_rmse_log"],
                "std_cv_rmse_log": cv_result["std_cv_rmse_log"],
                "mean_cv_r2_log": cv_result["mean_cv_r2_log"],
                "std_cv_r2_log": cv_result["std_cv_r2_log"],
                "rank": None,
            }
            all_rows.append(row)

        else:
            # GridSearchCV for regularised models
            _, results_df = run_grid_search(
                model_name,
                train_df,
                n_splits=n_splits,
                random_state=random_state,
                use_stratified=use_stratified,
            )
            for _, r in results_df.iterrows():
                all_rows.append(
                    {
                        "model": model_name,
                        "alpha": r.get("alpha"),
                        "l1_ratio": r.get("l1_ratio"),
                        "mean_cv_mae_log": None,  # not collected in grid search by default
                        "std_cv_mae_log": None,
                        "mean_cv_rmse_log": r["mean_cv_rmse_log"],
                        "std_cv_rmse_log": r["std_cv_rmse_log"],
                        "mean_cv_r2_log": None,
                        "std_cv_r2_log": None,
                        "rank": r["rank"],
                    }
                )

    df = pd.DataFrame(all_rows)
    logger.success(f"[run_all_cv] Completed. {len(df)} rows. tune={tune}")
    return df


# ---------------------------------------------------------------------------
# Persist CV results
# ---------------------------------------------------------------------------


def save_cv_results(
    cv_df: pd.DataFrame,
    path: Path | None = None,
) -> Path:
    """
    Save the CV results DataFrame to CSV.

    Default path: reports/metrics/cv_results.csv

    This file MUST NOT contain test-set metrics.  Only CV metrics from the
    training set are recorded here.

    Parameters
    ----------
    cv_df : pd.DataFrame
    path : Path or None

    Returns
    -------
    Path where the file was saved.
    """
    out = path or (METRICS_DIR / "cv_results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(out, index=False)
    logger.success(f"[save_cv_results] CV results saved → {out}")
    return out
