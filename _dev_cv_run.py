"""STEP 18 — Real-data development CV run. Training set only. Test set NEVER evaluated."""
import sys
from pathlib import Path
# Replicate conftest.py path setup so standalone script can import the package
_root = Path(__file__).parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))
import src as _src_pkg
_alias = "ecommerce_customer_value_prediction"
if _alias not in sys.modules:
    sys.modules[_alias] = _src_pkg
    for key in list(sys.modules):
        if key.startswith("src."):
            alias_key = _alias + key[len("src"):]
            if alias_key not in sys.modules:
                sys.modules[alias_key] = sys.modules[key]

import pandas as pd
from ecommerce_customer_value_prediction.models.tune import run_cv, run_grid_search, save_cv_results
from ecommerce_customer_value_prediction.pipelines.training_pipeline import _make_stratified_split

df = pd.read_csv("data/processed/customer_dataset.csv")
df["Customer ID"] = df["Customer ID"].astype("Int64")
print(f"Full dataset: {len(df):,} customers")

train_df, test_df = _make_stratified_split(df)
print(f"Train: {len(train_df):,}  |  Test (FROZEN — not evaluated): {len(test_df):,}")
print()

# ── OLS ──────────────────────────────────────────────────────────────────────
r = run_cv("linear", train_df)
print("OLS (no tuning)  — log1p scale, NOT £")
print(f"  CV RMSE_log : {r['mean_cv_rmse_log']:.4f} ± {r['std_cv_rmse_log']:.4f}")
print(f"  CV MAE_log  : {r['mean_cv_mae_log']:.4f}  ± {r['std_cv_mae_log']:.4f}")
print(f"  CV R²       : {r['mean_cv_r2_log']:.4f}  ± {r['std_cv_r2_log']:.4f}")
print()

# ── Ridge ─────────────────────────────────────────────────────────────────────
gs, res = run_grid_search("ridge", train_df)
bp = gs.best_params_
print("Ridge — log1p scale, NOT £")
print(f"  Best alpha      : {bp['regressor__alpha']}")
print(f"  Best CV RMSE_log: {-gs.best_score_:.4f}")
print()

# ── Lasso ─────────────────────────────────────────────────────────────────────
gs2, res2 = run_grid_search("lasso", train_df)
bp2 = gs2.best_params_
print("Lasso — log1p scale, NOT £")
print(f"  Best alpha      : {bp2['regressor__alpha']}")
print(f"  Best CV RMSE_log: {-gs2.best_score_:.4f}")
print()

# ── ElasticNet ────────────────────────────────────────────────────────────────
gs3, res3 = run_grid_search("elasticnet", train_df)
bp3 = gs3.best_params_
print("ElasticNet — log1p scale, NOT £")
print(f"  Best alpha      : {bp3['regressor__alpha']}")
print(f"  Best l1_ratio   : {bp3['regressor__l1_ratio']}")
print(f"  Best CV RMSE_log: {-gs3.best_score_:.4f}")
print()

# ── Save combined CV results ──────────────────────────────────────────────────
rows = []
for model, gs_obj in [("ridge", gs), ("lasso", gs2), ("elasticnet", gs3)]:
    for i, params in enumerate(gs_obj.cv_results_["params"]):
        rows.append({
            "model": model,
            "alpha": params.get("regressor__alpha"),
            "l1_ratio": params.get("regressor__l1_ratio"),
            "mean_cv_rmse_log": round(-float(gs_obj.cv_results_["mean_test_score"][i]), 6),
            "std_cv_rmse_log":  round(float(gs_obj.cv_results_["std_test_score"][i]),  6),
            "rank": int(gs_obj.cv_results_["rank_test_score"][i]),
        })

ols_row = {
    "model": "linear", "alpha": None, "l1_ratio": None,
    "mean_cv_rmse_log": r["mean_cv_rmse_log"],
    "std_cv_rmse_log": r["std_cv_rmse_log"],
    "rank": None,
}
rows.insert(0, ols_row)
cv_df = pd.DataFrame(rows)
save_cv_results(cv_df)
print("CV results saved to reports/metrics/cv_results.csv")
print()
print("=" * 60)
print("TEST SET WAS NOT EVALUATED IN THIS RUN.")
print("Frozen for final model comparison checkpoint only.")
print("=" * 60)
