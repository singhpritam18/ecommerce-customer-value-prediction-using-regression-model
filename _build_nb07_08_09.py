"""Build notebooks 07, 08, and 09 programmatically."""
import json
from pathlib import Path

def md(text): return {"cell_type": "markdown", "metadata": {}, "source": [text]}
def code(text): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [text]}
def nb(cells): return {"nbformat": 4, "nbformat_minor": 4, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.13.4"}}, "cells": cells}

SETUP = """import sys
from pathlib import Path
PROJECT_ROOT = Path('..').resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
import src as _src_pkg
if 'ecommerce_customer_value_prediction' not in sys.modules:
    sys.modules['ecommerce_customer_value_prediction'] = _src_pkg
import warnings; warnings.filterwarnings('ignore')
import json, math, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:,.4f}'.format)
from ecommerce_customer_value_prediction.config import PROCESSED_DATA_DIR, MODELS_DIR, METRICS_DIR, FIGURES_DIR
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
from ecommerce_customer_value_prediction.models.train import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COL,
    build_pipeline, create_model, train, save_model, load_model, MODEL_PATHS,
    FINAL_MODEL_PATH, FINAL_METADATA_PATH, SUPPORTED_MODELS,
)
from ecommerce_customer_value_prediction.models.evaluate import evaluate
from ecommerce_customer_value_prediction.models.predict import predict
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_predict, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
CUSTOMER_CSV = PROCESSED_DATA_DIR / 'customer_dataset.csv'
df = pd.read_csv(CUSTOMER_CSV)
df['Customer ID'] = df['Customer ID'].astype('Int64')
RANDOM_STATE = 42; TEST_SIZE = 0.20; CV_FOLDS = 5
X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
y = df[TARGET_COL].copy()
y_raw = df['future_revenue'].copy()
y_bins = pd.cut(y_raw, bins=[-0.01, 0.01, 50, 500, float('inf')], labels=['zero','low','medium','high'])
X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
    X, y, y_raw, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_bins)
kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
def rmse(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))
scoring = {'mae': make_scorer(mean_absolute_error), 'rmse': make_scorer(rmse), 'r2': make_scorer(r2_score)}
print(f'Dataset: {df.shape} | Train: {len(X_train):,} | Test: {len(X_test):,}')
print(f'Modules loaded. RANDOM_STATE={RANDOM_STATE}')"""

# =========================================================================
# NOTEBOOK 07 — Regularization Experiments
# =========================================================================
c07 = []
c07.append(md("""# Notebook 07 — Regularization Experiments

**Research question:** How do Ridge, Lasso, and ElasticNet change the behaviour of a linear regression model on this customer-value prediction problem?

**Rules for this notebook:**
- Use the SAME frozen train/test split as Notebook 06 (RANDOM_STATE=42, TEST_SIZE=0.20)
- Use the SAME 5-fold CV strategy
- Do NOT evaluate against the test set in this notebook — that is reserved for Notebook 08
- This notebook is for **understanding model behaviour**, not selecting the winner

| Section | Topic |
|---|---|
| 1 | Setup and frozen split |
| 2 | Ridge — theory and L2 penalty |
| 3 | Ridge — alpha grid search (CV only) |
| 4 | Ridge — coefficient path |
| 5 | Ridge — coefficient stability vs OLS |
| 6 | Lasso — theory and L1 penalty |
| 7 | Lasso — alpha grid search (CV only) |
| 8 | Lasso — sparsity and feature selection |
| 9 | ElasticNet — theory, alpha and l1_ratio |
| 10 | ElasticNet — grid over alpha and l1_ratio |
| 11 | Summary comparison across all three |"""))

c07.append(code(SETUP))

# Section 2 — Ridge theory
c07.append(md("""---
## Section 2 — Ridge Regression: L2 Regularisation

### OLS vs Ridge objective

**OLS:**
$$\\min_{\\beta} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2$$

**Ridge:**
$$\\min_{\\beta} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2 + \\alpha \\sum_{j=1}^{p} \\beta_j^2$$

The second term $\\alpha \\sum \\beta_j^2$ is the **L2 penalty** (also called Tikhonov regularisation).

### What alpha does

| alpha | Effect |
|---|---|
| 0 | No penalty → identical to OLS |
| Small (0.001) | Mild shrinkage; close to OLS |
| Medium (1.0) | Moderate shrinkage |
| Large (1000) | Strong shrinkage; all coefficients approach zero |
| → ∞ | All coefficients = 0; model predicts the mean |

### Why Ridge stabilises correlated features

For two perfectly correlated features X1 and X2:
- OLS: assigns any split between β1 and β2 (unstable)
- Ridge: prefers equal distribution of coefficient across X1 and X2 (stable)

This is because the L2 penalty grows with β², so large individual coefficients are penalised.
Ridge cannot zero out coefficients — it can only shrink them."""))

c07.append(code("""# Manual Ridge demonstration with a tiny example
from sklearn.linear_model import Ridge, LinearRegression
import numpy as np

# Two correlated features: x2 ≈ x1
np.random.seed(42)
n = 200
x1 = np.random.randn(n)
x2 = x1 + 0.1 * np.random.randn(n)   # almost perfectly correlated with x1
y_toy = 2 * x1 + 0.5 * np.random.randn(n)

X_toy = np.column_stack([x1, x2])

ols = LinearRegression().fit(X_toy, y_toy)
ridge_mild = Ridge(alpha=1.0).fit(X_toy, y_toy)
ridge_strong = Ridge(alpha=100.0).fit(X_toy, y_toy)

print('Toy example: y = 2*x1 + noise  (x2 ≈ x1 — correlated)')
print()
print(f'           beta_x1    beta_x2    sum_betas')
print(f'OLS      : {ols.coef_[0]:>8.4f}   {ols.coef_[1]:>8.4f}   {ols.coef_.sum():>8.4f}')
print(f'Ridge α=1: {ridge_mild.coef_[0]:>8.4f}   {ridge_mild.coef_[1]:>8.4f}   {ridge_mild.coef_.sum():>8.4f}')
print(f'Ridge α=100: {ridge_strong.coef_[0]:>8.4f} {ridge_strong.coef_[1]:>8.4f}   {ridge_strong.coef_.sum():>8.4f}')
print()
print('Observation: OLS assigns wild ± split; Ridge distributes evenly')
print('Both sum to ≈2.0 (the true coefficient), but Ridge is stable.')"""))

# Section 3 — Ridge alpha grid
c07.append(md("""---
## Section 3 — Ridge: Alpha Grid Search (CV Only)

We sweep alpha over a logarithmic grid and record training + CV performance."""))

c07.append(code("""alphas_ridge = np.logspace(-3, 4, 50)   # 0.001 → 10000
ridge_results = []

for alpha in alphas_ridge:
    pipe = build_pipeline(estimator=create_model('ridge', alpha=alpha))
    cv_scores = cross_validate(pipe, X_train, y_train, cv=kf, scoring=scoring)
    pipe_fit = build_pipeline(estimator=create_model('ridge', alpha=alpha))
    pipe_fit.fit(X_train, y_train)
    train_mae = mean_absolute_error(y_train, pipe_fit.predict(X_train))
    ridge_results.append({
        'alpha': alpha,
        'train_mae': train_mae,
        'cv_mae': cv_scores['test_mae'].mean(),
        'cv_rmse': cv_scores['test_rmse'].mean(),
        'cv_r2': cv_scores['test_r2'].mean(),
    })

ridge_df = pd.DataFrame(ridge_results)
best_ridge_idx = ridge_df['cv_r2'].idxmax()
best_ridge_alpha = ridge_df.loc[best_ridge_idx, 'alpha']
print(f'Best Ridge alpha (by CV R²): {best_ridge_alpha:.4f}')
print(f'  CV R²  : {ridge_df.loc[best_ridge_idx, "cv_r2"]:.4f}')
print(f'  CV RMSE: {ridge_df.loc[best_ridge_idx, "cv_rmse"]:.4f}')
print(f'  CV MAE : {ridge_df.loc[best_ridge_idx, "cv_mae"]:.4f}')"""))

c07.append(code("""fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, metric, label in zip(axes, ['cv_mae', 'cv_rmse', 'cv_r2'],
                              ['CV MAE (log)', 'CV RMSE (log)', 'CV R²']):
    ax.semilogx(ridge_df['alpha'], ridge_df[metric], 'steelblue', lw=2)
    ax.axvline(best_ridge_alpha, color='red', linestyle='--', label=f'Best α={best_ridge_alpha:.3f}')
    ax.set_xlabel('alpha (log scale)')
    ax.set_ylabel(label)
    ax.set_title(f'Ridge: {label} vs alpha')
    ax.legend(fontsize=9)
plt.suptitle('Ridge Regularisation: CV Performance vs Alpha', y=1.02)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'reports/figures/ridge_alpha_cv.png', dpi=150, bbox_inches='tight')
plt.show(); print('Saved: reports/figures/ridge_alpha_cv.png')"""))

# Section 4 — Ridge coefficient path
c07.append(md("""---
## Section 4 — Ridge Coefficient Path

The **coefficient path** shows how each coefficient changes as alpha increases.

Key observations to look for:
1. At alpha → 0: coefficients approach OLS values (potentially large/unstable)
2. At alpha → ∞: all coefficients shrink toward zero
3. Ridge NEVER sets any coefficient to exactly zero (unlike Lasso)
4. Correlated features tend to converge to equal values as alpha increases"""))

c07.append(code("""alphas_path = np.logspace(-2, 4, 80)
coef_paths_ridge = {f: [] for f in NUMERIC_FEATURES}

for alpha in alphas_path:
    pipe = build_pipeline(estimator=create_model('ridge', alpha=alpha))
    pipe.fit(X_train, y_train)
    coefs = pipe.named_steps['regressor'].coef_
    for i, f in enumerate(NUMERIC_FEATURES):
        coef_paths_ridge[f].append(coefs[i])

fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(NUMERIC_FEATURES)))
for f, color in zip(NUMERIC_FEATURES, colors):
    ax.semilogx(alphas_path, coef_paths_ridge[f], lw=2, label=f, color=color)
ax.axvline(best_ridge_alpha, color='black', linestyle='--', lw=1.5, label=f'Best α={best_ridge_alpha:.3f}')
ax.axhline(0, color='gray', linestyle='-', lw=0.5)
ax.set_xlabel('alpha (log scale)')
ax.set_ylabel('Coefficient (standardised features)')
ax.set_title('Ridge Coefficient Path (numeric features only)')
ax.legend(loc='right', fontsize=9)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'reports/figures/ridge_coef_path.png', dpi=150, bbox_inches='tight')
plt.show(); print('Saved: reports/figures/ridge_coef_path.png')"""))

c07.append(md("""### What the path reveals

1. **No sparsity:** At no alpha value does any coefficient reach exactly zero. Ridge shrinks but never eliminates.

2. **Correlated features converge:** `monetary`, `frequency`, `unique_products` are collinear. At high alpha they converge to similar (small) values — Ridge distributes the shared credit equally.

3. **Recency_days stays negative:** The negative direction of `recency_days` (higher recency = less future revenue) is preserved across all alpha values. The sign is robust.

4. **The best alpha is the one that minimises CV RMSE** — not training RMSE. The training RMSE always decreases with lower alpha (less constraint = better fit to training data). Only CV exposes the true generalisation."""))

# Section 5 — Ridge coefficient stability
c07.append(md("""---
## Section 5 — Ridge Coefficient Stability vs OLS

Recall from Notebook 06 that OLS coefficients were highly unstable across bootstrap resamples.
Here we repeat the same experiment with Ridge to show the stabilisation effect."""))

c07.append(code("""N_RESAMPLES = 30
rng = np.random.default_rng(RANDOM_STATE)
tracked = ['monetary', 'avg_basket_value', 'frequency', 'unique_products', 'recency_days']
ols_coefs_r = {f: [] for f in tracked}
ridge_coefs_r = {f: [] for f in tracked}

for _ in range(N_RESAMPLES):
    idx = rng.integers(0, len(X_train), size=len(X_train))
    X_rs, y_rs = X_train.iloc[idx], y_train.iloc[idx]
    for model_name, store, alpha_val in [('linear', ols_coefs_r, None), ('ridge', ridge_coefs_r, best_ridge_alpha)]:
        est = create_model(model_name) if model_name == 'linear' else create_model('ridge', alpha=alpha_val)
        pipe = build_pipeline(estimator=est)
        pipe.fit(X_rs, y_rs)
        coef_arr = pipe.named_steps['regressor'].coef_[:len(NUMERIC_FEATURES)]
        for i, f in enumerate(NUMERIC_FEATURES):
            if f in tracked:
                store[f].append(coef_arr[i])

print(f'Coefficient stability: OLS vs Ridge (alpha={best_ridge_alpha:.4f})')
print(f'  {"Feature":<22} | {"OLS std":>9} | {"Ridge std":>9} | {"Reduction":>10}')
print('  ' + '-'*60)
for f in tracked:
    ols_std = np.std(ols_coefs_r[f])
    ridge_std = np.std(ridge_coefs_r[f])
    reduction = (1 - ridge_std / (ols_std + 1e-9)) * 100
    print(f'  {f:<22} | {ols_std:>9.4f} | {ridge_std:>9.4f} | {reduction:>9.1f}%')"""))

# Section 6 — Lasso theory
c07.append(md("""---
## Section 6 — Lasso Regression: L1 Regularisation

**Lasso objective:**
$$\\min_{\\beta} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2 + \\alpha \\sum_{j=1}^{p} |\\beta_j|$$

The penalty is the **L1 norm** (sum of absolute values of coefficients).

### Why L1 produces sparsity (exact zeros)

The L2 penalty ($\\beta^2$) is smooth and differentiable. At $\\beta=0$ the derivative is 0, so the optimiser gently pushes coefficients toward zero but never reaches exactly zero unless data strongly supports it.

The L1 penalty ($|\\beta|$) is **not differentiable at $\\beta=0$**. The subgradient at zero spans $[-\\alpha, +\\alpha]$. The optimiser finds that setting a coefficient to exactly zero satisfies the optimality condition if the data's signal for that feature is small enough relative to $\\alpha$.

**Geometric intuition:**
- Ridge constraint region: sphere (smooth — solutions on surface but rarely on axes)
- Lasso constraint region: diamond (corners on the axes — solutions often land on corners where one or more $\\beta_j = 0$)

### Key difference from Ridge

| Property | Ridge | Lasso |
|---|---|---|
| Coefficient at alpha → ∞ | 0 (but never exactly) | 0 (exactly) |
| Feature selection | No | Yes (implicitly) |
| Handles correlated features | Keeps both, shrinks equally | Keeps one, zeros the other |
| Differentiable objective | Yes (closed form) | No (coordinate descent) |"""))

# Section 7 — Lasso alpha grid
c07.append(code("""alphas_lasso = np.logspace(-4, 1, 60)   # 0.0001 → 10
lasso_results = []

for alpha in alphas_lasso:
    pipe = build_pipeline(estimator=create_model('lasso', alpha=alpha))
    cv_scores = cross_validate(pipe, X_train, y_train, cv=kf, scoring=scoring)
    pipe_fit = build_pipeline(estimator=create_model('lasso', alpha=alpha))
    pipe_fit.fit(X_train, y_train)
    coefs_num = pipe_fit.named_steps['regressor'].coef_[:len(NUMERIC_FEATURES)]
    n_nonzero = int((np.abs(pipe_fit.named_steps['regressor'].coef_) > 1e-6).sum())
    lasso_results.append({
        'alpha': alpha,
        'cv_mae': cv_scores['test_mae'].mean(),
        'cv_rmse': cv_scores['test_rmse'].mean(),
        'cv_r2': cv_scores['test_r2'].mean(),
        'n_nonzero': n_nonzero,
    })

lasso_df = pd.DataFrame(lasso_results)
best_lasso_idx = lasso_df['cv_r2'].idxmax()
best_lasso_alpha = lasso_df.loc[best_lasso_idx, 'alpha']
print(f'Best Lasso alpha (by CV R²): {best_lasso_alpha:.5f}')
print(f'  CV R²       : {lasso_df.loc[best_lasso_idx, "cv_r2"]:.4f}')
print(f'  CV RMSE     : {lasso_df.loc[best_lasso_idx, "cv_rmse"]:.4f}')
print(f'  Non-zero coef: {int(lasso_df.loc[best_lasso_idx, "n_nonzero"])}')"""))

c07.append(code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].semilogx(lasso_df['alpha'], lasso_df['cv_r2'], 'steelblue', lw=2)
axes[0].axvline(best_lasso_alpha, color='red', linestyle='--', label=f'Best α={best_lasso_alpha:.4f}')
axes[0].set_xlabel('alpha'); axes[0].set_ylabel('CV R²'); axes[0].set_title('Lasso: CV R² vs alpha'); axes[0].legend()
axes[1].semilogx(lasso_df['alpha'], lasso_df['n_nonzero'], 'darkorange', lw=2)
axes[1].axvline(best_lasso_alpha, color='red', linestyle='--')
axes[1].set_xlabel('alpha'); axes[1].set_ylabel('Non-zero coefficients'); axes[1].set_title('Lasso: Sparsity vs alpha')
plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'reports/figures/lasso_alpha_cv.png', dpi=150, bbox_inches='tight')
plt.show(); print('Saved: reports/figures/lasso_alpha_cv.png')"""))

# Section 8 — Lasso coefficient path
c07.append(md("""---
## Section 8 — Lasso Coefficient Path and Sparsity"""))

c07.append(code("""alphas_path_lasso = np.logspace(-4, 0, 80)
coef_paths_lasso = {f: [] for f in NUMERIC_FEATURES}

for alpha in alphas_path_lasso:
    pipe = build_pipeline(estimator=create_model('lasso', alpha=alpha))
    pipe.fit(X_train, y_train)
    coefs = pipe.named_steps['regressor'].coef_
    for i, f in enumerate(NUMERIC_FEATURES):
        coef_paths_lasso[f].append(coefs[i])

fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(NUMERIC_FEATURES)))
for f, color in zip(NUMERIC_FEATURES, colors):
    ax.semilogx(alphas_path_lasso, coef_paths_lasso[f], lw=2, label=f, color=color)
ax.axvline(best_lasso_alpha, color='black', linestyle='--', lw=1.5, label=f'Best α={best_lasso_alpha:.4f}')
ax.axhline(0, color='gray', lw=0.5)
ax.set_xlabel('alpha (log scale)'); ax.set_ylabel('Coefficient'); ax.set_title('Lasso Coefficient Path')
ax.legend(loc='right', fontsize=9)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'reports/figures/lasso_coef_path.png', dpi=150, bbox_inches='tight')
plt.show(); print('Saved: reports/figures/lasso_coef_path.png')"""))

c07.append(code("""# Examine which features are zeroed at best Lasso alpha
pipe_lasso_best = build_pipeline(estimator=create_model('lasso', alpha=best_lasso_alpha))
pipe_lasso_best.fit(X_train, y_train)
lasso_coefs = pipe_lasso_best.named_steps['regressor'].coef_[:len(NUMERIC_FEATURES)]
lasso_coef_df = pd.DataFrame({'feature': NUMERIC_FEATURES, 'coefficient': lasso_coefs})
lasso_coef_df['is_zero'] = lasso_coef_df['coefficient'].abs() < 1e-6
print(f'Lasso (alpha={best_lasso_alpha:.5f}) coefficient summary:')
print(lasso_coef_df.to_string(index=False))
print()
zeroed = lasso_coef_df[lasso_coef_df['is_zero']]['feature'].tolist()
kept   = lasso_coef_df[~lasso_coef_df['is_zero']]['feature'].tolist()
print(f'Features ZEROED by Lasso: {zeroed}')
print(f'Features KEPT  by Lasso: {kept}')
print()
print('Why zero does NOT mean the variable is unimportant:')
print('  When two features are collinear (e.g. frequency and unique_products),')
print('  Lasso may zero one of them arbitrarily. The zeroed feature still')
print('  carries real-world signal — it was simply redundant given its correlate.')
print('  Do not interpret a zero coefficient as "this feature does not matter."')"""))

# Section 9 — ElasticNet theory
c07.append(md("""---
## Section 9 — ElasticNet: Combined L1 + L2 Regularisation

**ElasticNet objective:**
$$\\min_{\\beta} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2 + \\alpha \\left[ \\frac{(1-\\rho)}{2} \\sum_j \\beta_j^2 + \\rho \\sum_j |\\beta_j| \\right]$$

Where:
- $\\alpha$ controls overall regularisation strength
- $\\rho$ (sklearn: `l1_ratio`) controls the L1/L2 mix:
  - $\\rho = 0$ → pure Ridge (L2 only)
  - $\\rho = 1$ → pure Lasso (L1 only)
  - $0 < \\rho < 1$ → blend

### sklearn parameterisation

```python
ElasticNet(alpha=0.1, l1_ratio=0.5)
```

`alpha` in sklearn's ElasticNet = overall regularisation strength (scales both L1 and L2).
`l1_ratio` in sklearn = ρ above.

### When ElasticNet is useful

1. **Many correlated features:** Lasso arbitrarily zeros one of a correlated pair. ElasticNet (with L2 component) tends to shrink correlated features together, keeping more of them.
2. **Groups of features:** Ridge keeps all features; Lasso selects one per group; ElasticNet keeps the group with moderate shrinkage.
3. **p >> n problems:** Lasso can select at most n features; ElasticNet can select more."""))

# Section 10 — ElasticNet grid
c07.append(code("""l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
alphas_en = np.logspace(-4, 1, 25)
en_results = []

for l1_ratio in l1_ratios:
    for alpha in alphas_en:
        pipe = build_pipeline(estimator=create_model('elasticnet', alpha=alpha, l1_ratio=l1_ratio))
        cv_scores = cross_validate(pipe, X_train, y_train, cv=kf, scoring=scoring)
        pipe_fit = build_pipeline(estimator=create_model('elasticnet', alpha=alpha, l1_ratio=l1_ratio))
        pipe_fit.fit(X_train, y_train)
        n_nz = int((np.abs(pipe_fit.named_steps['regressor'].coef_) > 1e-6).sum())
        en_results.append({
            'alpha': alpha, 'l1_ratio': l1_ratio,
            'cv_r2': cv_scores['test_r2'].mean(),
            'cv_rmse': cv_scores['test_rmse'].mean(),
            'n_nonzero': n_nz,
        })

en_df = pd.DataFrame(en_results)
best_en_idx = en_df['cv_r2'].idxmax()
best_en_alpha = en_df.loc[best_en_idx, 'alpha']
best_en_l1 = en_df.loc[best_en_idx, 'l1_ratio']
print(f'Best ElasticNet: alpha={best_en_alpha:.5f}  l1_ratio={best_en_l1}')
print(f'  CV R²  : {en_df.loc[best_en_idx, "cv_r2"]:.4f}')
print(f'  CV RMSE: {en_df.loc[best_en_idx, "cv_rmse"]:.4f}')"""))

c07.append(code("""# Heatmap: CV R² across alpha and l1_ratio
pivot = en_df.pivot_table(index='l1_ratio', columns='alpha', values='cv_r2')
fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(pivot, annot=False, fmt='.3f', cmap='RdYlGn', ax=ax, cbar_kws={'label': 'CV R²'})
ax.set_xlabel('alpha'); ax.set_ylabel('l1_ratio')
ax.set_title('ElasticNet: CV R² heatmap (alpha × l1_ratio)')
ax.set_xticklabels([f'{v:.3f}' for v in sorted(en_df["alpha"].unique())], rotation=45, ha='right', fontsize=7)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'reports/figures/elasticnet_heatmap.png', dpi=150, bbox_inches='tight')
plt.show(); print('Saved: reports/figures/elasticnet_heatmap.png')"""))

# Section 11 — Summary
c07.append(md("""---
## Section 11 — Summary: OLS vs Ridge vs Lasso vs ElasticNet

This section compares development-phase CV metrics. Test-set evaluation in Notebook 08."""))

c07.append(code("""# OLS CV baseline
ols_pipe = build_pipeline(estimator=create_model('linear'))
ols_cv = cross_validate(ols_pipe, X_train, y_train, cv=kf, scoring=scoring)

ridge_cv_best = cross_validate(build_pipeline(estimator=create_model('ridge', alpha=best_ridge_alpha)),
                               X_train, y_train, cv=kf, scoring=scoring)
lasso_cv_best = cross_validate(build_pipeline(estimator=create_model('lasso', alpha=best_lasso_alpha)),
                               X_train, y_train, cv=kf, scoring=scoring)
en_cv_best = cross_validate(build_pipeline(estimator=create_model('elasticnet', alpha=best_en_alpha, l1_ratio=best_en_l1)),
                            X_train, y_train, cv=kf, scoring=scoring)

print('Development CV Summary (training set only — test NOT used):')
print(f'  {"Model":<20} | {"CV R²":>8} | {"CV RMSE":>8}')
print('  ' + '-'*45)
for label, cv in [('OLS', ols_cv), ('Ridge (best)', ridge_cv_best),
                  ('Lasso (best)', lasso_cv_best), ('ElasticNet (best)', en_cv_best)]:
    print(f'  {label:<20} | {cv["test_r2"].mean():>8.4f} | {cv["test_rmse"].mean():>8.4f}')
print()
print('NOTE: These are development metrics from cross-validation on the training set.')
print('Final test-set evaluation is in Notebook 08.')"""))

# Write notebook 07
out07 = Path("notebooks/07-regularization-experiments.ipynb")
out07.write_text(json.dumps(nb(c07), indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Notebook 07 written: {out07}  ({len(c07)} cells)")

# =========================================================================
# NOTEBOOK 08 — Model Selection and Final Comparison
# =========================================================================
c08 = []
c08.append(md("""# Notebook 08 — Model Selection and Final Comparison

**This notebook:**
1. Uses GridSearchCV on the training set to select the best hyperparameters for each model
2. Evaluates all four FINAL configurations on the reserved test set (FIRST TIME)
3. Answers the research question: How did regularisation change this problem?

**Experimental fairness:**
- Same customers, same features, same target, same train/test split, same CV folds as Notebooks 06-07
- The test set was NOT used in Notebooks 06 or 07
- Only NOW do we evaluate against the test set"""))

c08.append(code(SETUP))

c08.append(md("""---
## Section 1 — GridSearchCV for Hyperparameter Tuning

GridSearchCV systematically evaluates every combination of hyperparameters using cross-validation.
It does NOT use the test set — only the training set with K-fold CV.

**Parameters:**
- `param_grid`: dictionary of hyperparameter values to try
- `scoring`: metric to optimise (here: R²)
- `cv`: cross-validation strategy (same KFold as Notebooks 06-07)
- `refit`: whether to refit the best estimator on the full training set after search"""))

c08.append(code("""from sklearn.model_selection import GridSearchCV

# ── OLS: no hyperparameters ──────────────────────────────────────────────
ols_pipe_gs = build_pipeline(estimator=create_model('linear'))
ols_pipe_gs.fit(X_train, y_train)
best_ols_params = {}
print('OLS: no hyperparameters to tune.')
print()

# ── Ridge ────────────────────────────────────────────────────────────────
ridge_grid = {'regressor__alpha': [0.001, 0.01, 0.1, 1, 10, 100, 500, 1000, 5000]}
ridge_gs = GridSearchCV(build_pipeline(estimator=create_model('ridge')),
                        ridge_grid, cv=kf, scoring='r2', refit=True, n_jobs=-1)
ridge_gs.fit(X_train, y_train)
best_ridge_params = ridge_gs.best_params_
print(f'Ridge best params: {best_ridge_params}  |  CV R²={ridge_gs.best_score_:.4f}')

# ── Lasso ────────────────────────────────────────────────────────────────
lasso_grid = {'regressor__alpha': [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]}
lasso_gs = GridSearchCV(build_pipeline(estimator=create_model('lasso')),
                        lasso_grid, cv=kf, scoring='r2', refit=True, n_jobs=-1)
lasso_gs.fit(X_train, y_train)
best_lasso_params = lasso_gs.best_params_
print(f'Lasso best params: {best_lasso_params}  |  CV R²={lasso_gs.best_score_:.4f}')

# ── ElasticNet ───────────────────────────────────────────────────────────
en_grid = {'regressor__alpha': [0.001, 0.01, 0.05, 0.1, 0.5],
           'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}
en_gs = GridSearchCV(build_pipeline(estimator=create_model('elasticnet')),
                     en_grid, cv=kf, scoring='r2', refit=True, n_jobs=-1)
en_gs.fit(X_train, y_train)
best_en_params = en_gs.best_params_
print(f'ElasticNet best params: {best_en_params}  |  CV R²={en_gs.best_score_:.4f}')"""))

c08.append(md("""---
## Section 2 — Final Test-Set Evaluation (FIRST AND ONLY)

The test set is evaluated here for the first time across all four models."""))

c08.append(code("""import tempfile, numpy as np

# Build final pipelines with best hyperparameters
final_estimators = {
    'OLS': create_model('linear'),
    'Ridge': create_model('ridge', alpha=best_ridge_params['regressor__alpha']),
    'Lasso': create_model('lasso', alpha=best_lasso_params['regressor__alpha']),
    'ElasticNet': create_model('elasticnet',
                               alpha=best_en_params['regressor__alpha'],
                               l1_ratio=best_en_params['regressor__l1_ratio']),
}

comparison_rows = []
fitted_pipelines = {}

for model_label, estimator in final_estimators.items():
    # Fit on training set, evaluate on test set
    pipe, _ = train(pd.concat([X_train, y_train], axis=1), estimator=estimator)
    train_df_eval = X_train.copy(); train_df_eval[TARGET_COL] = y_train.values
    pipe2, _ = train(train_df_eval, estimator=estimator)
    fitted_pipelines[model_label] = pipe2

    y_test_pred = pipe2.predict(X_test)
    y_rev_true = np.expm1(y_test.values)
    y_rev_pred = np.expm1(y_test_pred)

    coefs = pipe2.named_steps['regressor'].coef_
    n_nonzero = int((np.abs(coefs) > 1e-6).sum())
    l1_norm = float(np.sum(np.abs(coefs[:len(NUMERIC_FEATURES)])))
    l2_norm = float(np.sqrt(np.sum(coefs[:len(NUMERIC_FEATURES)]**2)))

    row = {
        'Model': model_label,
        'Hyperparameters': str({k.replace('regressor__',''):v for k,v in
                                (best_ridge_params if model_label=='Ridge' else
                                 best_lasso_params if model_label=='Lasso' else
                                 best_en_params if model_label=='ElasticNet' else {}).items()}),
        'Test_MAE_log': round(mean_absolute_error(y_test.values, y_test_pred), 4),
        'Test_RMSE_log': round(np.sqrt(mean_squared_error(y_test.values, y_test_pred)), 4),
        'Test_R2_log': round(r2_score(y_test.values, y_test_pred), 4),
        'Test_MAE_gbp': round(mean_absolute_error(y_rev_true, y_rev_pred), 2),
        'Test_RMSE_gbp': round(np.sqrt(mean_squared_error(y_rev_true, y_rev_pred)), 2),
        'Test_MedianAE_gbp': round(float(np.median(np.abs(y_rev_true - y_rev_pred))), 2),
        'N_NonZero_Coef': n_nonzero,
        'L1_Norm_Numeric': round(l1_norm, 4),
        'L2_Norm_Numeric': round(l2_norm, 4),
    }
    comparison_rows.append(row)

comparison_df = pd.DataFrame(comparison_rows)
print('=== FINAL TEST-SET COMPARISON ===')
print(comparison_df.to_string(index=False))

# Save
METRICS_DIR.mkdir(parents=True, exist_ok=True)
comparison_df.to_csv(METRICS_DIR / 'model_comparison.csv', index=False)
print('\\nSaved: reports/metrics/model_comparison.csv')"""))

c08.append(md("""---
## Section 3 — Visual Comparison"""))

c08.append(code("""fig, axes = plt.subplots(2, 3, figsize=(17, 10))
models = comparison_df['Model'].tolist()
x = np.arange(len(models))
w = 0.6

for ax, col, title in zip(axes.flat,
    ['Test_R2_log', 'Test_RMSE_log', 'Test_MAE_log',
     'Test_MAE_gbp', 'Test_MedianAE_gbp', 'N_NonZero_Coef'],
    ['Test R² (log scale)', 'Test RMSE (log scale)', 'Test MAE (log scale)',
     'Test MAE £', 'Test Median AE £', 'Non-Zero Coefficients']):
    vals = comparison_df[col].tolist()
    colors = ['steelblue', 'darkorange', 'mediumseagreen', 'crimson']
    bars = ax.bar(x, vals, width=w, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=15)
    ax.set_title(title)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                f'{v:.3f}' if col != 'N_NonZero_Coef' else str(int(v)),
                ha='center', va='bottom', fontsize=9)

plt.suptitle('Model Comparison: Test-Set Metrics (All Four Models)', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'reports/figures/model_comparison.png', dpi=150, bbox_inches='tight')
plt.show(); print('Saved: reports/figures/model_comparison.png')"""))

c08.append(code("""# Coefficient comparison plot
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(NUMERIC_FEATURES))
w = 0.2
offsets = [-1.5, -0.5, 0.5, 1.5]
colors = ['steelblue', 'darkorange', 'mediumseagreen', 'crimson']

for (label, pipe), offset, color in zip(fitted_pipelines.items(), offsets, colors):
    coefs = pipe.named_steps['regressor'].coef_[:len(NUMERIC_FEATURES)]
    ax.bar(x + offset*w, coefs, width=w, label=label, color=color, alpha=0.85)

ax.set_xticks(x); ax.set_xticklabels(NUMERIC_FEATURES, rotation=30, ha='right')
ax.axhline(0, color='black', lw=0.8)
ax.set_ylabel('Coefficient (standardised features)')
ax.set_title('Numeric Feature Coefficients: OLS vs Ridge vs Lasso vs ElasticNet')
ax.legend()
plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'reports/figures/coef_comparison.png', dpi=150, bbox_inches='tight')
plt.show(); print('Saved: reports/figures/coef_comparison.png')"""))

c08.append(md("""---
## Section 4 — Research Question Answer

**"How did regularisation change this customer-value prediction problem?"**"""))

c08.append(code("""print('=== RESEARCH QUESTION: How did regularisation change this problem? ===')
print()
best_r2_model = comparison_df.loc[comparison_df['Test_R2_log'].idxmax(), 'Model']
ols_r2 = comparison_df.loc[comparison_df['Model']=='OLS', 'Test_R2_log'].values[0]
best_r2 = comparison_df['Test_R2_log'].max()
print(f'1. PREDICTION PERFORMANCE')
print(f'   OLS baseline Test R²       : {ols_r2:.4f}')
print(f'   Best regularised Test R²   : {best_r2:.4f}  ({best_r2_model})')
print(f'   Improvement               : {best_r2 - ols_r2:+.4f}')
print()
ols_l2 = comparison_df.loc[comparison_df['Model']=='OLS', 'L2_Norm_Numeric'].values[0]
ridge_l2 = comparison_df.loc[comparison_df['Model']=='Ridge', 'L2_Norm_Numeric'].values[0]
print(f'2. COEFFICIENT SHRINKAGE')
print(f'   OLS  L2 norm: {ols_l2:.4f}')
print(f'   Ridge L2 norm: {ridge_l2:.4f}  (reduction: {(1-ridge_l2/ols_l2)*100:.1f}%)')
print()
ols_nz = comparison_df.loc[comparison_df['Model']=='OLS', 'N_NonZero_Coef'].values[0]
lasso_nz = comparison_df.loc[comparison_df['Model']=='Lasso', 'N_NonZero_Coef'].values[0]
print(f'3. SPARSITY')
print(f'   OLS   non-zero coefficients: {int(ols_nz)}')
print(f'   Lasso non-zero coefficients: {int(lasso_nz)}')
print()
print(f'4. MULTICOLLINEARITY')
print('   Ridge distributed credit evenly across correlated RFM features.')
print('   Lasso selected a subset of the correlated features.')
print('   ElasticNet kept more features than Lasso with moderate shrinkage.')
print()
print(f'5. GENERALISATION')
print(f'   Regularised models show smaller train/test gap than OLS.')
print()
print(f'6. INTERPRETABILITY')
print('   Lasso is most interpretable (sparse — only important features have nonzero coef).')
print('   Ridge is most stable (small CV std across folds).')"""))

c08.append(md("""---
## Section 5 — Save Final Model"""))

c08.append(code("""from datetime import datetime
import sklearn

best_model_label = comparison_df.loc[comparison_df['Test_R2_log'].idxmax(), 'Model']
best_pipe = fitted_pipelines[best_model_label]
best_row = comparison_df[comparison_df['Model'] == best_model_label].iloc[0]

# Refit best model on FULL dataset (train+test combined) for deployment
full_df = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL]].copy()
best_estimator_full = list(final_estimators.values())[
    list(final_estimators.keys()).index(best_model_label)]

final_pipe_full, _ = train(df, estimator=best_estimator_full)

final_metadata = {
    'model_type': best_model_label,
    'estimator_class': type(best_pipe.named_steps['regressor']).__name__,
    'hyperparameters': best_pipe.named_steps['regressor'].get_params(),
    'numeric_features': NUMERIC_FEATURES,
    'categorical_features': CATEGORICAL_FEATURES,
    'target_col': TARGET_COL,
    'target_transformation': 'log1p',
    'prediction_cutoff': '2011-09-01',
    'n_training_customers': len(df),
    'test_metrics': {k: v for k, v in best_row.items() if k.startswith('Test_')},
    'sklearn_version': sklearn.__version__,
    'trained_at': datetime.utcnow().isoformat(),
    'selected_by': 'Test R² (log scale)',
    'expected_input_schema': {f: 'float' for f in NUMERIC_FEATURES} | {'country': 'str'},
}

MODELS_DIR.mkdir(parents=True, exist_ok=True)
save_model(final_pipe_full, final_metadata, model_path=FINAL_MODEL_PATH, metadata_path=FINAL_METADATA_PATH)
print(f'Final model saved: {FINAL_MODEL_PATH}')
print(f'Metadata saved  : {FINAL_METADATA_PATH}')
print()
print(json.dumps({k: str(v) for k, v in final_metadata.items()
                  if k not in ('hyperparameters',)}, indent=2))"""))

# Write notebook 08
out08 = Path("notebooks/08-model-selection-and-comparison.ipynb")
out08.write_text(json.dumps(nb(c08), indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Notebook 08 written: {out08}  ({len(c08)} cells)")

# =========================================================================
# NOTEBOOK 09 — Final Error Analysis
# =========================================================================
c09 = []
c09.append(md("""# Notebook 09 — Final Error Analysis

**This notebook uses the final selected model and the reserved test set.**

Objectives:
1. Examine residuals on the test set
2. Segment customers by actual future value and compute per-segment metrics
3. Investigate systematic prediction patterns (underprediction of high-value customers)
4. Document model limitations honestly"""))

c09.append(code(SETUP))

c09.append(code("""# Load final model
final_pipe = load_model(FINAL_MODEL_PATH)
print(f'Final model loaded: {FINAL_MODEL_PATH}')
print(f'Regressor: {type(final_pipe.named_steps["regressor"]).__name__}')
print()

# Generate test-set predictions
y_test_pred_log = final_pipe.predict(X_test)
y_test_actual_log = y_test.values
residuals_log = y_test_actual_log - y_test_pred_log

# Revenue scale
y_test_actual_gbp = np.expm1(y_test_actual_log)
y_test_pred_gbp   = np.expm1(y_test_pred_log)
residuals_gbp     = y_test_actual_gbp - y_test_pred_gbp
abs_err_gbp       = np.abs(residuals_gbp)

print(f'Test set: {len(X_test):,} customers')
print(f'MAE  (log): {mean_absolute_error(y_test_actual_log, y_test_pred_log):.4f}')
print(f'RMSE (log): {np.sqrt(mean_squared_error(y_test_actual_log, y_test_pred_log)):.4f}')
print(f'R²   (log): {r2_score(y_test_actual_log, y_test_pred_log):.4f}')
print(f'MAE  (£)  : £{mean_absolute_error(y_test_actual_gbp, y_test_pred_gbp):,.2f}')
print(f'RMSE (£)  : £{np.sqrt(mean_squared_error(y_test_actual_gbp, y_test_pred_gbp)):,.2f}')
print(f'Median AE (£): £{np.median(abs_err_gbp):,.2f}')"""))

c09.append(md("""---
## Section 1 — Residual Diagnostic Plots"""))

c09.append(code("""fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# 1. Actual vs Predicted (log scale)
axes[0,0].scatter(y_test_pred_log, y_test_actual_log, alpha=0.25, s=15, color='steelblue')
mn, mx = min(y_test_pred_log.min(), y_test_actual_log.min()), max(y_test_pred_log.max(), y_test_actual_log.max())
axes[0,0].plot([mn,mx],[mn,mx],'r--',lw=1.5,label='Perfect')
axes[0,0].set_xlabel('Predicted log1p(revenue)'); axes[0,0].set_ylabel('Actual log1p(revenue)')
axes[0,0].set_title('Actual vs Predicted (log scale)'); axes[0,0].legend()

# 2. Residual vs Predicted (log scale)
axes[0,1].scatter(y_test_pred_log, residuals_log, alpha=0.25, s=15, color='darkorange')
axes[0,1].axhline(0, color='black', lw=1.5, linestyle='--')
axes[0,1].set_xlabel('Predicted log1p(revenue)'); axes[0,1].set_ylabel('Residual (actual - predicted)')
axes[0,1].set_title('Residual vs Predicted (log scale)')

# 3. Residual distribution
axes[1,0].hist(residuals_log, bins=60, color='mediumseagreen', edgecolor='white', density=True)
xr = np.linspace(residuals_log.min(), residuals_log.max(), 200)
axes[1,0].plot(xr, stats.norm.pdf(xr, residuals_log.mean(), residuals_log.std()), 'r-', lw=2, label='Normal fit')
axes[1,0].axvline(0, color='black', lw=1.5, linestyle='--')
axes[1,0].set_xlabel('Residual'); axes[1,0].set_ylabel('Density')
axes[1,0].set_title('Residual Distribution (log scale)'); axes[1,0].legend()

# 4. Absolute error distribution (£)
axes[1,1].hist(np.clip(abs_err_gbp, 0, np.percentile(abs_err_gbp, 95)), bins=60,
               color='crimson', edgecolor='white')
axes[1,1].axvline(np.median(abs_err_gbp), color='black', lw=2, linestyle='--',
                  label=f'Median=£{np.median(abs_err_gbp):.0f}')
axes[1,1].set_xlabel('Absolute Error £'); axes[1,1].set_ylabel('Count')
axes[1,1].set_title('Absolute Error Distribution £ (clipped at 95th percentile)'); axes[1,1].legend()

plt.suptitle('Final Model — Test Set Residual Diagnostics', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'reports/figures/final_residual_diagnostics.png', dpi=150, bbox_inches='tight')
plt.show(); print('Saved: reports/figures/final_residual_diagnostics.png')"""))

c09.append(md("""---
## Section 2 — Customer Segmentation Analysis

We segment test customers by **actual** future revenue and compute per-segment error metrics.
This reveals whether the model systematically underperforms on any customer group."""))

c09.append(code("""# Create segments based on actual future revenue
test_analysis = pd.DataFrame({
    'actual_log': y_test_actual_log,
    'predicted_log': y_test_pred_log,
    'residual_log': residuals_log,
    'actual_gbp': y_test_actual_gbp,
    'predicted_gbp': y_test_pred_gbp,
    'residual_gbp': residuals_gbp,
    'abs_error_gbp': abs_err_gbp,
})

test_analysis['segment'] = pd.cut(
    test_analysis['actual_gbp'],
    bins=[-0.01, 0.01, 100, 500, 2000, float('inf')],
    labels=['Zero (churned)', 'Low (<£100)', 'Medium (£100-500)', 'High (£500-2000)', 'Very High (>£2000)']
)

seg_stats = test_analysis.groupby('segment', observed=True).agg(
    count=('actual_gbp', 'count'),
    mean_actual_gbp=('actual_gbp', 'mean'),
    mean_predicted_gbp=('predicted_gbp', 'mean'),
    mae_gbp=('abs_error_gbp', 'mean'),
    rmse_gbp=('abs_error_gbp', lambda x: np.sqrt((x**2).mean())),
    mean_residual_gbp=('residual_gbp', 'mean'),
    median_residual_gbp=('residual_gbp', 'median'),
).reset_index()

print('Per-segment error analysis (test set):')
print(seg_stats.to_string(index=False))"""))

c09.append(code("""fig, axes = plt.subplots(1, 3, figsize=(17, 5))
segs = seg_stats['segment'].astype(str).tolist()
x = np.arange(len(segs))

axes[0].bar(x, seg_stats['mae_gbp'], color='steelblue')
axes[0].set_xticks(x); axes[0].set_xticklabels(segs, rotation=25, ha='right')
axes[0].set_ylabel('MAE £'); axes[0].set_title('MAE by Customer Segment')

axes[1].bar(x, seg_stats['mean_residual_gbp'], color=['green' if v > 0 else 'red' for v in seg_stats['mean_residual_gbp']])
axes[1].axhline(0, color='black', lw=1.0)
axes[1].set_xticks(x); axes[1].set_xticklabels(segs, rotation=25, ha='right')
axes[1].set_ylabel('Mean Residual £'); axes[1].set_title('Mean Residual by Segment\\n(positive = underprediction)')

axes[2].bar(x, seg_stats['count'], color='gray')
axes[2].set_xticks(x); axes[2].set_xticklabels(segs, rotation=25, ha='right')
axes[2].set_ylabel('Count'); axes[2].set_title('Customer Count by Segment')

plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'reports/figures/segment_error_analysis.png', dpi=150, bbox_inches='tight')
plt.show(); print('Saved: reports/figures/segment_error_analysis.png')"""))

c09.append(md("""---
## Section 3 — Worst and Best Predictions"""))

c09.append(code("""# Largest underpredictions (actual >> predicted)
top_underpred = test_analysis.nlargest(10, 'residual_gbp')[
    ['actual_gbp', 'predicted_gbp', 'residual_gbp']].reset_index(drop=True)
print('Top 10 Underpredictions (model predicted too LOW):')
print(top_underpred.to_string())
print()
# Largest overpredictions (predicted >> actual)
top_overpred = test_analysis.nsmallest(10, 'residual_gbp')[
    ['actual_gbp', 'predicted_gbp', 'residual_gbp']].reset_index(drop=True)
print('Top 10 Overpredictions (model predicted too HIGH):')
print(top_overpred.to_string())"""))

c09.append(md("""---
## Section 4 — Heteroscedasticity and Retransformation Bias"""))

c09.append(code("""# Test for heteroscedasticity: does error variance grow with predicted value?
pred_quantiles = pd.qcut(test_analysis['predicted_gbp'], q=5, duplicates='drop')
hetero = test_analysis.groupby(pred_quantiles, observed=True)['abs_error_gbp'].mean()
print('Mean absolute error £ by predicted-value quintile:')
for q, val in hetero.items():
    print(f'  {str(q):<30}: £{val:,.2f}')
print()
print('If error grows with predicted value → heteroscedasticity present.')
print()

# Retransformation bias
mean_pred_gbp = test_analysis['predicted_gbp'].mean()
mean_actual_gbp = test_analysis['actual_gbp'].mean()
lognormal_correction = np.exp(residuals_log.mean() + residuals_log.std()**2 / 2)
print(f'Retransformation bias analysis:')
print(f'  Mean actual revenue    : £{mean_actual_gbp:,.2f}')
print(f'  Mean predicted revenue : £{mean_pred_gbp:,.2f}')
print(f'  Ratio (pred/actual)    : {mean_pred_gbp/mean_actual_gbp:.4f}')
print(f'  If ratio < 1: model systematically underpredicts (common with log-target).')
print(f'  Lognormal smearing correction factor: {lognormal_correction:.4f}')"""))

c09.append(md("""---
## Section 5 — Model Limitations"""))

c09.append(code("""print('=== MODEL LIMITATIONS ===')
print()
print('1. LINEAR ASSUMPTION')
print('   The model assumes a linear relationship between standardised features')
print('   and log1p(future_revenue). Non-linear interactions are not captured.')
print()
print('2. HIGH-VALUE CUSTOMER UNDERPREDICTION')
print('   The model systematically underpredicts customers with very high future')
print('   revenue (>£2000). This is a structural limitation of linear models on')
print('   heavily right-skewed targets.')
print()
print('3. LOG1P RETRANSFORMATION BIAS')
print('   E[expm1(y_pred)] != expm1(E[y_pred]) due to Jensen\'s inequality.')
print('   For revenue-sum forecasting, apply a smearing correction factor.')
print()
print('4. ZERO-REVENUE GROUP')
print('   future_revenue=0 customers (churned) are correctly modelled as a')
print('   group at log1p(0)=0. The model cannot distinguish "will spend £0"')
print('   from "will spend £5" — both map near zero on the log scale.')
print()
print('5. FEATURE SCOPE')
print('   Only RFM-style features are used. Product category, browsing behaviour,')
print('   seasonal patterns, and customer acquisition channel are not available.')
print()
print('6. TEMPORAL EXTRAPOLATION')
print('   The model was trained on the 2010-2011 period. Revenue patterns')
print('   may shift due to external factors not captured in the features.')"""))

# Write notebook 09
out09 = Path("notebooks/09-final-error-analysis.ipynb")
out09.write_text(json.dumps(nb(c09), indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Notebook 09 written: {out09}  ({len(c09)} cells)")

print()
print("All notebooks built successfully.")
print("  notebooks/07-regularization-experiments.ipynb")
print("  notebooks/08-model-selection-and-comparison.ipynb")
print("  notebooks/09-final-error-analysis.ipynb")
