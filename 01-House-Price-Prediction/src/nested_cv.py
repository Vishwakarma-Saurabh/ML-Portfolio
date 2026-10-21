import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from .features import FEATURE_BUILDERS
from .preprocess import TARGET, build_preprocessor
from .train import apply_features, load_data


def nested_score(make_pipeline, param_dist, X, y, outer_splits=5, inner_splits=3, n_iter_inner=15, seed=42,):
  if isinstance(y, (pd.DataFrame, pd.Series)):
    y = y.values.ravel()
  else:
    y = np.ravel(y)

  outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=seed)
  y_true_all, y_pred_all = [], []

  for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X)):
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=seed)

    if param_dist is not None:
      search = RandomizedSearchCV(make_pipeline(X_tr.columns), param_dist, n_iter=n_iter_inner, cv=inner_cv, scoring="neg_mean_absolute_error", random_state=seed, n_jobs=-1,)
      search.fit(X_tr, y_tr)
      best = search.best_estimator_
    else:
      best = make_pipeline(X_tr.columns).fit(X_tr, y_tr)

    y_pred_all.extend(best.predict(X_te))
    y_true_all.extend(y_te)
    print(f"  fold {fold+1}/{outer_splits} done")

  y_true_all = np.array(y_true_all)
  y_pred_all = np.array(y_pred_all)

  return {
      "MAE": mean_absolute_error(y_true_all, y_pred_all),
      "RMSE": np.sqrt(mean_squared_error(y_true_all, y_pred_all)),
      "R2": r2_score(y_true_all, y_pred_all),
  }


def main():
  df = apply_features(load_data(), list(FEATURE_BUILDERS.keys()))

  target_col = TARGET[0] if isinstance(TARGET, (list, tuple)) else TARGET

  X = df.drop(columns=[target_col])
  y = df[target_col]

  def make_ridge(cols):
    return Pipeline([
        ("prep", build_preprocessor(list(cols))),
        ("model", TransformedTargetRegressor(
                regressor=RidgeCV(alphas=np.logspace(-2, 4, 50)),
                func=np.log1p,
                inverse_func=np.expm1,
            ),
        ),
    ])

  print("═══ Ridge — nested CV ═══")
  ridge_result = nested_score(make_ridge, None, X, y)
  print(f"  Nested: MAE ₹{ridge_result['MAE']:,.0f}  R² {ridge_result['R2']:.3f}")

  def make_rf(cols):
    return Pipeline([
        ("prep", build_preprocessor(list(cols))),
        ("model",TransformedTargetRegressor(regressor=RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1), func=np.log1p, inverse_func=np.expm1,),),
    ])

  rf_dist = {
      "model__regressor__max_depth": randint(3, 25),
      "model__regressor__min_samples_leaf": randint(1, 10),
  }

  print("═══ RandomForest — nested CV (small inner search) ═══")
  rf_result = nested_score(make_rf, rf_dist, X, y, n_iter_inner=10)
  print(f"  Nested: MAE ₹{rf_result['MAE']:,.0f}  R² {rf_result['R2']:.3f}")


if __name__ == "__main__":
  main()