
import numpy as np
import pandas as pd
from scipy.stats import randint, loguniform
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.compose import TransformedTargetRegressor

from .preprocess import build_preprocessor, TARGET
from .features import FEATURE_BUILDERS
from .train import apply_features, load_data


def build_pipeline(present_cols, regressor, log_target=True):
    est = TransformedTargetRegressor(
        regressor=regressor, func=np.log1p, inverse_func=np.expm1
    ) if log_target else regressor
    return Pipeline([
        ("prep",  build_preprocessor(present_cols)),
        ("model", est),
    ])


def tune_rf(X, y, cv):
    pipe = build_pipeline(list(X.columns), RandomForestRegressor(random_state=42, n_jobs=-1))
    param_dist = {
        "model__regressor__n_estimators":     randint(200, 600),
        "model__regressor__max_depth":        randint(3, 25),
        "model__regressor__min_samples_leaf": randint(1, 10),
        "model__regressor__min_samples_split":randint(2, 20),
        "model__regressor__max_features":     ["sqrt", "log2", 0.5, 0.75],
    }
    search = RandomizedSearchCV(
        pipe, param_dist, n_iter=40, cv=cv,
        scoring=make_scorer(mean_absolute_error, greater_is_better=False),
        random_state=42, n_jobs=-1, verbose=1,
    )
    search.fit(X, y)
    return search


def tune_gb(X, y, cv):
    pipe = build_pipeline(list(X.columns), GradientBoostingRegressor(random_state=42))
    param_dist = {
        "model__regressor__n_estimators":  randint(100, 600),
        "model__regressor__max_depth":     randint(2, 6),
        "model__regressor__learning_rate": loguniform(1e-3, 0.3),
        "model__regressor__subsample":     [0.6, 0.7, 0.8, 0.9, 1.0],
        "model__regressor__min_samples_leaf": randint(1, 10),
    }
    search = RandomizedSearchCV(
        pipe, param_dist, n_iter=40, cv=cv,
        scoring=make_scorer(mean_absolute_error, greater_is_better=False),
        random_state=42, n_jobs=-1, verbose=1,
    )
    search.fit(X, y)
    return search


def run_tuning():
    df = load_data()
    df = apply_features(df, list(FEATURE_BUILDERS.keys())) 
    X, y = df.drop(columns=TARGET), df[TARGET]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    print("═══ RandomForest tuning ═══")
    rf_search = tune_rf(X, y, cv)
    print(f"Best params: {rf_search.best_params_}")
    print(f"Best CV MAE: ₹{-rf_search.best_score_:,.0f}")

    print("\n═══ GradientBoosting tuning ═══")
    gb_search = tune_gb(X, y, cv)
    print(f"Best params: {gb_search.best_params_}")
    print(f"Best CV MAE: ₹{-gb_search.best_score_:,.0f}")

    pd.DataFrame([
        {"model": "RandomForest",     "cv_mae": -rf_search.best_score_, "params": rf_search.best_params_},
        {"model": "GradientBoosting", "cv_mae": -gb_search.best_score_, "params": gb_search.best_params_},
    ]).to_csv("reports/phase4a_tuning.csv", index=False)
    print("\nSaved → reports/phase4a_tuning.csv")


if __name__ == "__main__":
    run_tuning()