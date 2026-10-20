
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import TransformedTargetRegressor

from .preprocess import build_preprocessor, TARGET


def load_data(path: str = "data/Housing_dataset.csv") -> pd.DataFrame:
    return pd.read_csv(path)

def get_models() -> dict:
    alphas = np.logspace(-2, 4, 50)
    return {
        "Ridge": RidgeCV(alphas=alphas),
        "Lasso": LassoCV(alphas=alphas, max_iter=10000, n_jobs=-1),
        "ElasticNet": ElasticNetCV(alphas=alphas, l1_ratio=[0.1, 0.5, 0.9, 0.95, 1.0], max_iter=10000, n_jobs=-1),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }

def evaluate_cv(model, X, y, cv, log_target: bool) -> dict:
    estimator = (
        TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1) if log_target else model
    )
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", estimator),
    ])

    y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
    return {
        "MAE": mean_absolute_error(y, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
        "R2": r2_score(y, y_pred)
    }

def run_phase2():
    df = load_data()
    X = df.drop(columns=TARGET)
    y = df[TARGET]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    for target_name, log_target in [("price (raw)", False), ("log1p(price)", True)]:
        for model_name, model in get_models().items():
            metrics = evaluate_cv(model, X, y, cv, log_target)
            rows.append({
                "target": target_name,
                "model": model_name,
                **metrics
            })

            print(f"[{target_name:14s}] {model_name:16s} "
                  f"MAE ₹{metrics['MAE']:>10,.0f}  "
                  f"RMSE ₹{metrics['RMSE']:>10,.0f}  "
                  f"R² {metrics['R2']:.3f}")

    results = pd.DataFrame(rows).sort_values(["target", "MAE"])
    results.to_csv("reports/phase2_cv_results.csv", index=False)
    print("\nSaved → reports/phase2_cv_results.csv")
    return results


if __name__ == "__main__":
    run_phase2()           
