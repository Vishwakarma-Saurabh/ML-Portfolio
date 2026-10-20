
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import TransformedTargetRegressor

from .preprocess import build_preprocessor, TARGET
from .features import FEATURE_BUILDERS


def load_data(path="data/Housing_dataset.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def apply_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    out = df.copy()
    for name in feature_names:
        out = FEATURE_BUILDERS[name](out)
    return out


def evaluate(model, X, y, cv, log_target: bool) -> dict:
    est = (TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1) if log_target else model)
    pipe = Pipeline([
        ("prep", build_preprocessor(list(X.columns))),
        ("model", est),
    ])
    y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
    return {
        "MAE":  mean_absolute_error(y, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
        "R2":   r2_score(y, y_pred),
    }


def run_ablation():
    df = load_data()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 50))

    experiments = [
        ("E0_baseline",           []),
        ("E1_F1_total_rooms",     ["F1_total_rooms"]),
        ("E2_F2_area_per_room",   ["F2_area_per_room"]),    
        ("E3_F3_amenities",       ["F3_amenities"]),
        ("E4_F4_area_x_prefarea", ["F4_area_x_prefarea"]),
        ("E5_F5_bath_per_bed",    ["F5_bath_per_bedroom"]),
        ("E6_all_features",       list(FEATURE_BUILDERS.keys())),
    ]

    rows = []
    for exp_name, feats in experiments:
        df_exp = apply_features(df, feats)
        X, y = df_exp.drop(columns=TARGET), df_exp[TARGET]
        m = evaluate(ridge, X, y, cv, log_target=True)
        rows.append({"experiment": exp_name, **m})
        print(f"{exp_name:26s} MAE ₹{m['MAE']:>9,.0f}  RMSE ₹{m['RMSE']:>9,.0f}  R² {m['R2']:.3f}")

    baseline_mae = rows[0]["MAE"]

    print("\n— Delta vs E0 —")
    for r in rows[1:]:
        delta_pct = 100 * (baseline_mae - r["MAE"]) / baseline_mae
        marker = "✅" if delta_pct >= 1.0 else "⚠️"
        print(f"{r['experiment']:26s} ΔMAE {delta_pct:+5.2f}%  {marker}")

    print("\n— Trees on all engineered features (log1p target) —")
    df_all = apply_features(df, list(FEATURE_BUILDERS.keys()))
    X_all, y_all = df_all.drop(columns=TARGET), df_all[TARGET]
    for name, mdl in {
        "RandomForest":     RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }.items():
        m = evaluate(mdl, X_all, y_all, cv, log_target=True)
        print(f"{name:16s} MAE ₹{m['MAE']:>9,.0f}  RMSE ₹{m['RMSE']:>9,.0f}  R² {m['R2']:.3f}")

    pd.DataFrame(rows).to_csv("reports/phase3_summary.csv", index=False)
    print("\nSaved → reports/phase3_ablation.csv")


if __name__ == "__main__":
    run_ablation()