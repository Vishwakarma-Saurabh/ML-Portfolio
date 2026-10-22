
import json
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocess import NUMERIC_COLS, BINARY_COLS
from src.features import FEATURE_BUILDERS
from src.train import apply_features, load_data


RF_PARAMS = dict(
    n_estimators=329,
    max_depth=19,
    max_features=0.5,
    min_samples_leaf=1,
    min_samples_split=3,
    random_state=42,
    n_jobs=-1,
)


FEATURE_NAMES = list(FEATURE_BUILDERS.keys())


def build_final_pipeline():

    prep = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("bin", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), BINARY_COLS),
            ("furn", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["furnishingstatus"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline([
        ("prep",  prep),
        ("model", TransformedTargetRegressor(
            regressor=RandomForestRegressor(**RF_PARAMS),
            func=np.log1p, inverse_func=np.expm1,
        )),
    ])


def cv_evaluate(pipe, X, y, cv):
    y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
    return {
        "MAE":  mean_absolute_error(y, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
        "R2":   r2_score(y, y_pred),
    }


def main():
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)


    df = apply_features(load_data(), FEATURE_NAMES)
    X = df.drop(columns=["price"])
    y = df["price"]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics = cv_evaluate(build_final_pipeline(), X, y, cv)
    print(f"FINAL CV   MAE ₹{metrics['MAE']:,.0f}  "
          f"RMSE ₹{metrics['RMSE']:,.0f}  R² {metrics['R2']:.3f}")

    # ---- Fit on ALL data ----
    final_pipeline = build_final_pipeline()
    final_pipeline.fit(X, y)
    joblib.dump(final_pipeline, "models/house_price_pipeline.joblib")
    print("Saved → models/house_price_pipeline.joblib")


    metadata = {
        "target": "price (INR)",
        "metrics_cv": metrics,
        "features_input": {
            "numeric":    NUMERIC_COLS,
            "binary":     BINARY_COLS,
            "categorical": ["furnishingstatus"],
        },
        "engineered_features": FEATURE_NAMES,
        "raw_input_schema": {
            "area": "float (>0)",
            "bedrooms": "int (>0)",
            "bathrooms": "int (>0)",
            "stories": "int (>0)",
            "mainroad": "yes|no",
            "guestroom": "yes|no",
            "basement": "yes|no",
            "hotwaterheating": "yes|no",
            "airconditioning": "yes|no",
            "parking": "int (>=0)",
            "prefarea": "yes|no",
            "furnishingstatus": "unfurnished|semi-furnished|furnished",
        },
        "notes": (
            "Predictions are log1p-trained / expm1-inverted; on average they "
            "under-predict mean by ~2.15% due to Jensen bias. Apply the "
            "smearing factor 1.0215 when calibrated means matter; NOT when "
            "MAE is the reporting metric."
        ),
    }
    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved → models/metadata.json")

    with open("reports/final_cv_report.txt", "w") as f:
        f.write(f"Final pipeline CV metrics\n")
        f.write(f"MAE  : {metrics['MAE']:,.0f} INR\n")
        f.write(f"RMSE : {metrics['RMSE']:,.0f} INR\n")
        f.write(f"R2   : {metrics['R2']:.4f}\n")
        f.write(f"\nModel: RandomForestRegressor {RF_PARAMS}\n")
        f.write(f"Features: numeric={NUMERIC_COLS} binary={BINARY_COLS} "
                f"categorical=furnishingstatus\n")
        f.write(f"Engineered: {FEATURE_NAMES}\n")


if __name__ == "__main__":
    main()