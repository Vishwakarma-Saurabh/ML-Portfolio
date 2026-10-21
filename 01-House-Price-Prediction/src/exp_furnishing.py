
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error

from .preprocess import NUMERIC_COLS, BINARY_COLS
from .features import FEATURE_BUILDERS
from .train import apply_features, load_data

def build_pipeline_furnishing(mode: str):
    furn_enc = (
        OrdinalEncoder(categories=[["unfurnished", "semi-furnished", "furnished"]],
                       handle_unknown="use_encoded_value", unknown_value=-1)
        if mode == "ordinal"
        else OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        ("bin", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), BINARY_COLS),
        ("ord", furn_enc, ["furnishingstatus"]),
    ], remainder="drop", verbose_feature_names_out=False)

    return Pipeline([
        ("prep",  prep),
        ("model", TransformedTargetRegressor(
            regressor=RidgeCV(alphas=np.logspace(-2, 4, 50)),
            func=np.log1p, inverse_func=np.expm1,
        )),
    ])

def run():
    df = apply_features(load_data(), list(FEATURE_BUILDERS.keys()))
    X, y = df.drop(columns=["price"]), df["price"]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for mode in ["ordinal", "onehot"]:
        pipe = build_pipeline_furnishing(mode)
        y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
        mae = mean_absolute_error(y, y_pred)
        print(f"{mode:8s} CV MAE: ₹{mae:,.0f}")

if __name__ == "__main__":
    run()