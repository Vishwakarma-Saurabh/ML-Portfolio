
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

NUMERIC_COLS = ["area", "bedrooms", "bathrooms", "stories", "parking", "total_rooms", "area_per_room", "amenities", "area_x_prefarea", "bathrooms_per_bedroom"]
BINARY_COLS = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
ORDINAL_COLS = ["furnishingstatus"]
TARGET = ["price"]

FURNISHING_ORDER = [["unfurnished", "semi-furnished", "furnished"]]

def build_preprocessor(present_cols: list[str]):
    num   = [c for c in NUMERIC_COLS if c in present_cols]
    bin_  = [c for c in BINARY_COLS  if c in present_cols]
    ord_  = [c for c in ORDINAL_COLS if c in present_cols]

    numeric_pipe = Pipeline([("scaler", StandardScaler())])
    binary_pipe  = Pipeline([("ohe", OneHotEncoder(drop="if_binary", handle_unknown="ignore"))])
    ordinal_pipe = Pipeline([("ord", OrdinalEncoder(categories=FURNISHING_ORDER, handle_unknown="use_encoded_value", unknown_value=-1))])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num),
            ("bin", binary_pipe,  bin_),
            ("ord", ordinal_pipe, ord_),
        ], remainder="drop", verbose_feature_names_out=False,
    )