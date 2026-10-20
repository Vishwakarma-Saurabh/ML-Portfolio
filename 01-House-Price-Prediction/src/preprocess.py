
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

# Column groups
NUMERIC_COLS = ["area", "bedrooms", "bathrooms", "stories", "parking"]
BINARY_COLS = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
ORDINAL_COLS = ["furnishingstatus"]
TARGET = ["price"]

FURNISHING_ORDER = [["unfurnished", "semi-furnished", "furnished"]]

def build_preprocessor() -> ColumnTransformer:
    # Pipeline Creation
    numeric_pipe = Pipeline([("scaler", StandardScaler())])
    binary_pipe = Pipeline([("ohe", OneHotEncoder(drop="if_binary", handle_unknown="ignore"))])
    ordinal_pipe = Pipeline([("ord", OrdinalEncoder(categories=FURNISHING_ORDER, handle_unknown="use_encoded_value", unknown_value=-1))])

    return ColumnTransformer(
        transformers=[
        ("num", numeric_pipe, NUMERIC_COLS),
        ("bin", binary_pipe, BINARY_COLS),
        ("ord", ordinal_pipe, ORDINAL_COLS),
    ], remainder="drop", verbose_feature_names_out=False,)