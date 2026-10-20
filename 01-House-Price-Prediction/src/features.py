import pandas as pd


AMENITY_COLS = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]


def add_total_rooms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    return df


def add_area_per_room(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "total_rooms" not in df.columns:
        df = add_total_rooms(df)
    denom = df["total_rooms"].clip(lower=1)
    df["area_per_room"] = df["area"] / denom
    return df


def add_amenities(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["amenities"] = (df[AMENITY_COLS] == "yes").sum(axis=1)
    return df


def add_area_x_prefarea(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    prefarea_flag = (df["prefarea"] == "yes").astype(int)
    df["area_x_prefarea"] = df["area"] * prefarea_flag
    return df


def add_bathrooms_per_bedroom(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bathrooms_per_bedroom"] = df["bathrooms"] / df["bedrooms"].clip(lower=1)
    return df


FEATURE_BUILDERS = {
    "F1_total_rooms":       add_total_rooms,
    "F2_area_per_room":     add_area_per_room,   
    "F3_amenities":         add_amenities,
    "F4_area_x_prefarea":   add_area_x_prefarea,
    "F5_bath_per_bedroom":  add_bathrooms_per_bedroom,
}