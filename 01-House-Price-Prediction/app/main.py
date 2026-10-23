
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import FEATURE_BUILDERS

MODEL = joblib.load("models/house_price_pipeline.joblib")
META = json.loads(Path("models/metadata.json").read_text())
SMEARING_FACTOR = 1.0215   # from Phase 4 (exp_smearing.py); documented in metadata

app = FastAPI(
    title="House Price Prediction API",
    version="1.0.0",
    description="Predicts INR price from structural and amenity features.",
)


class House(BaseModel):
    area: float = Field(..., gt=0, description="Total area in sq ft")
    bedrooms: int = Field(..., ge=1, le=10)
    bathrooms: int = Field(..., ge=1, le=10)
    stories: int = Field(..., ge=1, le=10)
    mainroad: Literal["yes", "no"]
    guestroom: Literal["yes", "no"]
    basement: Literal["yes", "no"]
    hotwaterheating: Literal["yes", "no"]
    airconditioning: Literal["yes", "no"]
    parking: int = Field(..., ge=0, le=10)
    prefarea: Literal["yes", "no"]
    furnishingstatus: Literal["unfurnished", "semi-furnished", "furnished"]


class PredictionResponse(BaseModel):
    price_inr: float
    price_inr_bias_corrected: float
    model_version: str


def _to_features(h: House) -> pd.DataFrame:
    row = h.model_dump()
    df = pd.DataFrame([row])
    for feat in FEATURE_BUILDERS:
        df = FEATURE_BUILDERS[feat](df)
    return df


def _predict(df: pd.DataFrame) -> np.ndarray:
    return MODEL.predict(df)  


# ---- Endpoints ----
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(h: House):
    try:
        df = _to_features(h)
        pred = float(_predict(df)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    return PredictionResponse(
        price_inr=pred,
        price_inr_bias_corrected=pred * SMEARING_FACTOR,
        model_version=META.get("version", "1.0.0"),
    )


@app.post("/predict/batch")
def predict_batch(houses: list[House]):
    if not houses:
        raise HTTPException(status_code=400, detail="Empty batch")
    df = pd.concat([_to_features(h) for h in houses], ignore_index=True)
    preds = _predict(df).tolist()
    return {"predictions": [{"price_inr": p,
                             "price_inr_bias_corrected": p * SMEARING_FACTOR}
                            for p in preds]}