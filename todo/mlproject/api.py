"""API d'inference d'un modele de classification (FastAPI).

Seance 12 - TP FastAPI
    Expose le modele entraine (models/model.joblib) via une API HTTP.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlproject.config import MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ml: dict = {}


# TODO (S12-3) : chargement du modele une seule fois au demarrage
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    ml["model"] = joblib.load(MODEL_DIR / "model.joblib")
    logger.info("Modele charge depuis %s", MODEL_DIR / "model.joblib")
    yield
    ml.clear()


app = FastAPI(title="Classification API - Fraude CB", version="0.1.0", lifespan=lifespan)


# TODO (S12-1) : schema d'entree = colonnes de config.py (Time, Amount, V1..V28)
class Features(BaseModel):
    Time: float = Field(..., ge=0, description="Secondes ecoulees depuis la 1re transaction")
    Amount: float = Field(..., ge=0, description="Montant de la transaction")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Time": 0.0, "Amount": 149.62,
                    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
                    "V6": 0.46, "V7": 0.24, "V8": 0.1, "V9": 0.36, "V10": 0.09,
                    "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
                    "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.4, "V20": 0.25,
                    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13,
                    "V26": -0.19, "V27": 0.13, "V28": -0.02,
                }
            ]
        }
    }


# TODO (S12-2) : schema de sortie
class PredictionOut(BaseModel):
    prediction: int
    probability: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# TODO (S12-4) : endpoint de prediction
@app.post("/predict", response_model=PredictionOut)
def predict(features: Features) -> PredictionOut:
    model = ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    row = pd.DataFrame([features.model_dump()])
    proba = float(model.predict_proba(row)[0, 1])
    return PredictionOut(prediction=int(proba >= 0.5), probability=round(proba, 4))


# TODO (S12-5 bonus) : info sur la version du modele servi
@app.get("/model-info")
def model_info() -> dict:
    return {"version": os.environ.get("MODEL_VERSION", "unknown")}
