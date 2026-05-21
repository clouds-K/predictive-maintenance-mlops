"""
API FastAPI pour la prédiction de pannes industrielles.
Auteur: Khouloud
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(
    title="Predictive Maintenance API",
    description="API MLOps - Prédiction de pannes AI4I 2020",
    version="1.0.0",
)

MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"

model = None
scaler = None


@app.on_event("startup")
def load_model() -> None:
    """Charge le modèle et le scaler au démarrage de l'API."""
    global model, scaler
    if not Path(MODEL_PATH).exists():
        raise RuntimeError(f"Modèle introuvable : {MODEL_PATH}")
    if not Path(SCALER_PATH).exists():
        raise RuntimeError(f"Scaler introuvable : {SCALER_PATH}")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Modèle et scaler chargés avec succès")


class PredictionRequest(BaseModel):
    """Schéma de la requête de prédiction."""

    type_machine: int = Field(
        ..., ge=0, le=2, description="Type machine: H=0, L=1, M=2"
    )
    air_temperature: float = Field(..., description="Température air (K)")
    process_temperature: float = Field(..., description="Température process (K)")
    rotational_speed: float = Field(..., description="Vitesse rotation (rpm)")
    torque: float = Field(..., description="Couple (Nm)")
    tool_wear: float = Field(..., description="Usure outil (min)")

    class Config:
        json_schema_extra = {
            "example": {
                "type_machine": 2,
                "air_temperature": 298.1,
                "process_temperature": 308.6,
                "rotational_speed": 1551.0,
                "torque": 42.8,
                "tool_wear": 0.0,
            }
        }


class PredictionResponse(BaseModel):
    """Schéma de la réponse de prédiction."""

    prediction: int
    probability_failure: float
    status: str
    message: str


@app.get("/")
def root():
    """Endpoint racine."""
    return {
        "message": "Predictive Maintenance API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
def health_check():
    """Vérifie l'état de santé de l'API."""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {"status": "healthy", "model": "XGBoost", "dataset": "AI4I 2020"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Prédit si une machine va tomber en panne.

    Args:
        request: Données capteurs de la machine.

    Returns:
        Prédiction avec probabilité de panne.
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    features = np.array(
        [
            [
                request.type_machine,
                request.air_temperature,
                request.process_temperature,
                request.rotational_speed,
                request.torque,
                request.tool_wear,
            ]
        ]
    )

    features_scaled = scaler.transform(features)
    prediction = int(model.predict(features_scaled)[0])
    probability = float(model.predict_proba(features_scaled)[0][1])

    status = "PANNE DÉTECTÉE" if prediction == 1 else "NORMAL"
    message = (
        "⚠️ Intervention recommandée immédiatement !"
        if prediction == 1
        else "✅ Machine en bon état de fonctionnement"
    )

    return PredictionResponse(
        prediction=prediction,
        probability_failure=round(probability, 4),
        status=status,
        message=message,
    )
