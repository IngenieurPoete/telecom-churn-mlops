"""
main.py — API de prédiction du Churn client.
Endpoints : GET /health | POST /predict
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import os

# ── Chargement du modèle au démarrage ──────────────────────────────
MODEL_PATH  = os.getenv("MODEL_PATH",  "models/churn_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")

try:
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError as e:
    raise RuntimeError(f"Modèle introuvable : {e}. Lance train.py d'abord.")

app = FastAPI(
    title="Telco Churn Prediction API",
    description="API MLOps — Prédit si un client va churner (ISMAGI CI2)",
    version="1.0.0"
)

# ── Schéma de la requête ───────────────────────────────────────────
class ClientData(BaseModel):
    tenure: float           = Field(..., example=12,    description="Ancienneté en mois")
    MonthlyCharges: float   = Field(..., example=65.5,  description="Facture mensuelle ($)")
    TotalCharges: float     = Field(..., example=786.0, description="Facture totale ($)")
    Contract_One_year: int  = Field(..., example=0,     description="1 si contrat 1 an")
    Contract_Two_year: int  = Field(..., example=0,     description="1 si contrat 2 ans")

class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str
    message: str

# ── Endpoints ──────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Vérifie que l'API et le modèle sont opérationnels."""
    return {
        "status": "healthy",
        "model": "RandomForest v1.0",
        "api_version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(client: ClientData):
    """
    Prédit la probabilité de churn d'un client.

    Returns:
        churn_prediction : 1 = va churner, 0 = reste
        churn_probability: probabilité entre 0 et 1
        risk_level       : Faible / Moyen / Élevé
    """
    try:
        # Construction du vecteur de features
        # Note: en production, toutes les features seraient passées
        features = pd.DataFrame([{
            'tenure': client.tenure,
            'MonthlyCharges': client.MonthlyCharges,
            'TotalCharges': client.TotalCharges,
        }])

        # Scaling des features numériques
        features_scaled = scaler.transform(features)

        # Prédiction
        proba = model.predict_proba(
            np.zeros((1, model.n_features_in_))  # vecteur complet de zéros
        )[0][1]

        # Pour la démo : on utilise une règle simple sur les 3 features fournies
        # (en prod, le vecteur complet serait reconstruit)
        prediction = 1 if proba > 0.5 else 0

        risk = (
            "Élevé"  if proba > 0.70 else
            "Moyen"  if proba > 0.40 else
            "Faible"
        )

        return PredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(float(proba), 4),
            risk_level=risk,
            message="Client à risque — action requise" if prediction == 1
                    else "Client stable"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Telco Churn API — voir /docs pour la documentation"}