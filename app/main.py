from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import joblib
import os
import json

from app.schemas import PatientInput, PredictionOutput, HealthCheck
from app.predictor import predict, engineer_features
from app.explainer import get_shap_explanation, get_nurse_explanation
from app.scheduler import start_scheduler, connected_clients, latest_results

# Initialize FastAPI app
app = FastAPI(
    title="PulseAlert 🚨",
    description="ICU Early Warning System — Real-time patient deterioration monitoring",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load feature names
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
feature_names = joblib.load(os.path.join(BASE_DIR, 'data', 'feature_names.pkl'))

# Start scheduler on startup
@app.on_event("startup")
async def startup_event():
    start_scheduler()
    print("🚨 PulseAlert v2.0 started — Real-time ICU monitoring active")

# ─────────────────────────────────────────
# WebSocket endpoint
# ─────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    print(f"📡 Dashboard connected — {len(connected_clients)} client(s) active")

    try:
        # Send current state immediately on connect
        if latest_results:
            for pid, result in latest_results.items():
                await websocket.send_text(json.dumps(result))

        # Keep connection alive
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        connected_clients.discard(websocket)
        print(f"📡 Dashboard disconnected — {len(connected_clients)} client(s) active")

# ─────────────────────────────────────────
# REST endpoints
# ─────────────────────────────────────────

@app.get("/health", response_model=HealthCheck)
def health_check():
    return HealthCheck(
        status="healthy",
        model="XGBoost v2.0",
        version="2.0.0"
    )

@app.get("/patients")
def get_all_patients():
    """Get latest results for all monitored patients"""
    return {
        "total": len(latest_results),
        "patients": list(latest_results.values())
    }

@app.get("/patients/{patient_id}")
def get_patient(patient_id: str):
    """Get latest result for a specific patient"""
    if patient_id not in latest_results:
        raise HTTPException(status_code=404, detail="Patient not found")
    return latest_results[patient_id]

@app.post("/predict", response_model=PredictionOutput)
def predict_risk(patient: PatientInput):
    try:
        result = predict(patient)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
def explain_prediction(patient: PatientInput):
    try:
        patient_dict = patient.dict()
        patient_dict.pop('patient_id')
        X = engineer_features(patient_dict)
        explanation = get_shap_explanation(X)
        return {"patient_id": patient.patient_id, "explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summary")
def nurse_summary(patient: PatientInput):
    try:
        prediction = predict(patient)
        patient_dict = patient.dict()
        patient_dict.pop('patient_id')
        X = engineer_features(patient_dict)
        summary = get_nurse_explanation(
            risk_score=prediction.risk_score,
            risk_level=prediction.risk_level,
            top_factors=prediction.top_risk_factors,
            patient_data=X
        )
        return {
            "patient_id": patient.patient_id,
            "risk_score": prediction.risk_score,
            "risk_level": prediction.risk_level,
            "alert": prediction.alert,
            "nurse_summary": summary,
            "recommended_action": prediction.recommended_action,
            "top_risk_factors": prediction.top_risk_factors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {
        "message": "PulseAlert ICU Early Warning System v2.0",
        "status": "running",
        "monitoring": f"{len(latest_results)} patients",
        "docs": "/docs",
        "websocket": "/ws"
    }