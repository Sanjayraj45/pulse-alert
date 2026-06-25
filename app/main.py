from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import os
import json
import io
import asyncio

from fastapi.responses import Response
from app.report import generate_patient_report
from app.schemas import PatientInput, PredictionOutput, HealthCheck
from app.predictor import predict, engineer_features
from app.explainer import get_shap_explanation, get_nurse_explanation
from app.scheduler import connected_clients, latest_results, broadcast
from app.database import (
    save_prediction, get_patient_history,
    get_all_patients, get_recent_alerts,
    get_risk_trend, init_db
)
from app.player import (
    load_csv_for_player, run_player_loop,
    pause_player, resume_player, reset_player, player_state
)

# Initialize FastAPI
app = FastAPI(
    title="PulseAlert 🚨",
    description="ICU Early Warning System",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
feature_names = joblib.load(os.path.join(BASE_DIR, 'data', 'feature_names.pkl'))

@app.on_event("startup")
async def startup_event():
    init_db()
    asyncio.create_task(run_player_loop())
    print("🚨 PulseAlert v4.0 started — Live player mode")

# ─────────────────────────────────────────
# WebSocket
# ─────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        patients = get_all_patients()
        for p in patients:
            await websocket.send_text(json.dumps({
                "type": "patient_update",
                "patient_id": str(p.get("patient_id", "")),
                "name": str(p.get("patient_id", "")),
                "age": p.get("age") or 0,
                "risk_score": float(p.get("latest_risk") or p.get("risk_score") or 0),
                "risk_level": str(p.get("latest_level") or p.get("risk_level") or "LOW"),
                "alert": bool(p.get("latest_alert") or p.get("alert") or False),
                "recommended_action": str(p.get("recommended_action") or ""),
                "nurse_summary": "",
                "top_risk_factors": [],
                "timestamp": str(p.get("latest_time") or p.get("timestamp") or ""),
                "vitals": {
                    "heart_rate": float(p.get("heart_rate") or 0),
                    "systolic_bp": float(p.get("systolic_bp") or 0),
                    "spo2": float(p.get("spo2") or 0),
                    "lactate": float(p.get("lactate") or 0),
                    "resp_rate": float(p.get("resp_rate") or 0),
                    "temperature": float(p.get("temperature") or 0)
                }
            }, default=str))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.discard(websocket)

# ─────────────────────────────────────────
# Health
# ─────────────────────────────────────────

@app.get("/health", response_model=HealthCheck)
def health_check():
    return HealthCheck(status="healthy", model="XGBoost v4.0", version="4.0.0")

@app.get("/")
def root():
    return {"message": "PulseAlert v4.0", "status": "running", "docs": "/docs"}

# ─────────────────────────────────────────
# Single prediction
# ─────────────────────────────────────────

@app.post("/predict", response_model=PredictionOutput)
def predict_risk(patient: PatientInput):
    try:
        result = predict(patient)
        patient_dict = patient.dict()
        patient_id = patient_dict.pop('patient_id')
        X = engineer_features(patient_dict)
        vitals = {
            'heart_rate': float(X['d1_heartrate_max'].values[0]),
            'systolic_bp': float(X['d1_sysbp_min'].values[0]),
            'spo2': float(X['d1_spo2_min'].values[0]),
            'lactate': float(X['d1_lactate_max'].values[0]),
            'resp_rate': float(X['d1_resprate_max'].values[0]),
            'temperature': float(X['d1_temp_max'].values[0]),
            'age': float(X['age'].values[0])
        }
        save_prediction(
            patient_id=patient_id,
            result={
                'risk_score': result.risk_score,
                'risk_level': result.risk_level,
                'alert': result.alert,
                'recommended_action': result.recommended_action,
                'top_risk_factors': [
                    {'feature': f.feature, 'impact': f.impact, 'value': f.value}
                    for f in result.top_risk_factors
                ]
            },
            vitals=vitals,
            source='manual'
        )
        return result
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

# ─────────────────────────────────────────
# CSV Batch Upload (one-shot, all patients at once)
# ─────────────────────────────────────────

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")

        xgb_model = joblib.load(os.path.join(BASE_DIR, 'models', 'xgboost_model.pkl'))
        optimal_threshold = joblib.load(os.path.join(BASE_DIR, 'models', 'optimal_threshold.pkl'))
        shap_explainer = joblib.load(os.path.join(BASE_DIR, 'models', 'shap_explainer.pkl'))

        results = []
        errors = []
        broadcast_patients = []

        for idx, row in df.iterrows():
            try:
                pid = str(row.get('patient_id', f'P-{idx+1}'))
                patient_dict = row.to_dict()

                for col in ['patient_id', 'hospital_death', 'name']:
                    patient_dict.pop(col, None)

                X = engineer_features(patient_dict)

                risk_score = float(xgb_model.predict_proba(X)[:, 1][0])
                alert = risk_score >= optimal_threshold

                if risk_score >= 0.75:
                    risk_level = "CRITICAL"
                elif risk_score >= 0.50:
                    risk_level = "HIGH"
                elif risk_score >= 0.35:
                    risk_level = "MEDIUM"
                else:
                    risk_level = "LOW"

                shap_values = shap_explainer.shap_values(X)
                shap_series = pd.Series(shap_values[0], index=feature_names)
                top_shap = shap_series.abs().sort_values(ascending=False).head(5)
                top_factors = []
                for f in top_shap.index:
                    sv = shap_series[f]
                    abs_sv = abs(sv)
                    impact = 'HIGH' if abs_sv > 0.22 else 'MEDIUM' if abs_sv > 0.10 else 'LOW'
                    direction = 'increases risk' if sv > 0 else 'decreases risk'
                    top_factors.append({
                        'feature': f,
                        'impact': impact,
                        'value': round(float(X[f].values[0]), 2),
                        'shap_value': round(float(sv), 4),
                        'direction': direction
                    })

                action_map = {
                    "CRITICAL": "Notify attending physician immediately.",
                    "HIGH": "Alert charge nurse within 30 minutes.",
                    "MEDIUM": "Increase monitoring frequency.",
                    "LOW": "Continue routine monitoring."
                }

                vitals = {
                    'heart_rate': round(float(X['d1_heartrate_max'].values[0]), 1),
                    'systolic_bp': round(float(X['d1_sysbp_min'].values[0]), 1),
                    'spo2': round(float(X['d1_spo2_min'].values[0]), 1),
                    'lactate': round(float(X['d1_lactate_max'].values[0]), 1),
                    'resp_rate': round(float(X['d1_resprate_max'].values[0]), 1),
                    'temperature': round(float(X['d1_temp_max'].values[0]), 1),
                    'age': round(float(X['age'].values[0]), 0)
                }

                nurse_sum = get_nurse_explanation(
                    risk_score=risk_score,
                    risk_level=risk_level,
                    top_factors=[],
                    patient_data=X
                )

                result = {
                    'patient_id': pid,
                    'risk_score': round(risk_score, 4),
                    'risk_level': risk_level,
                    'alert': alert,
                    'top_factors': top_factors,
                    'recommended_action': action_map[risk_level],
                    'nurse_summary': nurse_sum,
                    'vitals': vitals
                }

                save_prediction(
                    patient_id=pid,
                    result={
                        'risk_score': risk_score,
                        'risk_level': risk_level,
                        'alert': alert,
                        'recommended_action': action_map[risk_level],
                        'top_risk_factors': [
                            {'feature': f['feature'], 'impact': f['impact'], 'value': f['value'], 'direction': f['direction']}
                            for f in top_factors
                        ]
                    },
                    vitals=vitals,
                    source='csv_upload'
                )

                results.append(result)

                broadcast_patients.append({
                    "type": "patient_update",
                    "patient_id": pid,
                    "name": pid,
                    "age": int(vitals['age']),
                    "timestamp": __import__('datetime').datetime.now().strftime('%H:%M:%S'),
                    "risk_score": round(risk_score, 4),
                    "risk_level": risk_level,
                    "alert": alert,
                    "recommended_action": action_map[risk_level],
                    "nurse_summary": nurse_sum,
                    "top_risk_factors": [
                        {'feature': f['feature'], 'impact': f['impact'], 'value': f['value'], 'direction': f['direction']}
                        for f in top_factors[:5]
                    ],
                    "vitals": vitals
                })

            except Exception as row_error:
                errors.append({'row': idx + 1, 'error': str(row_error)})

        for payload in broadcast_patients:
            await broadcast(payload)

        total = len(results)
        alerts = sum(1 for r in results if r['alert'])
        critical = sum(1 for r in results if r['risk_level'] == 'CRITICAL')
        high = sum(1 for r in results if r['risk_level'] == 'HIGH')
        medium = sum(1 for r in results if r['risk_level'] == 'MEDIUM')
        low = sum(1 for r in results if r['risk_level'] == 'LOW')

        return {
            "file": file.filename,
            "total_processed": total,
            "total_errors": len(errors),
            "summary": {
                "alerts_fired": alerts,
                "critical": critical,
                "high": high,
                "medium": medium,
                "low": low
            },
            "results": results,
            "errors": errors
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────
# Live Player Controls
# ─────────────────────────────────────────

@app.post("/player/load")
async def player_load(file: UploadFile = File(...)):
    """Load a multi-row CSV for one patient into the live player"""
    try:
        contents = await file.read()
        temp_path = os.path.join(BASE_DIR, 'data', '_player_upload.csv')
        with open(temp_path, 'wb') as f:
            f.write(contents)

        total_rows = load_csv_for_player(temp_path)

        return {
            "status": "loaded",
            "patient_id": player_state["patient_id"],
            "total_readings": total_rows,
            "interval_seconds": player_state["interval_seconds"]
        }
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/player/pause")
def player_pause():
    pause_player()
    return {"status": "paused"}

@app.post("/player/resume")
def player_resume():
    resume_player()
    return {"status": "resumed"}

@app.post("/player/reset")
def player_reset():
    reset_player()
    return {"status": "reset"}


# ─────────────────────────────────────────
# Demo mode — server-side CSV loading
# ─────────────────────────────────────────

@app.post("/demo/start")
def demo_start(scenario: str = "deterioration"):
    """Load a pre-built demo scenario without needing a CSV upload"""
    scenarios = {
        "recovery":     "data/demo_recovery.csv",
        "deterioration":"data/demo_deterioration.csv",
        "crash":        "data/demo_crash.csv"
    }
    if scenario not in scenarios:
        raise HTTPException(status_code=400, detail=f"Unknown scenario: {scenario}")

    filepath = os.path.join(BASE_DIR, scenarios[scenario])
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Demo file not found: {filepath}")

    try:
        total_rows = load_csv_for_player(filepath)
        return {
            "status": "loaded",
            "scenario": scenario,
            "patient_id": player_state["patient_id"],
            "total_readings": total_rows,
            "interval_seconds": player_state["interval_seconds"]
        }
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear/patients")
def clear_all_patients():
    """Wipe all patient data from DB for a clean demo"""
    try:
        conn = __import__('app.database', fromlist=['get_connection']).get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        cursor.execute("DELETE FROM patients")
        conn.commit()
        cursor.close()
        conn.close()
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/player/status")
def player_status():
    return {
        "active": player_state["active"],
        "paused": player_state["paused"],
        "patient_id": player_state["patient_id"],
        "current_index": player_state["current_index"],
        "total_rows": player_state["total_rows"]
    }

# ─────────────────────────────────────────
# Database endpoints
# ─────────────────────────────────────────

@app.get("/patients")
def get_patients():
    try:
        patients = get_all_patients()
        return {"total": len(patients), "patients": patients}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patients/{patient_id}/history")
def patient_history(patient_id: str):
    try:
        history = get_patient_history(patient_id)
        return {"patient_id": patient_id, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patients/{patient_id}/trend")
def patient_trend(patient_id: str):
    try:
        trend = get_risk_trend(patient_id)
        return {"patient_id": patient_id, "trend": trend}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts")
def recent_alerts():
    try:
        alerts = get_recent_alerts()
        return {"total": len(alerts), "alerts": alerts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/report/{patient_id}")
def download_report(patient_id: str):
    """Generate and download a PDF report for a patient"""
    try:
        pdf_bytes = generate_patient_report(patient_id)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=PulseAlert_Report_{patient_id}.pdf"
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))