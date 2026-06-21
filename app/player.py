import asyncio
import pandas as pd
import joblib
import os
from datetime import datetime
from app.predictor import engineer_features
from app.explainer import get_nurse_explanation
from app.scheduler import broadcast
from app.database import save_prediction

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
feature_names = joblib.load(os.path.join(BASE_DIR, 'data', 'feature_names.pkl'))

# Player state
player_state = {
    "active": False,
    "paused": False,
    "patient_id": None,
    "rows": [],
    "current_index": 0,
    "total_rows": 0,
    "interval_seconds": 5,
    "task": None
}

def load_csv_for_player(filepath: str, patient_id_override: str = None):
    """Load a CSV into the player, ready to tick through"""

    # Block re-loading while a patient is actively playing — prevents
    # accidental duplicate uploads from restarting and polluting the DB
    if player_state["active"] and player_state["current_index"] < player_state["total_rows"]:
        raise RuntimeError(
            f"Player is still running for {player_state['patient_id']} "
            f"(reading {player_state['current_index']} of {player_state['total_rows']}). "
            f"Reset the player before loading a new patient."
        )

    df = pd.read_csv(filepath)
    rows = df.to_dict('records')

    player_state["rows"] = rows
    player_state["current_index"] = 0
    player_state["total_rows"] = len(rows)
    player_state["patient_id"] = patient_id_override or str(rows[0].get('patient_id', 'LIVE-001'))
    player_state["active"] = True
    player_state["paused"] = False

    return player_state["total_rows"]

async def process_one_tick():
    """Process the current row, predict, save, broadcast — then advance"""
    idx = player_state["current_index"]
    if idx >= player_state["total_rows"]:
        player_state["active"] = False
        return None

    row = player_state["rows"][idx]
    pid = player_state["patient_id"]

    patient_dict = dict(row)
    for col in ['patient_id', 'hospital_death', 'name']:
        patient_dict.pop(col, None)

    # Load models
    xgb_model = joblib.load(os.path.join(BASE_DIR, 'models', 'xgboost_model.pkl'))
    optimal_threshold = joblib.load(os.path.join(BASE_DIR, 'models', 'optimal_threshold.pkl'))
    shap_explainer = joblib.load(os.path.join(BASE_DIR, 'models', 'shap_explainer.pkl'))

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
        # Thresholds calibrated from real SHAP value percentiles (~90th / ~75th)
        impact = 'HIGH' if abs_sv > 0.22 else 'MEDIUM' if abs_sv > 0.10 else 'LOW'
        direction = 'increases risk' if sv > 0 else 'decreases risk'
        top_factors.append({
            'feature': f,
            'impact': impact,
            'value': round(float(X[f].values[0]), 2),
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
        risk_score=risk_score, risk_level=risk_level,
        top_factors=[], patient_data=X
    )

    save_prediction(
        patient_id=pid,
        result={
            'risk_score': risk_score, 'risk_level': risk_level, 'alert': alert,
            'recommended_action': action_map[risk_level],
            'top_risk_factors': top_factors
        },
        vitals=vitals,
        source='live_player'
    )

    payload = {
        "type": "patient_update",
        "patient_id": pid,
        "name": pid,
        "age": int(vitals['age']),
        "timestamp": datetime.now().strftime('%H:%M:%S'),
        "risk_score": round(risk_score, 4),
        "risk_level": risk_level,
        "alert": alert,
        "recommended_action": action_map[risk_level],
        "nurse_summary": nurse_sum,
        "top_risk_factors": top_factors,
        "vitals": vitals,
        "reading_number": idx + 1,
        "total_readings": player_state["total_rows"]
    }

    await broadcast(payload)

    player_state["current_index"] += 1
    if player_state["current_index"] >= player_state["total_rows"]:
        player_state["active"] = False

    return payload

async def run_player_loop():
    """Background loop — ticks every interval_seconds while active"""
    while True:
        if player_state["active"] and not player_state["paused"]:
            await process_one_tick()
        await asyncio.sleep(player_state["interval_seconds"])

def pause_player():
    player_state["paused"] = True

def resume_player():
    player_state["paused"] = False

def reset_player():
    player_state["active"] = False
    player_state["paused"] = False
    player_state["current_index"] = 0
    player_state["rows"] = []
    player_state["total_rows"] = 0