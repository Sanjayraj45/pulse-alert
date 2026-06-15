import asyncio
import json
import logging
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from app.simulator import PATIENT_PROFILES, update_patient_vitals
from app.predictor import predict, engineer_features
from app.schemas import PatientInput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store latest results for all patients
latest_results = {}
connected_clients = set()

async def broadcast(message: dict):
    """Send message to all connected WebSocket clients"""
    if connected_clients:
        data = json.dumps(message)
        dead = set()
        for client in connected_clients:
            try:
                await client.send_text(data)
            except Exception:
                dead.add(client)
        connected_clients.difference_update(dead)

async def run_prediction_for_patient(pid: str):
    """Run prediction for one patient and broadcast result"""
    try:
        # Get latest simulated vitals
        vitals = update_patient_vitals(pid)
        profile = PATIENT_PROFILES[pid]

        # Build PatientInput object
        patient = PatientInput(**{
            k: v for k, v in vitals.items()
            if k not in ['name', 'severity', 'timestamp']
        })

        # Run prediction
        result = predict(patient)

        # Build result payload
        payload = {
            "type": "patient_update",
            "patient_id": pid,
            "name": profile["name"],
            "age": profile["age"],
            "severity_label": vitals["severity"],
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "risk_score": result.risk_score,
            "risk_level": result.risk_level,
            "alert": result.alert,
            "recommended_action": result.recommended_action,
            "top_risk_factors": [
                {
                    "feature": f.feature,
                    "impact": f.impact,
                    "value": f.value
                }
                for f in result.top_risk_factors
            ],
            "vitals": {
                "heart_rate": round(vitals["d1_heartrate_max"], 1),
                "systolic_bp": round(vitals["d1_sysbp_min"], 1),
                "spo2": round(vitals["d1_spo2_min"], 1),
                "lactate": round(vitals["d1_lactate_max"], 1),
                "resp_rate": round(vitals["d1_resprate_max"], 1),
                "temperature": round(vitals["d1_temp_max"], 1)
            }
        }

        # Store latest result
        latest_results[pid] = payload

        # Broadcast to all connected dashboards
        await broadcast(payload)

        # Log alert
        if result.alert:
            logger.warning(
                f"🚨 ALERT — {profile['name']} ({pid}) "
                f"Risk: {result.risk_score:.0%} [{result.risk_level}]"
            )
        else:
            logger.info(
                f"✅ OK — {profile['name']} ({pid}) "
                f"Risk: {result.risk_score:.0%} [{result.risk_level}]"
            )

    except Exception as e:
        logger.error(f"Error processing {pid}: {e}")

async def run_all_patients():
    """Run predictions for all patients"""
    logger.info(f"⏱ Running predictions for all patients — {datetime.now().strftime('%H:%M:%S')}")
    for pid in PATIENT_PROFILES:
        await run_prediction_for_patient(pid)

def start_scheduler():
    """Start the background scheduler"""
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        run_all_patients,
        'interval',
        seconds=10,  # run every 10 seconds
        id='patient_monitor',
        name='ICU Patient Monitor'
    )
    scheduler.start()
    logger.info("🏥 PulseAlert scheduler started — monitoring every 10 seconds")
    return scheduler