import numpy as np
import random
from datetime import datetime

PATIENT_PROFILES = {
    "ICU-1001": {
        "name": "Rajesh Kumar",
        "age": 67, "gender": 1, "weight": 78, "height": 170, "bmi": 27.0,
        "elective_surgery": 0, "diabetes_mellitus": 1,
        "severity": "critical",
        "base_hr": 118, "base_sbp": 86, "base_spo2": 87,
        "base_lactate": 4.5, "base_rr": 28
    },
    "ICU-1002": {
        "name": "Priya Sharma",
        "age": 45, "gender": 0, "weight": 62, "height": 162, "bmi": 23.6,
        "elective_surgery": 1, "diabetes_mellitus": 0,
        "severity": "stable",
        "base_hr": 76, "base_sbp": 122, "base_spo2": 98,
        "base_lactate": 0.9, "base_rr": 15
    },
    "ICU-1003": {
        "name": "Mohammed Ali",
        "age": 72, "gender": 1, "weight": 85, "height": 175, "bmi": 27.8,
        "elective_surgery": 0, "diabetes_mellitus": 1,
        "severity": "high",
        "base_hr": 108, "base_sbp": 94, "base_spo2": 91,
        "base_lactate": 2.9, "base_rr": 23
    },
    "ICU-1004": {
        "name": "Ananya Reddy",
        "age": 34, "gender": 0, "weight": 58, "height": 158, "bmi": 23.2,
        "elective_surgery": 1, "diabetes_mellitus": 0,
        "severity": "low",
        "base_hr": 70, "base_sbp": 124, "base_spo2": 99,
        "base_lactate": 0.8, "base_rr": 13
    }
}

patient_state = {}

def init_patient_state():
    for pid, profile in PATIENT_PROFILES.items():
        patient_state[pid] = {
            "tick": 0,
            "hr": profile["base_hr"],
            "sbp": profile["base_sbp"],
            "spo2": profile["base_spo2"],
            "lactate": profile["base_lactate"],
            "rr": profile["base_rr"]
        }

def get_noise(scale=1.0):
    return np.random.normal(0, scale)

def update_patient_vitals(pid: str) -> dict:
    profile = PATIENT_PROFILES[pid]
    state = patient_state[pid]
    severity = profile["severity"]
    tick = state["tick"]

    if severity == "critical":
        drift = tick * 0.03
        state["hr"]      = profile["base_hr"] + drift + get_noise(3)
        state["sbp"]     = max(60, profile["base_sbp"] - drift * 0.2 + get_noise(4))
        state["spo2"]    = max(70, profile["base_spo2"] - drift * 0.05 + get_noise(1))
        state["lactate"] = min(15, profile["base_lactate"] + drift * 0.01 + get_noise(0.2))
        state["rr"]      = profile["base_rr"] + drift * 0.05 + get_noise(1)

    elif severity == "high":
        state["hr"]      = profile["base_hr"] + get_noise(6)
        state["sbp"]     = profile["base_sbp"] + get_noise(8)
        state["spo2"]    = profile["base_spo2"] + get_noise(1.5)
        state["lactate"] = profile["base_lactate"] + get_noise(0.2)
        state["rr"]      = profile["base_rr"] + get_noise(2)

    elif severity == "stable":
        state["hr"]      = profile["base_hr"] + get_noise(3)
        state["sbp"]     = profile["base_sbp"] + get_noise(5)
        state["spo2"]    = min(100, profile["base_spo2"] + get_noise(0.8))
        state["lactate"] = profile["base_lactate"] + get_noise(0.08)
        state["rr"]      = profile["base_rr"] + get_noise(1)

    else:
        state["hr"]      = profile["base_hr"] + get_noise(2)
        state["sbp"]     = profile["base_sbp"] + get_noise(3)
        state["spo2"]    = min(100, profile["base_spo2"] + get_noise(0.5))
        state["lactate"] = profile["base_lactate"] + get_noise(0.05)
        state["rr"]      = profile["base_rr"] + get_noise(0.5)

    state["tick"] += 1

    hr      = max(40,  min(200, state["hr"]))
    sbp     = max(60,  min(200, state["sbp"]))
    spo2    = max(70,  min(100, state["spo2"]))
    lactate = max(0.5, min(15.0, state["lactate"]))
    rr      = max(8,   min(50, state["rr"]))
    dbp     = sbp * 0.6 + get_noise(3)
    mbp     = (sbp + 2 * dbp) / 3

    # APACHE scores based on severity
    if severity == "critical":
        apache_hosp  = round(min(0.99, 0.65 + get_noise(0.03)), 3)
        apache_icu   = round(min(0.99, 0.55 + get_noise(0.03)), 3)
        gcs_eyes     = 2
        gcs_motor    = 3
        gcs_verbal   = 2
    elif severity == "high":
        apache_hosp  = round(min(0.99, max(0.01, 0.35 + get_noise(0.04))), 3)
        apache_icu   = round(min(0.99, max(0.01, 0.28 + get_noise(0.04))), 3)
        gcs_eyes     = 3
        gcs_motor    = 5
        gcs_verbal   = 4
    elif severity == "stable":
        apache_hosp  = round(max(0.01, 0.08 + get_noise(0.02)), 3)
        apache_icu   = round(max(0.01, 0.06 + get_noise(0.02)), 3)
        gcs_eyes     = 4
        gcs_motor    = 6
        gcs_verbal   = 5
    else:
        apache_hosp  = round(max(0.01, 0.03 + get_noise(0.01)), 3)
        apache_icu   = round(max(0.01, 0.02 + get_noise(0.01)), 3)
        gcs_eyes     = 4
        gcs_motor    = 6
        gcs_verbal   = 5

    return {
        "patient_id": pid,
        "name": profile["name"],
        "age": profile["age"],
        "gender": profile["gender"],
        "weight": profile["weight"],
        "height": profile["height"],
        "bmi": profile["bmi"],
        "elective_surgery": profile["elective_surgery"],
        "diabetes_mellitus": profile["diabetes_mellitus"],
        "severity": severity,
        "timestamp": datetime.now().isoformat(),

        "d1_heartrate_max": round(hr + get_noise(2), 1),
        "d1_heartrate_min": round(max(40, hr - 15 + get_noise(2)), 1),
        "d1_sysbp_max": round(sbp + get_noise(4), 1),
        "d1_sysbp_min": round(max(60, sbp - 10 + get_noise(4)), 1),
        "d1_diasbp_max": round(dbp + get_noise(3), 1),
        "d1_diasbp_min": round(max(40, dbp - 8 + get_noise(3)), 1),
        "d1_mbp_max": round(mbp + get_noise(3), 1),
        "d1_mbp_min": round(max(40, mbp - 8 + get_noise(3)), 1),
        "d1_resprate_max": round(rr + get_noise(1), 1),
        "d1_resprate_min": round(max(8, rr - 4 + get_noise(1)), 1),
        "d1_spo2_max": round(min(100, spo2 + 2 + get_noise(0.5)), 1),
        "d1_spo2_min": round(spo2 + get_noise(0.5), 1),
        "d1_temp_max": round(36.5 + get_noise(0.4) + (0.8 if severity == "critical" else 0.3 if severity == "high" else 0), 1),
        "d1_temp_min": round(36.2 + get_noise(0.3), 1),

        "d1_lactate_max": round(lactate + get_noise(0.1), 2),
        "d1_lactate_min": round(max(0.5, lactate - 0.3 + get_noise(0.1)), 2),
        "d1_creatinine_max": round(max(0.5, 1.2 + get_noise(0.2) + (1.2 if severity == "critical" else 0.4 if severity == "high" else 0)), 2),
        "d1_creatinine_min": round(max(0.4, 1.0 + get_noise(0.1)), 2),
        "d1_wbc_max": round(max(1, 10 + get_noise(2) + (5 if severity == "critical" else 2 if severity == "high" else 0)), 1),
        "d1_wbc_min": round(max(1, 8 + get_noise(1)), 1),
        "d1_hemaglobin_max": round(max(5, 10 + get_noise(0.5) - (2 if severity == "critical" else 0.5 if severity == "high" else 0)), 1),
        "d1_hemaglobin_min": round(max(4, 9 + get_noise(0.5)), 1),
        "d1_glucose_max": round(max(50, 140 + get_noise(20) + (40 if profile["diabetes_mellitus"] else 0)), 1),
        "d1_glucose_min": round(max(40, 110 + get_noise(15)), 1),
        "d1_bun_max": round(max(5, 22 + get_noise(3) + (15 if severity == "critical" else 5 if severity == "high" else 0)), 1),
        "d1_bun_min": round(max(4, 18 + get_noise(2)), 1),
        "d1_sodium_max": round(138 + get_noise(2), 1),
        "d1_sodium_min": round(135 + get_noise(2), 1),
        "d1_potassium_max": round(max(2.5, 4.2 + get_noise(0.3) + (0.6 if severity == "critical" else 0.2 if severity == "high" else 0)), 2),
        "d1_potassium_min": round(max(2.0, 3.8 + get_noise(0.2)), 2),

        "apache_4a_hospital_death_prob": apache_hosp,
        "apache_4a_icu_death_prob": apache_icu,
        "gcs_eyes_apache": gcs_eyes,
        "gcs_motor_apache": gcs_motor,
        "gcs_verbal_apache": gcs_verbal,
        "heart_rate_apache": round(hr, 1),
        "map_apache": round(mbp, 1),
        "temp_apache": round(36.5 + get_noise(0.4), 1),
        "creatinine_apache": round(max(0.5, 1.2 + get_noise(0.2)), 2),
        "bun_apache": round(max(5, 22 + get_noise(3)), 1),
        "wbc_apache": round(max(1, 10 + get_noise(2)), 1),

        "icu_type": 1,
        "icu_admit_source": 1,
        "aids": 0,
        "cirrhosis": 0,
        "hepatic_failure": 0,
        "immunosuppression": 0,
        "solid_tumor_with_metastasis": 0
    }

init_patient_state()