# 🚨 PulseAlert — ICU Early Warning System

> Real-time patient deterioration prediction for Intensive Care Units, powered by XGBoost and SHAP explainability.

## What It Does

PulseAlert monitors ICU patients in real time and predicts their risk of deterioration before a crisis occurs. It ingests patient vitals, runs them through a trained XGBoost model, and broadcasts risk scores, SHAP-based factor explanations, and clinical recommendations to a live dashboard — updated every 5 seconds.

**The core clinical insight:** Nurses managing 10–20 ICU beds simultaneously cannot manually track every vital trend. PulseAlert continuously scores each patient and fires alerts early, ensuring no patient silently deteriorates while attention is elsewhere.

---

## Live Demo

Three built-in patient scenarios — no CSV upload required:

| Scenario | Description | Outcome |
|---|---|---|
| 📈 Recovery | Patient stabilizing after treatment | Risk drops LOW |
| 📉 Gradual Deterioration | Slow decline over 30 readings | Risk climbs to CRITICAL |
| ⚡ Sudden Crash | Stable then acute septic shock | Rapid CRITICAL spike |

---

## Architecture

```
WiDS ICU Dataset (91,713 patients)
        ↓
  Feature Engineering (66 features, 10 engineered)
        ↓
  XGBoost Model  ←→  SHAP Explainer
        ↓
  FastAPI Backend (REST + WebSocket)
        ↓
  MySQL Database  →  PDF Report Generator
        ↓
  Live Dashboard (WebSocket real-time updates)
```

**Tech Stack**

| Layer | Technology |
|---|---|
| ML Model | XGBoost (AUC 0.887, Recall 0.86) |
| Explainability | SHAP TreeExplainer |
| Backend | FastAPI + Uvicorn |
| Real-time | WebSockets |
| Database | MySQL 8.0 |
| Reports | ReportLab + Matplotlib |
| Frontend | Vanilla JS + Chart.js |
| Dataset | WiDS Datathon 2020 (Kaggle) |

---

## ML Pipeline

### Dataset
- **Source:** WiDS Datathon 2020 — 91,713 ICU patients, 186 features
- **Target:** `hospital_death` (binary) — 8.63% positive rate (class imbalance)

### Feature Engineering
10 clinically meaningful features engineered on top of raw vitals:

| Feature | Clinical Meaning |
|---|---|
| `shock_index` | HR / Systolic BP — ratio > 1.0 indicates circulatory shock |
| `pulse_pressure` | Systolic max − Diastolic min — narrows in shock |
| `gcs_total` | Eyes + Motor + Verbal — composite consciousness score |
| `high_lactate_flag` | Lactate > 2.0 mmol/L — tissue hypoperfusion marker |
| `heartrate_range` | Variability in heart rate over 24h |
| `sysbp_range` | Variability in systolic BP |
| `spo2_range` | Variability in oxygen saturation |
| `temp_range` | Temperature variability |
| `bun_creatinine_ratio` | Renal function indicator |
| `age_group` | Binned age brackets (0–40, 40–60, 60–75, 75+) |

### Model Selection

| Model | AUC | Recall | Notes |
|---|---|---|---|
| Logistic Regression | 0.840 | 0.69 | Baseline, misses non-linear patterns |
| Random Forest | 0.874 | 0.59 | Better AUC but lower recall |
| **XGBoost** | **0.887** | **0.78** | Selected — best balance of AUC and recall |

**Why XGBoost?** Boosting builds trees sequentially, each correcting the previous one's errors. Captures non-linear vital interactions (e.g. high HR + low BP together = shock) that Random Forest's independent trees miss.

### Class Imbalance Handling
Two complementary techniques:
1. **SMOTE** on training data — synthetic minority oversampling from 73,370 → 134,076 balanced samples
2. **`scale_pos_weight = 10.59`** in XGBoost — further weights the minority class during training

### Threshold Optimization
Default threshold of 0.5 optimizes accuracy. In ICU medicine, **missing a death (false negative) is far more dangerous than a false alarm (false positive).**

Optimal threshold **0.35** selected at:
- Recall: **0.8629** (catches 86% of actual deaths)
- Specificity: **0.7304**

### SHAP Explainability
Every prediction includes a SHAP explanation — not just "this patient is high risk" but **why**, for this specific patient, right now.

- Impact thresholds calibrated from real SHAP value distribution (90th percentile ≈ 0.22, 75th ≈ 0.10)
- Direction field: "increases risk" / "decreases risk" per feature
- Top 5 SHAP factors shown on dashboard and in PDF report

---

## System Design

### Live Player Architecture
```
CSV Upload (multi-row, one patient)
    ↓
load_csv_for_player() — safeguard blocks duplicate loads
    ↓
run_player_loop() — asyncio background task, ticks every 5s
    ↓
process_one_tick():
    engineer_features() → XGBoost inference → SHAP values
    → save_prediction() → MySQL
    → broadcast() → WebSocket → Dashboard
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/player/load` | Load multi-row CSV for live replay |
| `POST` | `/player/pause` | Pause the player |
| `POST` | `/player/resume` | Resume the player |
| `POST` | `/player/reset` | Reset and clear player state |
| `POST` | `/demo/start?scenario=` | Start built-in demo scenario |
| `POST` | `/upload` | Batch CSV upload (all patients at once) |
| `GET`  | `/report/{patient_id}` | Download PDF report |
| `GET`  | `/patients/{id}/trend` | Risk score trend over time |
| `GET`  | `/alerts` | Recent alert log |
| `WS`   | `/ws` | WebSocket live updates |

---

## Project Structure

```
pulsealert/
├── app/
│   ├── main.py          # FastAPI app, all endpoints
│   ├── predictor.py     # XGBoost inference + feature engineering
│   ├── explainer.py     # SHAP + nurse summary generation
│   ├── player.py        # Live CSV replay engine
│   ├── database.py      # MySQL connection pool + queries
│   ├── report.py        # PDF report generation (ReportLab)
│   ├── scheduler.py     # WebSocket broadcast helper
│   └── schemas.py       # Pydantic models
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_shap_explainability.ipynb
├── data/
│   ├── demo_recovery.csv        # Built-in demo: recovery scenario
│   ├── demo_deterioration.csv   # Built-in demo: gradual decline
│   ├── demo_crash.csv           # Built-in demo: sudden crash
│   └── sample_patients.csv      # 20 random WiDS patients for testing
├── models/
│   ├── xgboost_model.pkl
│   ├── shap_explainer.pkl
│   ├── optimal_threshold.pkl
│   └── feature_names.pkl
├── dashboard.html       # Live monitoring dashboard
├── requirements.txt
└── run.py
```

---

## Running Locally

### Prerequisites
- Python 3.10+
- MySQL 8.0 running locally
- Git

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/pulsealert.git
cd pulsealert

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create MySQL database
mysql -u root -p -e "CREATE DATABASE pulsealert;"

# 5. Set your MySQL password in app/database.py
# DB_CONFIG = { ..., 'password': 'your_password', ... }

# 6. Download WiDS dataset from Kaggle
# Place training_v2.csv in data/

# 7. Run notebooks in order (01 → 05) to train models

# 8. Start the API
python run.py

# 9. Serve the dashboard (new terminal)
python -m http.server 5500

# 10. Open browser
# http://localhost:5500/dashboard.html
```

---

## Key Design Decisions

**Why recall over precision?**
In a clinical setting, a false negative (missing a deteriorating patient) can be fatal. A false alarm (unnecessary nurse check) wastes 5 minutes. Threshold 0.35 was chosen to maximize recall while keeping specificity above 0.73.

**Why WebSocket over polling?**
Polling asks "anything new?" every N seconds — adds latency and wastes server resources. WebSocket maintains a persistent connection, so the server pushes updates the moment a new prediction is ready.

**Why MySQL over SQLite?**
MySQL supports concurrent connections, connection pooling, and scales to multiple application instances. SQLite locks on writes, making it unsuitable for a real-time system with a background player loop and API requests running simultaneously.

**Why SHAP over feature importance?**
Global feature importance tells you which features matter across the whole model. SHAP gives local explanations — why *this specific patient* got *this specific score* — which is what makes the system clinically actionable.

---

## Limitations & Future Work

This system is a **clinical decision support tool**, not a diagnostic replacement. All alerts should be verified by qualified medical staff.

**Current limitations:**
- Snapshot classifier — each reading is treated independently, without memory of previous readings
- Trained on WiDS 2020 snapshot data — not true time-series deterioration forecasting
- Demo data is synthetically generated to simulate deterioration trajectories

**Planned future work:**
- Time-series forecasting using MIMIC-III chartevents (true windowed temporal labels)
- LSTM or windowed XGBoost for 30-minute ahead deterioration prediction
- HL7/FHIR integration for real EMR data feeds
- Multi-patient simultaneous player support
- Docker + cloud deployment (Railway / AWS)

---

## Results Summary

| Metric | Value |
|---|---|
| Dataset size | 91,713 ICU patients |
| Features used | 66 (56 selected + 10 engineered) |
| Model | XGBoost |
| AUC-ROC | 0.8871 |
| Recall @ threshold 0.35 | 0.8629 |
| Specificity @ threshold 0.35 | 0.7304 |
| Deaths caught (of 1,583 test) | 1,366 (86%) |
| False negatives | 217 missed deaths |

---

## Author

Built by SANJAY K C as a portfolio project demonstrating end-to-end ML engineering — from raw ICU data to a production-ready real-time monitoring system.

- GitHub: [@Sanjayraj45](https://github.com/Sanjayraj45)
- LinkedIn:[@Sanjay Kc](https://www.linkedin.com/in/sanjay-kc-43b1022a2/)

---

*Generated by PulseAlert. For clinical decision support only — does not replace professional medical judgment.*
