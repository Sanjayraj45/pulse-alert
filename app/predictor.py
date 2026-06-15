import numpy as np
import pandas as pd
import joblib
import os
from app.schemas import PatientInput, PredictionOutput, RiskFactor

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Load model and threshold
xgb_model = joblib.load(os.path.join(MODELS_DIR, 'xgboost_model.pkl'))
optimal_threshold = joblib.load(os.path.join(MODELS_DIR, 'optimal_threshold.pkl'))
shap_explainer = joblib.load(os.path.join(MODELS_DIR, 'shap_explainer.pkl'))
feature_names = joblib.load(os.path.join(BASE_DIR, 'data', 'feature_names.pkl'))

def engineer_features(data: dict) -> pd.DataFrame:
    """Engineer the same features we created in notebook 02"""
    df = pd.DataFrame([data])
    
    # Engineered features
    df['shock_index'] = df['d1_heartrate_max'] / df['d1_sysbp_min'].replace(0, np.nan)
    df['pulse_pressure'] = df['d1_sysbp_max'] - df['d1_diasbp_min']
    df['gcs_total'] = df['gcs_eyes_apache'] + df['gcs_motor_apache'] + df['gcs_verbal_apache']
    df['heartrate_range'] = df['d1_heartrate_max'] - df['d1_heartrate_min']
    df['sysbp_range'] = df['d1_sysbp_max'] - df['d1_sysbp_min']
    df['spo2_range'] = df['d1_spo2_max'] - df['d1_spo2_min']
    df['temp_range'] = df['d1_temp_max'] - df['d1_temp_min']
    df['bun_creatinine_ratio'] = df['d1_bun_max'] / df['d1_creatinine_max'].replace(0, np.nan)
    df['age_group'] = pd.cut(df['age'], bins=[0, 40, 60, 75, 100], labels=[0, 1, 2, 3]).astype(float)
    df['high_lactate_flag'] = (df['d1_lactate_max'] > 2.0).astype(int)
    
    # Fill any NaN
    df = df.fillna(df.median(numeric_only=True))
    
    # Reorder columns to match training
    df = df[feature_names]
    
    return df

def get_risk_level(risk_score: float) -> str:
    """Convert risk score to risk level"""
    if risk_score >= 0.75:
        return "CRITICAL"
    elif risk_score >= 0.50:
        return "HIGH"
    elif risk_score >= 0.35:
        return "MEDIUM"
    else:
        return "LOW"

def get_recommended_action(risk_level: str) -> str:
    """Get recommended action based on risk level"""
    actions = {
        "CRITICAL": "Notify attending physician immediately. Prepare for rapid response.",
        "HIGH": "Alert charge nurse and attending physician within 30 minutes.",
        "MEDIUM": "Increase monitoring frequency. Reassess vitals every 30 minutes.",
        "LOW": "Continue routine monitoring."
    }
    return actions[risk_level]

def predict(patient: PatientInput) -> PredictionOutput:
    """Main prediction function"""
    
    # Convert input to dict (exclude patient_id)
    patient_dict = patient.dict()
    patient_id = patient_dict.pop('patient_id')
    
    # Engineer features
    X = engineer_features(patient_dict)
    
    # Get risk score
    risk_score = float(xgb_model.predict_proba(X)[:, 1][0])
    risk_level = get_risk_level(risk_score)
    alert = risk_score >= optimal_threshold
    
    # Get SHAP values
    shap_values = shap_explainer.shap_values(X)
    shap_series = pd.Series(shap_values[0], index=feature_names)
    top_shap = shap_series.abs().sort_values(ascending=False).head(5)
    
    # Build top risk factors
    top_risk_factors = []
    for feature in top_shap.index:
        shap_val = shap_series[feature]
        abs_val = abs(shap_val)
        
        if abs_val > 0.3:
            impact = "HIGH"
        elif abs_val > 0.1:
            impact = "MEDIUM"
        else:
            impact = "LOW"
        
        top_risk_factors.append(RiskFactor(
            feature=feature,
            impact=impact,
            value=round(float(X[feature].values[0]), 2),
            shap_value=round(float(shap_val), 4)
        ))
    
    return PredictionOutput(
        patient_id=patient_id,
        risk_score=round(risk_score, 4),
        risk_level=risk_level,
        alert=alert,
        threshold_used=optimal_threshold,
        top_risk_factors=top_risk_factors,
        recommended_action=get_recommended_action(risk_level)
    )