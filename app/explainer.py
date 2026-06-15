import numpy as np
import pandas as pd
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Load explainer and feature names
shap_explainer = joblib.load(os.path.join(MODELS_DIR, 'shap_explainer.pkl'))
feature_names = joblib.load(os.path.join(BASE_DIR, 'data', 'feature_names.pkl'))

def get_shap_explanation(X: pd.DataFrame) -> dict:
    """Get full SHAP explanation for a patient"""
    
    # Calculate SHAP values
    shap_values = shap_explainer.shap_values(X)
    shap_series = pd.Series(shap_values[0], index=feature_names)
    
    # Sort by absolute impact
    shap_sorted = shap_series.abs().sort_values(ascending=False)
    
    # Build explanation dict
    explanation = []
    for feature in shap_sorted.index:
        shap_val = shap_series[feature]
        feature_val = float(X[feature].values[0])
        
        explanation.append({
            'feature': feature,
            'feature_value': round(feature_val, 4),
            'shap_value': round(float(shap_val), 4),
            'direction': 'increases_risk' if shap_val > 0 else 'decreases_risk',
            'impact_magnitude': round(float(abs(shap_val)), 4)
        })
    
    return {
        'feature_impacts': explanation,
        'base_value': round(float(shap_explainer.expected_value), 4),
        'total_features': len(feature_names)
    }

def get_nurse_explanation(
    risk_score: float,
    risk_level: str,
    top_factors: list,
    patient_data: pd.DataFrame
) -> str:
    """Generate a nurse-friendly explanation of the prediction"""
    
    # Extract key vitals
    spo2 = patient_data['d1_spo2_min'].values[0]
    hr = patient_data['d1_heartrate_max'].values[0]
    sbp = patient_data['d1_sysbp_min'].values[0]
    lactate = patient_data['d1_lactate_max'].values[0]
    rr = patient_data['d1_resprate_max'].values[0]
    gcs = patient_data['gcs_total'].values[0]
    shock_idx = patient_data['shock_index'].values[0]
    
    # Build explanation based on risk level
    concerns = []
    
    if spo2 < 90:
        concerns.append(f"critically low SpO2 at {spo2:.0f}%")
    elif spo2 < 94:
        concerns.append(f"low SpO2 at {spo2:.0f}%")
        
    if lactate > 4.0:
        concerns.append(f"critically elevated lactate at {lactate:.1f} mmol/L")
    elif lactate > 2.0:
        concerns.append(f"elevated lactate at {lactate:.1f} mmol/L")
        
    if sbp < 90:
        concerns.append(f"dangerously low systolic BP at {sbp:.0f} mmHg")
    elif sbp < 100:
        concerns.append(f"low systolic BP at {sbp:.0f} mmHg")
        
    if hr > 120:
        concerns.append(f"elevated heart rate at {hr:.0f} bpm")
        
    if rr > 30:
        concerns.append(f"high respiratory rate at {rr:.0f} breaths/min")
        
    if gcs < 10:
        concerns.append(f"reduced consciousness with GCS of {gcs:.0f}")
        
    if shock_idx > 1.0:
        concerns.append(f"shock index of {shock_idx:.2f} indicating circulatory compromise")
    
    # Build summary sentence
    if not concerns:
        summary = f"Patient has a risk score of {risk_score:.0%} based on combined clinical indicators."
    elif len(concerns) == 1:
        summary = f"Patient shows {concerns[0]}."
    else:
        summary = f"Patient shows {', '.join(concerns[:-1])}, and {concerns[-1]}."
    
    # Add risk context
    if risk_level == "CRITICAL":
        summary += " Multiple critical indicators suggest imminent deterioration."
    elif risk_level == "HIGH":
        summary += " Clinical trajectory suggests increased risk of deterioration."
    elif risk_level == "MEDIUM":
        summary += " Closely monitor for further deterioration."
        
    return summary