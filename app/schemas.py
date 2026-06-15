from pydantic import BaseModel, Field
from typing import List, Optional

# Input schema - what the hospital sends to PulseAlert
class PatientInput(BaseModel):
    patient_id: str
    
    # Vitals
    d1_heartrate_max: float = Field(..., description="Max heart rate day 1")
    d1_heartrate_min: float = Field(..., description="Min heart rate day 1")
    d1_sysbp_max: float = Field(..., description="Max systolic BP day 1")
    d1_sysbp_min: float = Field(..., description="Min systolic BP day 1")
    d1_diasbp_max: float = Field(..., description="Max diastolic BP day 1")
    d1_diasbp_min: float = Field(..., description="Min diastolic BP day 1")
    d1_mbp_max: float = Field(..., description="Max mean BP day 1")
    d1_mbp_min: float = Field(..., description="Min mean BP day 1")
    d1_resprate_max: float = Field(..., description="Max respiratory rate day 1")
    d1_resprate_min: float = Field(..., description="Min respiratory rate day 1")
    d1_spo2_max: float = Field(..., description="Max SpO2 day 1")
    d1_spo2_min: float = Field(..., description="Min SpO2 day 1")
    d1_temp_max: float = Field(..., description="Max temperature day 1")
    d1_temp_min: float = Field(..., description="Min temperature day 1")
    
    # Labs
    d1_lactate_max: float = Field(..., description="Max lactate day 1")
    d1_lactate_min: float = Field(..., description="Min lactate day 1")
    d1_creatinine_max: float = Field(..., description="Max creatinine day 1")
    d1_creatinine_min: float = Field(..., description="Min creatinine day 1")
    d1_wbc_max: float = Field(..., description="Max WBC day 1")
    d1_wbc_min: float = Field(..., description="Min WBC day 1")
    d1_hemaglobin_max: float = Field(..., description="Max hemoglobin day 1")
    d1_hemaglobin_min: float = Field(..., description="Min hemoglobin day 1")
    d1_glucose_max: float = Field(..., description="Max glucose day 1")
    d1_glucose_min: float = Field(..., description="Min glucose day 1")
    d1_bun_max: float = Field(..., description="Max BUN day 1")
    d1_bun_min: float = Field(..., description="Min BUN day 1")
    d1_sodium_max: float = Field(..., description="Max sodium day 1")
    d1_sodium_min: float = Field(..., description="Min sodium day 1")
    d1_potassium_max: float = Field(..., description="Max potassium day 1")
    d1_potassium_min: float = Field(..., description="Min potassium day 1")
    
    # Demographics
    age: float
    gender: int = Field(..., description="0=Female 1=Male")
    weight: float
    height: float
    bmi: float
    elective_surgery: int = Field(..., description="0=Emergency 1=Elective")
    icu_type: int
    icu_admit_source: int
    
    # APACHE scores
    apache_4a_hospital_death_prob: float
    apache_4a_icu_death_prob: float
    gcs_eyes_apache: float
    gcs_motor_apache: float
    gcs_verbal_apache: float
    heart_rate_apache: float
    map_apache: float
    temp_apache: float
    creatinine_apache: float
    bun_apache: float
    wbc_apache: float
    
    # Comorbidities
    aids: int = 0
    cirrhosis: int = 0
    diabetes_mellitus: int = 0
    hepatic_failure: int = 0
    immunosuppression: int = 0
    solid_tumor_with_metastasis: int = 0

# Risk factor output
class RiskFactor(BaseModel):
    feature: str
    impact: str
    value: float
    shap_value: float

# Output schema - what PulseAlert returns
class PredictionOutput(BaseModel):
    patient_id: str
    risk_score: float
    risk_level: str
    alert: bool
    threshold_used: float
    top_risk_factors: List[RiskFactor]
    recommended_action: str

# Health check output
class HealthCheck(BaseModel):
    status: str
    model: str
    version: str