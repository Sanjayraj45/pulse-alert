import mysql.connector
from mysql.connector import pooling
import os
import json
from datetime import datetime

# MySQL connection config — update password if different
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'sanjayraj123',
    'database': 'pulsealert'
}

# Connection pool for efficiency
connection_pool = mysql.connector.pooling.MySQLConnectionPool(
    pool_name="pulsealert_pool",
    pool_size=5,
    **DB_CONFIG
)

def get_connection():
    return connection_pool.get_connection()

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            patient_id VARCHAR(50) NOT NULL,
            timestamp DATETIME NOT NULL,
            risk_score FLOAT NOT NULL,
            risk_level VARCHAR(20) NOT NULL,
            alert TINYINT NOT NULL,
            heart_rate FLOAT,
            systolic_bp FLOAT,
            spo2 FLOAT,
            lactate FLOAT,
            resp_rate FLOAT,
            temperature FLOAT,
            age FLOAT,
            top_factor_1 VARCHAR(100),
            top_factor_2 VARCHAR(100),
            top_factor_3 VARCHAR(100),
            top_factors_json TEXT,
            recommended_action TEXT,
            source VARCHAR(20) DEFAULT 'manual'
        )
    ''')

    # Add the new column if the table already existed from before this update
    try:
        cursor.execute('ALTER TABLE predictions ADD COLUMN top_factors_json TEXT')
    except Exception:
        pass  # column already exists, safe to ignore

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            patient_id VARCHAR(50) PRIMARY KEY,
            name VARCHAR(100),
            age FLOAT,
            gender INT,
            first_seen DATETIME,
            last_seen DATETIME,
            total_predictions INT DEFAULT 0,
            highest_risk FLOAT DEFAULT 0,
            current_risk_level VARCHAR(20) DEFAULT 'LOW'
        )
    ''')

    conn.commit()
    cursor.close()
    conn.close()
    print("Database initialized ✅")

def save_prediction(patient_id: str, result: dict, vitals: dict, source: str = 'manual'):
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    factors = result.get('top_risk_factors', [])
    f1 = factors[0]['feature'] if len(factors) > 0 else ''
    f2 = factors[1]['feature'] if len(factors) > 1 else ''
    f3 = factors[2]['feature'] if len(factors) > 2 else ''
    factors_json = json.dumps(factors)

    cursor.execute('''
        INSERT INTO predictions (
            patient_id, timestamp, risk_score, risk_level, alert,
            heart_rate, systolic_bp, spo2, lactate, resp_rate, temperature, age,
            top_factor_1, top_factor_2, top_factor_3, top_factors_json,
            recommended_action, source
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ''', (
        patient_id, now,
        result['risk_score'], result['risk_level'], int(result['alert']),
        vitals.get('heart_rate'), vitals.get('systolic_bp'),
        vitals.get('spo2'), vitals.get('lactate'),
        vitals.get('resp_rate'), vitals.get('temperature'),
        vitals.get('age'),
        f1, f2, f3, factors_json,
        result['recommended_action'], source
    ))

    # Upsert into patients table
    cursor.execute('''
        INSERT INTO patients (patient_id, first_seen, last_seen, total_predictions, highest_risk, current_risk_level)
        VALUES (%s, %s, %s, 1, %s, %s)
        ON DUPLICATE KEY UPDATE
            last_seen = %s,
            total_predictions = total_predictions + 1,
            highest_risk = GREATEST(highest_risk, %s),
            current_risk_level = %s
    ''', (
        patient_id, now, now, result['risk_score'], result['risk_level'],
        now, result['risk_score'], result['risk_level']
    ))

    conn.commit()
    cursor.close()
    conn.close()

def get_patient_history(patient_id: str, limit: int = 50):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('''
        SELECT * FROM predictions
        WHERE patient_id = %s
        ORDER BY timestamp DESC
        LIMIT %s
    ''', (patient_id, limit))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def get_all_patients():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('''
        SELECT p.*,
               pr.risk_score as latest_risk,
               pr.risk_level as latest_level,
               pr.alert as latest_alert,
               pr.timestamp as latest_time
        FROM patients p
        LEFT JOIN predictions pr ON p.patient_id = pr.patient_id
        WHERE pr.id = (
            SELECT MAX(id) FROM predictions
            WHERE patient_id = p.patient_id
        )
        ORDER BY pr.risk_score DESC
    ''')
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def get_recent_alerts(limit: int = 20):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('''
        SELECT * FROM predictions
        WHERE alert = 1
        ORDER BY timestamp DESC
        LIMIT %s
    ''', (limit,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def get_risk_trend(patient_id: str, limit: int = 20):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('''
        SELECT timestamp, risk_score, risk_level, heart_rate, spo2, systolic_bp, lactate
        FROM predictions
        WHERE patient_id = %s
        ORDER BY timestamp ASC
        LIMIT %s
    ''', (patient_id, limit))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

# Initialize on import
init_db()