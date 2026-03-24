import sqlite3
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    jwt_required,
    get_jwt_identity
)
import json
import os

# ================= APP SETUP =================
app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "DELETE", "OPTIONS"])

app.config["JWT_SECRET_KEY"]           = "this-is-a-very-secure-32-byte-secret-key-2026"
app.config["JWT_TOKEN_LOCATION"]       = ["headers"]
app.config["JWT_HEADER_NAME"]          = "Authorization"
app.config["JWT_HEADER_TYPE"]          = "Bearer"
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24)

jwt    = JWTManager(app)
bcrypt = Bcrypt(app)

# ================= DATABASE =================
# Store DB outside project folder so Live Server never detects file changes
DB_PATH = os.path.join(os.path.expanduser("~"), "medpredict_users.db")

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            role TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            disease_type TEXT,
            prediction INTEGER,
            probability REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ================= LOAD ML MODELS =================
diabetes_model  = joblib.load("diabetes_model.joblib")
diabetes_scaler = joblib.load("diabetes_scaler.joblib")
heart_model     = joblib.load("heart_model.joblib")
heart_scaler    = joblib.load("heart_scaler.joblib")

# ================= HEART FEATURE ENGINEERING =================
EXPECTED_HEART_COLS = [
    "Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak",
    "Sex_M",
    "ChestPainType_ATA", "ChestPainType_NAP", "ChestPainType_TA",
    "RestingECG_Normal", "RestingECG_ST",
    "ExerciseAngina_Y",
    "ST_Slope_Flat", "ST_Slope_Up"
]

def encode_heart_input(age, sex, chest_pain, resting_bp, cholesterol,
                        fasting_bs, resting_ecg, max_hr, exercise_angina,
                        oldpeak, st_slope):
    row = pd.DataFrame([{
        "Age": int(age), "RestingBP": int(resting_bp),
        "Cholesterol": int(cholesterol), "FastingBS": int(fasting_bs),
        "MaxHR": int(max_hr), "Oldpeak": float(oldpeak),
        "Sex": str(sex), "ChestPainType": str(chest_pain),
        "RestingECG": str(resting_ecg), "ExerciseAngina": str(exercise_angina),
        "ST_Slope": str(st_slope),
    }])
    cat_dtypes = {
        "Sex":            pd.CategoricalDtype(["F", "M"]),
        "ChestPainType":  pd.CategoricalDtype(["ASY", "ATA", "NAP", "TA"]),
        "RestingECG":     pd.CategoricalDtype(["LVH", "Normal", "ST"]),
        "ExerciseAngina": pd.CategoricalDtype(["N", "Y"]),
        "ST_Slope":       pd.CategoricalDtype(["Down", "Flat", "Up"]),
    }
    for col, dtype in cat_dtypes.items():
        row[col] = row[col].astype(dtype)
    row_encoded = pd.get_dummies(row, columns=list(cat_dtypes.keys()), drop_first=True)
    for col in EXPECTED_HEART_COLS:
        if col not in row_encoded.columns:
            row_encoded[col] = 0
    return row_encoded[EXPECTED_HEART_COLS].astype(float)

# ================= HELPERS =================
def get_ist_now():
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

def get_current_user():
    identity = get_jwt_identity()
    if isinstance(identity, str):
        return json.loads(identity)
    return identity

# ================= AUTH ROUTES =================

@app.route("/signup", methods=["POST"])
def signup():
    data      = request.json
    username  = data.get("username")
    password  = data.get("password")
    role      = data.get("role")
    admin_key = data.get("admin_key")
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    if role == "admin":
        if admin_key != "ADMIN123":
            return jsonify({"error": "Invalid admin key"}), 403
    if role not in ["admin", "user"]:
        return jsonify({"error": "Invalid role"}), 400
    hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                  (username, hashed_pw, role))
        conn.commit()
        conn.close()
        return jsonify({"message": "Signup successful"})
    except:
        return jsonify({"error": "Username already exists"}), 400

@app.route("/login", methods=["POST"])
def login():
    data     = request.json
    username = data.get("username")
    password = data.get("password")
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, password, role FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    if user and bcrypt.check_password_hash(user[1], password):
        identity = json.dumps({"id": user[0], "role": user[2], "username": username})
        token = create_access_token(identity=identity)
        return jsonify({"message": "Login successful", "token": token, "role": user[2]})
    return jsonify({"error": "Invalid credentials"}), 401

# ================= ADMIN ROUTES =================

@app.route("/admin/history", methods=["GET"])
@jwt_required()
def admin_history():
    current_user = get_current_user()
    if current_user["role"] != "admin":
        return jsonify({"error": "Access denied"}), 403
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT users.username, disease_type, prediction, probability, timestamp
        FROM predictions JOIN users ON predictions.user_id = users.id
        ORDER BY timestamp DESC
    """)
    rows = c.fetchall()
    conn.close()
    return jsonify([{"username":r[0],"disease":r[1],"prediction":r[2],"probability":r[3],"timestamp":r[4]} for r in rows])

@app.route("/admin/users", methods=["GET"])
@jwt_required()
def admin_get_users():
    current_user = get_current_user()
    if current_user["role"] != "admin":
        return jsonify({"error": "Access denied"}), 403
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT u.id, u.username, u.role,
            COUNT(p.id),
            SUM(CASE WHEN p.disease_type='diabetes' THEN 1 ELSE 0 END),
            SUM(CASE WHEN p.disease_type='heart' THEN 1 ELSE 0 END),
            MAX(p.timestamp)
        FROM users u LEFT JOIN predictions p ON u.id=p.user_id
        GROUP BY u.id ORDER BY u.id ASC
    """)
    rows = c.fetchall()
    conn.close()
    return jsonify([{"id":r[0],"username":r[1],"role":r[2],"total_predictions":r[3],"diabetes_count":r[4],"heart_count":r[5],"last_active":r[6]} for r in rows])

@app.route("/admin/users/<int:user_id>", methods=["DELETE"])
@jwt_required()
def admin_delete_user(user_id):
    current_user = get_current_user()
    if current_user["role"] != "admin":
        return jsonify({"error": "Access denied"}), 403
    if current_user["id"] == user_id:
        return jsonify({"error": "You cannot delete your own account"}), 400
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, username FROM users WHERE id=?", (user_id,))
    user = c.fetchone()
    if not user:
        conn.close()
        return jsonify({"error": "User not found"}), 404
    c.execute("DELETE FROM predictions WHERE user_id=?", (user_id,))
    c.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    return jsonify({"message": f"User '{user[1]}' and all their data deleted successfully"})

# ================= USER HISTORY =================

@app.route("/history", methods=["GET"])
@jwt_required()
def user_history():
    current_user = get_current_user()
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT disease_type, prediction, probability, timestamp FROM predictions WHERE user_id=? ORDER BY timestamp DESC", (current_user["id"],))
    rows = c.fetchall()
    conn.close()
    return jsonify([{"disease":r[0],"prediction":r[1],"probability":r[2],"timestamp":r[3]} for r in rows])

# ================= PREDICTION ROUTES =================

@app.route("/predict/diabetes", methods=["POST"])
@jwt_required()
def predict_diabetes():
    current_user = get_current_user()
    data = request.json.get("input")
    try:
        DIABETES_COLS = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                         'Insulin','BMI','DiabetesPedigreeFunction','Age']
        arr = pd.DataFrame([data], columns=DIABETES_COLS, dtype=np.float64)
        arr_scaled = diabetes_scaler.transform(arr)
        prob = float(diabetes_model.predict_proba(arr_scaled)[0][1])
        prediction = int(prob > 0.5)
        conn = get_db()
        c = conn.cursor()
        c.execute("INSERT INTO predictions (user_id, disease_type, prediction, probability, timestamp) VALUES (?,?,?,?,?)",
                  (current_user["id"], "diabetes", prediction, prob, get_ist_now()))
        conn.commit()
        conn.close()
        return jsonify({"prediction": prediction, "probability": prob})
    except Exception as e:
        return jsonify({"error": f"Invalid diabetes input: {str(e)}"}), 400

@app.route("/predict/heart", methods=["POST"])
@jwt_required()
def predict_heart():
    current_user = get_current_user()
    data = request.json
    try:
        df_encoded = encode_heart_input(
            age=data["age"], sex=data["sex"], chest_pain=data["chest_pain"],
            resting_bp=data["resting_bp"], cholesterol=data["cholesterol"],
            fasting_bs=data["fasting_bs"], resting_ecg=data["resting_ecg"],
            max_hr=data["max_hr"], exercise_angina=data["exercise_angina"],
            oldpeak=data["oldpeak"], st_slope=data["st_slope"],
        )
        arr_scaled = heart_scaler.transform(df_encoded)
        prob = float(heart_model.predict_proba(arr_scaled)[0][1])
        prediction = int(prob > 0.5)
        conn = get_db()
        c = conn.cursor()
        c.execute("INSERT INTO predictions (user_id, disease_type, prediction, probability, timestamp) VALUES (?,?,?,?,?)",
                  (current_user["id"], "heart", prediction, prob, get_ist_now()))
        conn.commit()
        conn.close()
        return jsonify({"prediction": prediction, "probability": prob})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 400

# ================= START SERVER =================
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)