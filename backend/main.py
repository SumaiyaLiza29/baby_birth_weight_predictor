from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import numpy as np
import joblib
import os

# Create FastAPI instance
app = FastAPI(title="Baby Birth Weight Predictor API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (you can specify your frontend URL for more security)
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# ===============================
# Load Models
# ===============================
BASE = os.path.join(os.path.dirname(__file__), "..", "models")

try:
    # Load the models
    lgbm       = joblib.load(os.path.join(BASE, "lgbm.pkl"))
    xgb        = joblib.load(os.path.join(BASE, "xgb.pkl"))
    gb         = joblib.load(os.path.join(BASE, "gb.pkl"))
    meta_model = joblib.load(os.path.join(BASE, "meta_model.pkl"))
    scaler     = joblib.load(os.path.join(BASE, "scaler.pkl"))
    print("✅ All models loaded successfully.")
except Exception as e:
    print(f"⚠️ Failed to load models: {e}")
    lgbm = xgb = gb = meta_model = scaler = None

# ===============================
# Request Schema (Input Format)
# ===============================
class PredictRequest(BaseModel):
    gestation: float  # Weeks
    parity:    float  # Number of previous children
    age:       float  # Mother's age
    height:    float  # Height in inches
    weight:    float  # Weight in pounds
    smoke:     int    # 0 or 1 (whether the mother smokes)

# ===============================
# Helper Function (BMI Calculation)
# ===============================
def calc_bmi(weight_lb: float, height_in: float) -> float:
    """
    Calculate BMI based on weight (lbs) and height (inches).
    """
    kg = weight_lb * 0.453592  # Convert weight to kg
    m  = height_in * 0.0254    # Convert height to meters
    return kg / (m ** 2) if m > 0 else 0.0

# Class Labels and Ranges for Prediction
CLASS_LABELS = {
    0: "Low Weight",
    1: "Normal",
    2: "High Weight",
}

CLASS_RANGE = {
    0: "< 88 oz  (< 2.5 kg)",
    1: "88 – 141 oz  (2.5 – 4 kg)",
    2: "> 141 oz  (> 4 kg)",
}

# ===============================
# Predict Endpoint
# ===============================
@app.post("/predict")
def predict(req: PredictRequest):
    """
    Predicts the birth weight category based on the input features.
    """
    if any(m is None for m in [lgbm, xgb, gb, meta_model, scaler]):
        return {"error": "Models are not loaded. Please check the models/ folder."}

    # Calculate BMI
    bmi = calc_bmi(req.weight, req.height)

    # Prepare features for prediction
    features = np.array([[ 
        req.gestation,
        req.parity,
        req.age,
        req.height,
        req.weight,
        bmi,
        req.smoke,
    ]])

    # Scale the features
    X_scaled = scaler.transform(features)

    # Stacking: Use base models to predict probabilities, then use meta-model
    meta_input = np.hstack([
        lgbm.predict_proba(X_scaled),
        xgb.predict_proba(X_scaled),
        gb.predict_proba(X_scaled),
    ])

    # Get probabilities from the meta-model
    probs     = meta_model.predict_proba(meta_input)[0]
    pred_cls  = int(np.argmax(probs))  # Get the class with the highest probability

    # Return the prediction and probabilities
    return {
        "predicted_class":       pred_cls,
        "predicted_label":       CLASS_LABELS[pred_cls],
        "predicted_range":       CLASS_RANGE[pred_cls],
        "bmi":                   round(bmi, 2),
        "probabilities": {
            "low":    round(float(probs[0]), 4),
            "normal": round(float(probs[1]), 4),
            "high":   round(float(probs[2]), 4),
        },
    }

# ===============================
# Health Check Endpoint
# ===============================
@app.get("/health")
def health():
    """
    Simple health check to confirm if the server and models are loaded.
    """
    return {"status": "ok", "models_loaded": lgbm is not None}

# ===============================
# Favicon Setup (Optional)
# ===============================
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

# Mount static files (if you want to serve a favicon or other static assets)
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

# Favicon endpoint
@app.get("/favicon.ico")
def favicon():
    return RedirectResponse(url="/static/favicon.ico")