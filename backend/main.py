import os
from pathlib import Path
import numpy as np
import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ===============================
# Path Configurations
# ===============================
# BASE_DIR points to the 'backend' folder
BASE_DIR = Path(__file__).resolve().parent
# MODELS_DIR points to the 'models' folder in the project root
MODELS_DIR = BASE_DIR.parent / "models"
# STATIC_DIR points to the 'static' folder inside 'backend'
STATIC_DIR = BASE_DIR / "static"

# Create FastAPI instance
app = FastAPI(title="Baby Birth Weight Predictor API")

# Export for Vercel Serverless Functions
handler = app

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# Model Loading Logic
# ===============================
try:
    # Load pre-trained models and scaler from the models directory
    lgbm       = joblib.load(MODELS_DIR / "lgbm.pkl")
    xgb        = joblib.load(MODELS_DIR / "xgb.pkl")
    gb         = joblib.load(MODELS_DIR / "gb.pkl")
    meta_model = joblib.load(MODELS_DIR / "meta_model.pkl")
    scaler     = joblib.load(MODELS_DIR / "scaler.pkl")
    print("✅ Success: All models and scaler loaded successfully.")
except Exception as e:
    print(f"⚠️ Error: Failed to load models. Details: {e}")
    lgbm = xgb = gb = meta_model = scaler = None

# ===============================
# Data Schemas (Pydantic)
# ===============================
class PredictRequest(BaseModel):
    gestation: float  # Gestational age in weeks
    parity:    float  # Number of previous live births
    age:       float  # Mother's age
    height:    float  # Height in inches
    weight:    float  # Pre-pregnancy weight in pounds
    smoke:     int    # Smoking status: 1 for smoker, 0 for non-smoker

# ===============================
# Utility Functions
# ===============================
def calc_bmi(weight_lb: float, height_in: float) -> float:
    """Calculates Body Mass Index (BMI) using Imperial units."""
    kg = weight_lb * 0.453592
    m  = height_in * 0.0254
    return kg / (m ** 2) if m > 0 else 0.0

# Prediction Constants for response mapping
CLASS_LABELS = {0: "Low Weight", 1: "Normal", 2: "High Weight"}
CLASS_RANGE = {
    0: "< 88 oz (< 2.5 kg)", 
    1: "88 – 141 oz (2.5 – 4 kg)", 
    2: "> 141 oz (> 4 kg)"
}

# ===============================
# API Endpoints
# ===============================

@app.get("/")
def read_root():
    """Welcome endpoint for API status verification."""
    return {
        "status": "Online",
        "message": "Baby Birth Weight Predictor API is operational.",
        "documentation": "/docs"
    }

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Predicts the birth weight category using an ensemble stacking approach.
    Combines LightGBM, XGBoost, and GradientBoosting via a Meta-Model.
    """
    # Check if models are loaded before processing
    if any(m is None for m in [lgbm, xgb, gb, meta_model, scaler]):
        return {"error": "ML models are not initialized. Check server logs."}

    # 1. Feature Engineering: Calculate BMI
    bmi = calc_bmi(req.weight, req.height)
    
    # 2. Prepare feature array (must match training feature order)
    features = np.array([[
        req.gestation, req.parity, req.age, 
        req.height, req.weight, bmi, req.smoke
    ]])

    # 3. Scale features using the loaded scaler
    X_scaled = scaler.transform(features)

    # 4. Ensemble Stacking: Generate probabilities from base models
    try:
        base_probs = np.hstack([
            lgbm.predict_proba(X_scaled),
            xgb.predict_proba(X_scaled),
            gb.predict_proba(X_scaled),
        ])

        # 5. Final prediction using the Meta-Model (Logistic Regression or similar)
        final_probs = meta_model.predict_proba(base_probs)[0]
        predicted_class = int(np.argmax(final_probs))

        return {
            "predicted_class": predicted_class,
            "predicted_label": CLASS_LABELS[predicted_class],
            "predicted_range": CLASS_RANGE[predicted_class],
            "bmi": round(bmi, 2),
            "confidence_scores": {
                "low": round(float(final_probs[0]), 4),
                "normal": round(float(final_probs[1]), 4),
                "high": round(float(final_probs[2]), 4),
            },
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@app.get("/health")
def health_check():
    """System health check and model status."""
    return {
        "status": "active", 
        "models_ready": lgbm is not None,
        "environment": os.getenv("VERCEL_ENV", "local")
    }

# ===============================
# Static Assets & Routing
# ===============================
# Mount static folder if it exists (for CSS/JS/Images)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/favicon.ico", include_in_schema=False)
def get_favicon():
    """Handles automatic browser favicon requests."""
    favicon_path = STATIC_DIR / "favicon.ico"
    if favicon_path.exists():
        return RedirectResponse(url="/static/favicon.ico")
    return {"message": "No favicon found"}