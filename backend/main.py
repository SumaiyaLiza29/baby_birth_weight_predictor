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

# Vercel handler export
handler = app

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# Model Loading Logic
# ===============================
try:
    # Load serialized models and scaler using absolute paths
    lgbm       = joblib.load(MODELS_DIR / "lgbm.pkl")
    xgb        = joblib.load(MODELS_DIR / "xgb.pkl")
    gb         = joblib.load(MODELS_DIR / "gb.pkl")
    meta_model = joblib.load(MODELS_DIR / "meta_model.pkl")
    scaler     = joblib.load(MODELS_DIR / "scaler.pkl")
    print("✅ All models loaded successfully.")
except Exception as e:
    print(f"⚠️ Failed to load models: {e}")
    lgbm = xgb = gb = meta_model = scaler = None

# ===============================
# Data Schemas
# ===============================
class PredictRequest(BaseModel):
    gestation: float  # Measured in weeks
    parity:    float  # Number of previous births
    age:       float  # Mother's age in years
    height:    float  # Height in inches
    weight:    float  # Pre-pregnancy weight in pounds
    smoke:     int    # Binary: 1 for smoker, 0 for non-smoker

# ===============================
# Utility Functions
# ===============================
def calc_bmi(weight_lb: float, height_in: float) -> float:
    """Calculates BMI from Imperial units."""
    kg = weight_lb * 0.453592
    m  = height_in * 0.0254
    return kg / (m ** 2) if m > 0 else 0.0

# Prediction Constants
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
    """Root endpoint for deployment verification."""
    return {
        "message": "Baby Birth Weight Predictor API is online.",
        "docs": "/docs"
    }

@app.post("/predict")
def predict(req: PredictRequest):
    """Predicts birth weight category using an ensemble stacking model."""
    if any(m is None for m in [lgbm, xgb, gb, meta_model, scaler]):
        return {"error": "Machine Learning models are not initialized."}

    # Feature Engineering
    bmi = calc_bmi(req.weight, req.height)
    features = np.array([[
        req.gestation, req.parity, req.age, 
        req.height, req.weight, bmi, req.smoke
    ]])

    # Data Transformation
    X_scaled = scaler.transform(features)

    # Ensemble Stacking: Aggregate base model probabilities
    meta_input = np.hstack([
        lgbm.predict_proba(X_scaled),
        xgb.predict_proba(X_scaled),
        gb.predict_proba(X_scaled),
    ])

    # Final prediction via Meta-Model
    probs = meta_model.predict_proba(meta_input)[0]
    pred_cls = int(np.argmax(probs))

    return {
        "predicted_class": pred_cls,
        "predicted_label": CLASS_LABELS[pred_cls],
        "predicted_range": CLASS_RANGE[pred_cls],
        "bmi": round(bmi, 2),
        "probabilities": {
            "low": round(float(probs[0]), 4),
            "normal": round(float(probs[1]), 4),
            "high": round(float(probs[2]), 4),
        },
    }

@app.get("/health")
def health():
    """Health check endpoint for monitoring."""
    return {"status": "active", "models_loaded": lgbm is not None}

# ===============================
# Static Assets & Routing
# ===============================
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Handles favicon requests to prevent 404 errors."""
    return RedirectResponse(url="/static/favicon.ico")