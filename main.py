from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os

# ─── App ────────────────────────────────────────────────────
app = FastAPI(
    title="AgroSense — Irrigation Need Predictor",
    description="Predict crop irrigation need (Low / Medium / High) from real-time field conditions.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ─── Constants ──────────────────────────────────────────────
CROP_STAGES = [
    "Sowing", "Germination", "Vegetative",
    "Flowering", "Fruiting", "Maturation", "Harvesting",
]
LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}

MODEL_PATH       = os.getenv("MODEL_PATH",       "model.pkl")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "preprocessor.pkl")

model        = None
preprocessor = None

# ─── Startup ─────────────────────────────────────────────────
@app.on_event("startup")
def load_artifacts():
    global model, preprocessor
    try:
        model        = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("✅  Model and preprocessor loaded.")
    except Exception as exc:
        print(f"⚠️   Could not load artifacts: {exc}")
        print("    Copy model.pkl and preprocessor.pkl into the project root.")

# ─── Schemas ─────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    soil_moisture:     float = Field(..., ge=0,   le=100,  description="Soil moisture (%)")
    temperature_c:     float = Field(..., ge=-10, le=60,   description="Temperature (°C)")
    wind_speed_kmh:    float = Field(..., ge=0,   le=300,  description="Wind speed (km/h)")
    rainfall_mm:       float = Field(..., ge=0,   le=5000, description="Rainfall (mm)")
    crop_growth_stage: str   = Field(..., description=f"One of: {', '.join(CROP_STAGES)}")

    model_config = {
        "json_schema_extra": {
            "example": {
                "soil_moisture": 32.58,
                "temperature_c": 15.01,
                "wind_speed_kmh": 16.79,
                "rainfall_mm": 725.99,
                "crop_growth_stage": "Sowing",
            }
        }
    }

class PredictionResponse(BaseModel):
    prediction:    str
    confidence:    float
    probabilities: dict
    input_summary: dict

# ─── Routes ──────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "crop_stages": CROP_STAGES},
    )

@app.get("/health", tags=["Utility"])
def health():
    return {
        "status": "ok",
        "model_loaded":        model is not None,
        "preprocessor_loaded": preprocessor is not None,
    }

@app.get("/crop-stages", tags=["Utility"])
def get_crop_stages():
    return {"crop_stages": CROP_STAGES}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(payload: PredictionRequest):
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not loaded. Add model.pkl and preprocessor.pkl to the project root.",
        )
    if payload.crop_growth_stage not in CROP_STAGES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid crop_growth_stage. Must be one of: {CROP_STAGES}",
        )

    input_df = pd.DataFrame([{
        "Soil_Moisture":     payload.soil_moisture,
        "Temperature_C":     payload.temperature_c,
        "Wind_Speed_kmh":    payload.wind_speed_kmh,
        "Rainfall_mm":       payload.rainfall_mm,
        "Crop_Growth_Stage": payload.crop_growth_stage,
    }])

    try:
        processed  = preprocessor.transform(input_df)
        pred_class = int(model.predict(processed)[0])
        proba      = model.predict_proba(processed)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    probabilities = {LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(proba)}

    return PredictionResponse(
        prediction=LABEL_MAP[pred_class],
        confidence=round(float(proba[pred_class]), 4),
        probabilities=probabilities,
        input_summary={
            "Soil Moisture": f"{payload.soil_moisture}%",
            "Temperature":   f"{payload.temperature_c}°C",
            "Wind Speed":    f"{payload.wind_speed_kmh} km/h",
            "Rainfall":      f"{payload.rainfall_mm} mm",
            "Crop Stage":    payload.crop_growth_stage,
        },
    )
