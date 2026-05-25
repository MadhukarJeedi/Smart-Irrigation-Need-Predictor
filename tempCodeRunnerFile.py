
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