# 🌱 AgroSense — Irrigation Need Predictor

A production-ready FastAPI app that serves a trained **Random Forest** classifier
to predict crop irrigation need — **Low / Medium / High** — from real-time field conditions.

---

## 📂 Project Structure

```
agrosense/
├── main.py               ← FastAPI application & API routes
├── requirements.txt      ← Python dependencies (pinned)
├── render.yaml           ← Render one-click deployment config
├── .gitignore
├── model.pkl             ← ⬅ YOU add this (from your notebook)
├── preprocessor.pkl      ← ⬅ YOU add this (from your notebook)
├── static/               ← (optional) CSS / JS / images
└── templates/
    └── index.html        ← AgroSense web UI
```

---

## 🚀 Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your trained artifacts in the project root
#    model.pkl  +  preprocessor.pkl

# 3. Start the server
uvicorn main:app --reload

# 4. Open http://localhost:8000
```

---

## ☁️ Deploy on Render

### Option A — Blueprint (recommended)
1. Push this folder to a **GitHub repo** (include `model.pkl` & `preprocessor.pkl`).
2. Render Dashboard → **New → Blueprint** → connect your repo.
3. Render reads `render.yaml` automatically — done!

### Option B — Manual Web Service
1. **New → Web Service** → connect repo.
2. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. Environment Variables:
   | Key | Value |
   |-----|-------|
   | `MODEL_PATH` | `model.pkl` |
   | `PREPROCESSOR_PATH` | `preprocessor.pkl` |

> ⚠️ Render free tier has an **ephemeral filesystem** — files reset on redeploy.
> Commit your `.pkl` files to the repo, or use a **Render Disk** for persistence.

---

## 📡 API Reference

### `GET /`
Returns the AgroSense web UI.

### `GET /health`
```json
{ "status": "ok", "model_loaded": true, "preprocessor_loaded": true }
```

### `GET /crop-stages`
```json
{ "crop_stages": ["Sowing", "Germination", "Vegetative", ...] }
```

### `POST /predict`

**Request:**
```json
{
  "soil_moisture": 32.58,
  "temperature_c": 15.01,
  "wind_speed_kmh": 16.79,
  "rainfall_mm": 725.99,
  "crop_growth_stage": "Sowing"
}
```

**Response:**
```json
{
  "prediction": "Low",
  "confidence": 0.82,
  "probabilities": { "Low": 0.82, "Medium": 0.13, "High": 0.05 },
  "input_summary": {
    "Soil Moisture": "32.58%",
    "Temperature": "15.01°C",
    "Wind Speed": "16.79 km/h",
    "Rainfall": "725.99 mm",
    "Crop Stage": "Sowing"
  }
}
```

Interactive docs at `/docs` (Swagger) and `/redoc`.

---

## 🧠 Model Details

| Item | Value |
|------|-------|
| Algorithm | Random Forest Classifier |
| Classes | Low (0) · Medium (1) · High (2) |
| Balancing | SMOTE oversampling |
| Preprocessing | MinMaxScaler + OneHotEncoder (ColumnTransformer) |
| Features | Soil_Moisture, Temperature_C, Wind_Speed_kmh, Rainfall_mm, Crop_Growth_Stage |

---

## 💾 Export Artifacts (add to end of notebook)

```python
import joblib
joblib.dump(final_model, 'model.pkl')
joblib.dump(processer, 'preprocessor.pkl')   # your variable name in the notebook
```

Copy both files into this project root before deploying.
