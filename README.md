# 🌱 AgroSense — Irrigation Need Predictor

A production-ready FastAPI app that serves a trained **Random Forest** classifier
to predict crop irrigation need — **Low / Medium / High** — from real-time field conditions.

---

## 🌐 Live Demo

🚀 AgroSense is deployed on Render:

🔗 Live App:https://smart-irrigation-need-predictor.onrender.com

## 🌱 Project Overview

AgroSense is a Machine Learning-powered irrigation prediction system built to help farmers and agricultural analysts make smarter irrigation decisions based on environmental and crop conditions.

The application uses a trained Random Forest Classifier to predict the irrigation requirement level — Low, Medium, or High — using important agricultural parameters such as soil moisture, temperature, rainfall, wind speed, and crop growth stage.

The project is developed using FastAPI for the backend API service and includes a modern responsive web interface for real-time predictions. The trained machine learning model and preprocessing pipeline are serialized using Joblib and deployed seamlessly on Render cloud infrastructure.

AgroSense provides:
- Real-time irrigation prediction
- Probability confidence scores
- Interactive API documentation
- Responsive web UI
- Cloud deployment support
- Production-ready ML inference pipeline

This project demonstrates the complete end-to-end machine learning workflow:
1. Data preprocessing
2. Feature engineering
3. Model training & evaluation
4. Model serialization
5. FastAPI backend development
6. Frontend integration
7. Cloud deployment using Render

The system is designed to support precision agriculture by reducing water wastage and improving irrigation efficiency through AI-driven decision making.

# 🖥 Application Screenshots
## Home Page
![Home Page](screenshots/home_page.png)
## Enter skills
![Entering skills](screenshots/enter_skills.png)
## filter process
![filters](screenshots/filters.png)
## Final Result
![Rsults](screenshots/recommendation_results.png)


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

## ⚙️ Working System

AgroSense uses a trained Random Forest Machine Learning model to predict crop irrigation requirements based on agricultural and environmental conditions.

### 🔄 Workflow
1. User enters field data through the web interface:
   - Soil Moisture
   - Temperature
   - Wind Speed
   - Rainfall
   - Crop Growth Stage

2. FastAPI backend receives the input data.

3. The preprocessing pipeline transforms the data using:
   - MinMaxScaler
   - OneHotEncoder
   - ColumnTransformer

4. Processed data is passed to the trained Random Forest model.

5. The model predicts the irrigation requirement:
   - Low
   - Medium
   - High

6. Prediction results with confidence scores are returned and displayed on the web UI in real time.

---

### 🌐 System Flow

User Input → FastAPI Backend → Preprocessing → ML Model → Prediction → Result Display

---

### ☁️ Deployment Flow

GitHub Repository → Render Deployment → FastAPI Server → Live Prediction System

## 📊 Dataset

Dataset used for training the irrigation prediction model:

🔗 Kaggle Dataset:
https://www.kaggle.com/competitions/playground-series-s6e4

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

