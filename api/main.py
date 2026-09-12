from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path


# -------------------------------------------------
# Project paths
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "xgboost_weather_model.joblib"
)


# -------------------------------------------------
# Load trained model
# -------------------------------------------------

model = joblib.load(MODEL_FILE)


# -------------------------------------------------
# Create FastAPI application
# -------------------------------------------------

app = FastAPI(
    title="Weather Intelligence API",
    description="API for next-hour temperature prediction",
    version="1.0.0"
)


# -------------------------------------------------
# Input data model
# -------------------------------------------------

class WeatherInput(BaseModel):

    temperature_2m: float
    relative_humidity_2m: float
    precipitation: float
    surface_pressure: float
    cloud_cover: float
    wind_speed_10m: float
    wind_direction_10m: float
    wind_gusts_10m: float

    year: int
    month: int
    day: int
    hour: int
    day_of_year: int

    temperature_lag_1: float
    temperature_lag_3: float
    temperature_lag_24: float
    humidity_lag_1: float


# -------------------------------------------------
# Root endpoint
# -------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Weather Intelligence API is running"
    }


# -------------------------------------------------
# Health check endpoint
# -------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "XGBoost"
    }


# -------------------------------------------------
# Prediction endpoint
# -------------------------------------------------

@app.post("/predict")
def predict_weather(data: WeatherInput):

    input_data = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_data)[0]

    return {
        "predicted_temperature": round(float(prediction), 2),
        "unit": "°C"
    }