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

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "kochi_weather_ml.csv"
)

# -------------------------------------------------
# Load model and dataset
# -------------------------------------------------
print("Loading trained XGBoost model...")
model = joblib.load(MODEL_FILE)

print("Loading ML dataset...")
df = pd.read_csv(DATA_FILE)

df["time"] = pd.to_datetime(df["time"])

# -------------------------------------------------
# Feature columns
# -------------------------------------------------
feature_columns = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "year",
    "month",
    "day",
    "hour",
    "day_of_year",
    "temperature_lag_1",
    "temperature_lag_3",
    "temperature_lag_24",
    "humidity_lag_1"
]

target_column = "target_temperature_next_hour"

# -------------------------------------------------
# Select a test observation
# -------------------------------------------------
# Use the first observation from the 2025 test period
test_data = df[df["time"] >= "2025-01-01"].iloc[0]

X_input = test_data[feature_columns].to_frame().T

# Ensure all model inputs are numeric
X_input = X_input.apply(pd.to_numeric, errors="coerce")

actual_temperature = test_data[target_column]
current_temperature = test_data["temperature_2m"]

prediction = model.predict(X_input)[0]

# -------------------------------------------------
# Display prediction
# -------------------------------------------------
print("\n========== WEATHER PREDICTION ==========")

print(f"Observation time:       {test_data['time']}")
print(f"Current temperature:    {current_temperature:.2f} °C")
print(f"Actual next-hour temp:  {actual_temperature:.2f} °C")
print(f"Predicted next-hour:    {prediction:.2f} °C")

error = abs(actual_temperature - prediction)

print(f"Absolute error:         {error:.2f} °C")

# -------------------------------------------------
# Final status
# -------------------------------------------------
print("\n========== PREDICTION TEST COMPLETE ==========")
print("XGBoost successfully generated a next-hour temperature prediction.")