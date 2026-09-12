import requests
import pandas as pd
from pathlib import Path

# -------------------------------------------------
# Project paths
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "kochi_weather_2015_2025.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Location: Kochi, Kerala
# -------------------------------------------------
LATITUDE = 9.9312
LONGITUDE = 76.2673

# -------------------------------------------------
# Historical period
# -------------------------------------------------
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

# -------------------------------------------------
# Weather variables
# -------------------------------------------------
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "weather_code",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m"
]

# -------------------------------------------------
# Open-Meteo Historical Weather API
# -------------------------------------------------
URL = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": ",".join(HOURLY_VARIABLES),
    "timezone": "Asia/Kolkata",
    "temperature_unit": "celsius",
    "wind_speed_unit": "kmh",
    "precipitation_unit": "mm"
}

print("Downloading historical weather data...")
print(f"Location: Kochi, Kerala")
print(f"Period: {START_DATE} to {END_DATE}")

response = requests.get(URL, params=params, timeout=60)

if response.status_code != 200:
    print("Error while downloading data.")
    print(response.text)
    raise SystemExit(1)

data = response.json()

# -------------------------------------------------
# Convert API response to DataFrame
# -------------------------------------------------
hourly = data["hourly"]

df = pd.DataFrame(hourly)

# -------------------------------------------------
# Save dataset
# -------------------------------------------------
df.to_csv(OUTPUT_FILE, index=False)

print("\nDataset downloaded successfully.")
print(f"Saved to: {OUTPUT_FILE}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\nColumns:")
for column in df.columns:
    print(f"- {column}")

print("\nFirst 5 rows:")
print(df.head())