import pandas as pd
from pathlib import Path

# -------------------------------------------------
# Project paths
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "kochi_weather_processed.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "kochi_weather_ml.csv"

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
print("Loading processed weather dataset...")

df = pd.read_csv(INPUT_FILE)

df["time"] = pd.to_datetime(df["time"])

print(f"Original rows: {len(df):,}")

# -------------------------------------------------
# Create historical lag features
# -------------------------------------------------
print("\nCreating historical weather features...")

# Temperature from previous hour
df["temperature_lag_1"] = df["temperature_2m"].shift(1)

# Temperature from 3 hours earlier
df["temperature_lag_3"] = df["temperature_2m"].shift(3)

# Temperature from 24 hours earlier
df["temperature_lag_24"] = df["temperature_2m"].shift(24)

# Humidity from previous hour
df["humidity_lag_1"] = df["relative_humidity_2m"].shift(1)

# -------------------------------------------------
# Create forecasting target
# -------------------------------------------------
print("Creating next-hour temperature target...")

df["target_temperature_next_hour"] = (
    df["temperature_2m"].shift(-1)
)

# -------------------------------------------------
# Select ML features
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
# Keep only ML columns
# -------------------------------------------------
ml_df = df[
    ["time"] + feature_columns + [target_column]
].copy()

# -------------------------------------------------
# Remove rows created by lag/target operations
# -------------------------------------------------
print("\nRemoving rows with unavailable lag/target values...")

before = len(ml_df)

ml_df = ml_df.dropna().reset_index(drop=True)

after = len(ml_df)

print(f"Rows before cleaning: {before:,}")
print(f"Rows after cleaning:  {after:,}")
print(f"Rows removed:        {before - after:,}")

# -------------------------------------------------
# Check for missing values
# -------------------------------------------------
missing_values = ml_df.isnull().sum().sum()

print(f"\nRemaining missing values: {missing_values}")

# -------------------------------------------------
# Check chronological order
# -------------------------------------------------
is_sorted = ml_df["time"].is_monotonic_increasing

print(f"Chronological order maintained: {is_sorted}")

# -------------------------------------------------
# Display ML dataset information
# -------------------------------------------------
print("\n========== ML DATASET ==========")

print(f"Rows: {len(ml_df):,}")
print(f"Columns: {len(ml_df.columns)}")

print("\nFeatures:")
for feature in feature_columns:
    print(f"- {feature}")

print(f"\nTarget:")
print(f"- {target_column}")

print("\nFirst 5 rows:")
print(ml_df.head())

# -------------------------------------------------
# Save ML dataset
# -------------------------------------------------
ml_df.to_csv(OUTPUT_FILE, index=False)

print("\n========== ML PREPARATION COMPLETE ==========")
print("ML-ready dataset saved to:")
print(OUTPUT_FILE)