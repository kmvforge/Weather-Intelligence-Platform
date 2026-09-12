import pandas as pd
from pathlib import Path

# -------------------------------------------------
# Project paths
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "kochi_weather_2015_2025.csv"

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
print("Loading weather dataset...")

df = pd.read_csv(DATA_FILE)

# -------------------------------------------------
# Basic information
# -------------------------------------------------
print("\n========== DATASET OVERVIEW ==========")

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\nColumn names:")
for column in df.columns:
    print(f"- {column}")

# -------------------------------------------------
# Data types
# -------------------------------------------------
print("\n========== DATA TYPES ==========")
print(df.dtypes)

# -------------------------------------------------
# Date range
# -------------------------------------------------
df["time"] = pd.to_datetime(df["time"])

print("\n========== DATE RANGE ==========")
print(f"Start: {df['time'].min()}")
print(f"End:   {df['time'].max()}")

# -------------------------------------------------
# Missing values
# -------------------------------------------------
print("\n========== MISSING VALUES ==========")

missing = df.isnull().sum()

print(missing)

print(f"\nTotal missing values: {missing.sum():,}")

# -------------------------------------------------
# Duplicate records
# -------------------------------------------------
print("\n========== DUPLICATES ==========")

duplicates = df.duplicated().sum()

print(f"Duplicate rows: {duplicates:,}")

# -------------------------------------------------
# Statistical summary
# -------------------------------------------------
print("\n========== STATISTICAL SUMMARY ==========")

print(df.describe())

# -------------------------------------------------
# Check unique weather codes
# -------------------------------------------------
print("\n========== WEATHER CODES ==========")

print(sorted(df["weather_code"].dropna().unique()))

# -------------------------------------------------
# Final confirmation
# -------------------------------------------------
print("\n========== INSPECTION COMPLETE ==========")
print("Dataset loaded and inspected successfully.")