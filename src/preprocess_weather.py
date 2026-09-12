import pandas as pd
from pathlib import Path

# -------------------------------------------------
# Project paths
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "kochi_weather_2015_2025.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "kochi_weather_processed.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Load raw dataset
# -------------------------------------------------
print("Loading raw weather dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Original rows: {len(df):,}")
print(f"Original columns: {len(df.columns)}")

# -------------------------------------------------
# Convert time column
# -------------------------------------------------
df["time"] = pd.to_datetime(df["time"])

# -------------------------------------------------
# Create time-based features
# -------------------------------------------------
df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month
df["day"] = df["time"].dt.day
df["hour"] = df["time"].dt.hour
df["day_of_year"] = df["time"].dt.dayofyear

# -------------------------------------------------
# Create season feature
# -------------------------------------------------
def get_season(month):
    if month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    elif month in [10, 11]:
        return "Post-Monsoon"
    else:
        return "Winter"


df["season"] = df["month"].apply(get_season)

# -------------------------------------------------
# Reorder columns
# -------------------------------------------------
first_columns = [
    "time",
    "year",
    "month",
    "day",
    "hour",
    "day_of_year",
    "season"
]

remaining_columns = [
    column for column in df.columns
    if column not in first_columns
]

df = df[first_columns + remaining_columns]

# -------------------------------------------------
# Final data-quality check
# -------------------------------------------------
print("\n========== PREPROCESSING CHECK ==========")

print(f"Rows after preprocessing: {len(df):,}")
print(f"Columns after preprocessing: {len(df.columns)}")
print(f"Missing values: {df.isnull().sum().sum():,}")
print(f"Duplicate rows: {df.duplicated().sum():,}")

# -------------------------------------------------
# Display sample
# -------------------------------------------------
print("\n========== SAMPLE DATA ==========")
print(df.head())

# -------------------------------------------------
# Display new features
# -------------------------------------------------
print("\n========== NEW FEATURES ==========")

print(df[
    [
        "time",
        "year",
        "month",
        "day",
        "hour",
        "day_of_year",
        "season"
    ]
].head())

# -------------------------------------------------
# Save processed dataset
# -------------------------------------------------
df.to_csv(OUTPUT_FILE, index=False)

print("\n========== PREPROCESSING COMPLETE ==========")
print(f"Processed dataset saved to:")
print(OUTPUT_FILE)