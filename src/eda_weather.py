import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

OUTPUT_DIR = PROJECT_ROOT / "screenshots" / "EDA"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
print("Loading processed weather dataset...")

df = pd.read_csv(INPUT_FILE)

df["time"] = pd.to_datetime(df["time"])

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

# -------------------------------------------------
# 1. Temperature Distribution
# -------------------------------------------------
print("\nCreating temperature distribution...")

plt.figure(figsize=(10, 6))

plt.hist(
    df["temperature_2m"],
    bins=40,
    edgecolor="black"
)

plt.title("Temperature Distribution in Kochi (2015–2025)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_temperature_distribution.png",
    dpi=300
)

plt.show()
plt.close()

# -------------------------------------------------
# 2. Monthly Average Temperature
# -------------------------------------------------
print("Creating monthly temperature analysis...")

monthly_temperature = (
    df.groupby("month")["temperature_2m"]
    .mean()
)

plt.figure(figsize=(10, 6))

plt.plot(
    monthly_temperature.index,
    monthly_temperature.values,
    marker="o"
)

plt.title("Average Monthly Temperature in Kochi")
plt.xlabel("Month")
plt.ylabel("Average Temperature (°C)")
plt.xticks(range(1, 13))
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_monthly_temperature.png",
    dpi=300
)

plt.show()
plt.close()

# -------------------------------------------------
# 3. Monthly Rainfall
# -------------------------------------------------
print("Creating monthly rainfall analysis...")

monthly_rainfall = (
    df.groupby(["year", "month"])["precipitation"]
    .sum()
    .groupby("month")
    .mean()
)

plt.figure(figsize=(10, 6))

plt.bar(
    monthly_rainfall.index,
    monthly_rainfall.values
)

plt.title("Average Monthly Precipitation in Kochi")
plt.xlabel("Month")
plt.ylabel("Average Precipitation (mm)")
plt.xticks(range(1, 13))
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_monthly_rainfall.png",
    dpi=300
)

plt.show()
plt.close()

# -------------------------------------------------
# 4. Seasonal Temperature
# -------------------------------------------------
print("Creating seasonal analysis...")

season_order = [
    "Winter",
    "Summer",
    "Monsoon",
    "Post-Monsoon"
]

seasonal_temperature = (
    df.groupby("season", observed=False)["temperature_2m"]
    .mean()
    .reindex(season_order)
)

plt.figure(figsize=(10, 6))

plt.bar(
    seasonal_temperature.index,
    seasonal_temperature.values
)

plt.title("Average Temperature by Season in Kochi")
plt.xlabel("Season")
plt.ylabel("Average Temperature (°C)")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "04_seasonal_temperature.png",
    dpi=300
)

plt.show()
plt.close()

# -------------------------------------------------
# 5. Yearly Temperature Trend
# -------------------------------------------------
print("Creating yearly temperature trend...")

yearly_temperature = (
    df.groupby("year")["temperature_2m"]
    .mean()
)

plt.figure(figsize=(11, 6))

plt.plot(
    yearly_temperature.index,
    yearly_temperature.values,
    marker="o"
)

plt.title("Yearly Average Temperature Trend in Kochi")
plt.xlabel("Year")
plt.ylabel("Average Temperature (°C)")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "05_yearly_temperature_trend.png",
    dpi=300
)

plt.show()
plt.close()

# -------------------------------------------------
# 6. Correlation Heatmap
# -------------------------------------------------
print("Creating correlation heatmap...")

numeric_columns = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m"
]

correlation = df[numeric_columns].corr()

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Weather Variable Correlation Matrix")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "06_correlation_heatmap.png",
    dpi=300
)

plt.show()
plt.close()

# -------------------------------------------------
# Final summary
# -------------------------------------------------
print("\n========== EDA COMPLETE ==========")

print("Generated visualizations:")

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"- {file.name}")

print("\nEDA analysis completed successfully.")