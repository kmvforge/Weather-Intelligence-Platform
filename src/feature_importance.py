import pandas as pd
import matplotlib.pyplot as plt
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

OUTPUT_DIR = PROJECT_ROOT / "screenshots" / "Models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Load model
# -------------------------------------------------
print("Loading trained XGBoost model...")

model = joblib.load(MODEL_FILE)

# -------------------------------------------------
# Feature names
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

# -------------------------------------------------
# Extract feature importance
# -------------------------------------------------
importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

# -------------------------------------------------
# Display results
# -------------------------------------------------
print("\n========== FEATURE IMPORTANCE ==========")

for index, row in importance_df.iterrows():
    print(
        f"{index + 1:2}. "
        f"{row['Feature']:<30} "
        f"{row['Importance']:.6f}"
    )

# -------------------------------------------------
# Save importance data
# -------------------------------------------------
importance_file = (
    PROJECT_ROOT
    / "models"
    / "feature_importance.csv"
)

importance_df.to_csv(
    importance_file,
    index=False
)

# -------------------------------------------------
# Plot feature importance
# -------------------------------------------------
top_features = importance_df.head(12).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()

plot_file = (
    OUTPUT_DIR
    / "05_xgboost_feature_importance.png"
)

plt.savefig(
    plot_file,
    dpi=300
)

plt.show()
plt.close()

# -------------------------------------------------
# Final status
# -------------------------------------------------
print("\n========== FEATURE IMPORTANCE COMPLETE ==========")
print(f"Data saved to: {importance_file}")
print(f"Chart saved to: {plot_file}")