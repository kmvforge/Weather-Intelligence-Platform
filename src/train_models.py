import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

# -------------------------------------------------
# Project paths
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "kochi_weather_ml.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = PROJECT_ROOT / "screenshots" / "Models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Load ML dataset
# -------------------------------------------------
print("Loading ML dataset...")

df = pd.read_csv(INPUT_FILE)
df["time"] = pd.to_datetime(df["time"])

print(f"Total rows: {len(df):,}")

# -------------------------------------------------
# Feature and target columns
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

X = df[feature_columns]
y = df[target_column]

# -------------------------------------------------
# Time-series train/test split
# -------------------------------------------------
print("\nCreating chronological train/test split...")

train_mask = df["time"] < "2024-01-01"

X_train = X[train_mask]
y_train = y[train_mask]

X_test = X[~train_mask]
y_test = y[~train_mask]

print(f"Training rows: {len(X_train):,}")
print(f"Testing rows:  {len(X_test):,}")

print(
    f"Training period: "
    f"{df.loc[train_mask, 'time'].min()} → "
    f"{df.loc[train_mask, 'time'].max()}"
)

print(
    f"Testing period:  "
    f"{df.loc[~train_mask, 'time'].min()} → "
    f"{df.loc[~train_mask, 'time'].max()}"
)

# -------------------------------------------------
# Define models
# -------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror"
    )
}

# -------------------------------------------------
# Train and evaluate
# -------------------------------------------------
results = []
predictions = {}

print("\n========== MODEL TRAINING ==========")

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions_test = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions_test))
    r2 = r2_score(y_test, predictions_test)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    predictions[name] = predictions_test

    print(f"MAE:  {mae:.4f} °C")
    print(f"RMSE: {rmse:.4f} °C")
    print(f"R²:   {r2:.4f}")

# -------------------------------------------------
# Results table
# -------------------------------------------------
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE"
).reset_index(drop=True)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# -------------------------------------------------
# Save results
# -------------------------------------------------
results_file = OUTPUT_DIR / "model_comparison.csv"

results_df.to_csv(
    results_file,
    index=False
)

# -------------------------------------------------
# Select best model
# -------------------------------------------------
best_model_name = results_df.iloc[0]["Model"]

print(f"\nBest model based on MAE: {best_model_name}")

# -------------------------------------------------
# Plot model comparison
# -------------------------------------------------
plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["MAE"]
)

plt.title("Model Comparison - Mean Absolute Error")
plt.xlabel("Model")
plt.ylabel("MAE (°C)")
plt.tight_layout()

comparison_plot = (
    RESULTS_DIR / "02_model_comparison_mae.png"
)

plt.savefig(
    comparison_plot,
    dpi=300
)

plt.show()
plt.close()

# -------------------------------------------------
# Plot actual vs predicted for best model
# -------------------------------------------------
best_predictions = predictions[best_model_name]

plot_points = min(500, len(y_test))

plt.figure(figsize=(12, 6))

plt.plot(
    y_test.iloc[:plot_points].values,
    label="Actual"
)

plt.plot(
    best_predictions[:plot_points],
    label="Predicted"
)

plt.title(
    f"Actual vs Predicted Temperature - {best_model_name}"
)

plt.xlabel("Test Observation")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.tight_layout()

prediction_plot = (
    RESULTS_DIR / "03_actual_vs_predicted.png"
)

plt.savefig(
    prediction_plot,
    dpi=300
)

plt.show()
plt.close()

# -------------------------------------------------
# Save the best model
# -------------------------------------------------
best_model = models[best_model_name]

model_filename = (
    best_model_name
    .lower()
    .replace(" ", "_")
    .replace("²", "2")
    + "_weather_model.joblib"
)

try:
    import joblib

    joblib.dump(
        best_model,
        OUTPUT_DIR / model_filename
    )

    print(
        f"\nBest model saved to: "
        f"{OUTPUT_DIR / model_filename}"
    )

except ImportError:
    print("\njoblib is not available. Model file was not saved.")

# -------------------------------------------------
# Final status
# -------------------------------------------------
print("\n========== MODEL TRAINING COMPLETE ==========")
print(f"Results saved to: {results_file}")
print(f"Best model: {best_model_name}")