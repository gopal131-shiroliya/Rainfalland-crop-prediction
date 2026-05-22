import joblib
import matplotlib.pyplot as plt
import numpy as np

model = joblib.load("models/rainfall_model.pkl")

importance = model.feature_importances_
features = ["temperature", "humidity", "wind_speed"]

# Convert to percentage
importance_percent = 100 * (importance / importance.sum())

for f, imp in zip(features, importance_percent):
    print(f"{f}: {imp:.2f}%")

plt.figure()
plt.barh(features, importance_percent)
plt.xlabel("Importance (%)")
plt.title("Rainfall Model Feature Importance (%)")
plt.show()