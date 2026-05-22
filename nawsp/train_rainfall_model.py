import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("data/rainfall_data.csv")

# Features and target
X = df.drop("rainfall", axis=1)
y = df["rainfall"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
xgb = XGBRegressor(
    objective='reg:squarederror',
    random_state=42
)

# Hyperparameter tuning
params = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1]
}

grid = GridSearchCV(
    xgb,
    params,
    cv=3,
    scoring='r2',
    verbose=1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Evaluation
y_pred = best_model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("Best Parameters:", grid.best_params_)

# Save model
joblib.dump(best_model, "models/rainfall_model.pkl")

print("Improved Rainfall model saved successfully!")