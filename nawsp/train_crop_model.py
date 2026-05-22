import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("data/Crop_recommendation.csv")

X = df.drop("label", axis=1)
y = df["label"]

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# XGBoost Model
model = XGBClassifier(
    objective="multi:softmax",
    eval_metric="mlogloss",
    use_label_encoder=False
)

# Hyperparameter Grid
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1]
}

grid = GridSearchCV(model, param_grid, cv=3)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Improved Model Accuracy:", accuracy)
print("Best Parameters:", grid.best_params_)

# Save model
joblib.dump(best_model, "models/crop_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")

print("Improved crop model saved successfully!")