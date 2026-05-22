import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib


def train_rainfall_model(data_path):

    # Load dataset
    df = pd.read_csv(data_path)

    # Features & Target
    X = df.drop("rainfall", axis=1)
    y = df["rainfall"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    # Prediction
    predictions = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, predictions)
    print("Model Trained Successfully")
    print("MSE:", mse)

    # Save model
    joblib.dump(model, "rainfall_model.pkl")

    return model