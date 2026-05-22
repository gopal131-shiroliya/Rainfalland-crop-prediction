import csv
import random
from pathlib import Path

import joblib
import numpy as np
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent / "nawsp"
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
CROP_DATA_PATH = DATA_DIR / "Crop_recommendation.csv"


@st.cache_resource
def load_models():
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    rainfall_model = joblib.load(MODEL_DIR / "rainfall_model.pkl")
    crop_model = joblib.load(MODEL_DIR / "crop_model.pkl")
    return label_encoder, rainfall_model, crop_model


@st.cache_data
def load_crop_rows():
    rows = []
    with CROP_DATA_PATH.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            rows.append(
                {
                    "N": float(row["N"]),
                    "P": float(row["P"]),
                    "K": float(row["K"]),
                    "temperature": float(row["temperature"]),
                    "humidity": float(row["humidity"]),
                    "ph": float(row["ph"]),
                    "rainfall": float(row["rainfall"]),
                    "label": row["label"],
                }
            )
    return rows


def estimate_rainfall_from_dataset(rows, temperature, humidity, k=7):
    ranked = sorted(
        rows,
        key=lambda row: ((row["temperature"] - temperature) ** 2)
        + ((row["humidity"] - humidity) ** 2),
    )
    neighbours = ranked[: max(1, min(k, len(ranked)))]
    avg_rainfall = sum(row["rainfall"] for row in neighbours) / len(neighbours)
    return round(float(avg_rainfall), 2)


def get_crop_recommendations(label_encoder, crop_model, crop_input, top_n=3):
    primary_encoded = int(crop_model.predict(crop_input)[0])
    primary_crop = str(label_encoder.inverse_transform([primary_encoded])[0])

    recommendations = [{"crop": primary_crop, "confidence": 1.0}]
    if hasattr(crop_model, "predict_proba") and hasattr(crop_model, "classes_"):
        probabilities = crop_model.predict_proba(crop_input)[0]
        sorted_idx = np.argsort(probabilities)[::-1][:top_n]
        recommendations = []
        for idx in sorted_idx:
            encoded_class = int(crop_model.classes_[idx])
            crop_name = str(label_encoder.inverse_transform([encoded_class])[0])
            recommendations.append(
                {"crop": crop_name, "confidence": round(float(probabilities[idx]), 4)}
            )

    return primary_crop, recommendations


st.set_page_config(page_title="Rainfall and Crop Prediction")
st.title("Rainfall and Crop Prediction")

try:
    label_encoder, rainfall_model, crop_model = load_models()
    crop_rows = load_crop_rows()
except Exception as exc:
    st.error(f"App setup failed: {exc}")
    st.stop()

sample = random.choice(crop_rows)

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        n_value = st.number_input("Nitrogen (N)", min_value=0.0, value=float(sample["N"]))
        temperature = st.number_input(
            "Temperature", min_value=-20.0, max_value=70.0, value=float(sample["temperature"])
        )
    with col2:
        p_value = st.number_input("Phosphorus (P)", min_value=0.0, value=float(sample["P"]))
        humidity = st.number_input(
            "Humidity", min_value=0.0, max_value=100.0, value=float(sample["humidity"])
        )
    with col3:
        k_value = st.number_input("Potassium (K)", min_value=0.0, value=float(sample["K"]))
        ph_value = st.number_input("pH", min_value=0.0, max_value=14.0, value=float(sample["ph"]))

    wind_speed = st.number_input("Wind Speed", min_value=0.0, value=10.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    rainfall_features = np.array([[temperature, humidity, wind_speed]])
    model_rainfall = rainfall_model.predict(rainfall_features)[0]
    model_rainfall = round(max(0.0, float(model_rainfall)), 2)
    estimated_rainfall = estimate_rainfall_from_dataset(crop_rows, temperature, humidity)

    crop_features = np.array(
        [[n_value, p_value, k_value, temperature, humidity, ph_value, estimated_rainfall]]
    )
    recommended_crop, recommendations = get_crop_recommendations(
        label_encoder, crop_model, crop_features
    )

    st.subheader("Prediction Result")
    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Predicted Rainfall", f"{estimated_rainfall} mm")
    metric_col2.metric("Recommended Crop", recommended_crop.title())

    st.caption(f"Rainfall model output: {model_rainfall} mm")
    st.write("Top crop recommendations")
    for item in recommendations:
        st.write(f"- {item['crop'].title()} ({item['confidence'] * 100:.2f}%)")
