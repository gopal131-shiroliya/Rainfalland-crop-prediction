import csv
import io
import random
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .database import get_connection, init_db

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
CROP_DATA_PATH = DATA_DIR / "Crop_recommendation.csv"

app = FastAPI(title="Rainfall and Crop Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
rainfall_model = joblib.load(MODEL_DIR / "rainfall_model.pkl")
crop_model = joblib.load(MODEL_DIR / "crop_model.pkl")


def _load_crop_rows():
    if not CROP_DATA_PATH.exists():
        return []

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


CROP_ROWS = _load_crop_rows()


def estimate_rainfall_from_dataset(temperature: float, humidity: float, k: int = 7) -> float:
    if not CROP_ROWS:
        return 0.0

    ranked = sorted(
        CROP_ROWS,
        key=lambda row: ((row["temperature"] - temperature) ** 2) + ((row["humidity"] - humidity) ** 2),
    )
    neighbours = ranked[: max(1, min(k, len(ranked)))]
    avg_rainfall = sum(row["rainfall"] for row in neighbours) / len(neighbours)
    return round(float(avg_rainfall), 2)


def get_top_crop_recommendations(crop_input: np.ndarray, top_n: int = 3):
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
                {
                    "crop": crop_name,
                    "confidence": round(float(probabilities[idx]), 4),
                }
            )

    return primary_crop, recommendations


class FullPredictionInput(BaseModel):
    N: float = Field(..., ge=0)
    P: float = Field(..., ge=0)
    K: float = Field(..., ge=0)
    temperature: float = Field(..., ge=-20, le=70)
    humidity: float = Field(..., ge=0, le=100)
    ph: float = Field(..., ge=0, le=14)
    wind_speed: float = Field(..., ge=0)


class RainfallInput(BaseModel):
    temperature: float = Field(..., ge=-20, le=70)
    humidity: float = Field(..., ge=0, le=100)
    wind_speed: float = Field(..., ge=0)


@app.get("/")
def home():
    return {"message": "Rainfall and Crop Prediction API is running"}


@app.get("/sample-input")
def sample_input():
    if not CROP_ROWS:
        raise HTTPException(status_code=500, detail="Crop dataset not found or empty")

    sample = random.choice(CROP_ROWS)
    return {
        "N": round(sample["N"], 2),
        "P": round(sample["P"], 2),
        "K": round(sample["K"], 2),
        "temperature": round(sample["temperature"], 2),
        "humidity": round(sample["humidity"], 2),
        "ph": round(sample["ph"], 2),
        "wind_speed": 10.0,
        "dataset_rainfall_mm": round(sample["rainfall"], 2),
        "dataset_crop_label": sample["label"],
    }


@app.post("/predict-rainfall")
def predict_rainfall(data: RainfallInput):
    features = np.array([[data.temperature, data.humidity, data.wind_speed]])
    model_prediction = rainfall_model.predict(features)
    model_rainfall = round(max(0.0, float(model_prediction[0])), 2)
    dataset_estimated_rainfall = estimate_rainfall_from_dataset(data.temperature, data.humidity)

    return {
        "temperature": data.temperature,
        "humidity": data.humidity,
        "wind_speed": data.wind_speed,
        "predicted_rainfall_mm": dataset_estimated_rainfall,
        "rainfall_model_mm": model_rainfall,
        "dataset_estimated_rainfall_mm": dataset_estimated_rainfall,
    }


@app.post("/predict")
def full_prediction(data: FullPredictionInput):
    rainfall_input = np.array([[data.temperature, data.humidity, data.wind_speed]])
    model_rainfall = rainfall_model.predict(rainfall_input)[0]
    model_rainfall = round(max(0.0, float(model_rainfall)), 2)
    dataset_estimated_rainfall = estimate_rainfall_from_dataset(data.temperature, data.humidity)

    # Crop model was trained on crop dataset rainfall range, so use dataset-estimated rainfall.
    crop_rainfall = dataset_estimated_rainfall

    crop_input = np.array(
        [[data.N, data.P, data.K, data.temperature, data.humidity, data.ph, crop_rainfall]]
    )

    recommended_crop, top_recommendations = get_top_crop_recommendations(crop_input)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions
        (N, P, K, temperature, humidity, ph, wind_speed, rainfall_prediction, recommended_crop)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            data.N,
            data.P,
            data.K,
            data.temperature,
            data.humidity,
            data.ph,
            data.wind_speed,
            crop_rainfall,
            recommended_crop,
        ),
    )
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return {
        "id": prediction_id,
        "temperature": data.temperature,
        "humidity": data.humidity,
        "wind_speed": data.wind_speed,
        "N": data.N,
        "P": data.P,
        "K": data.K,
        "ph": data.ph,
        "predicted_rainfall_mm": crop_rainfall,
        "rainfall_prediction": crop_rainfall,
        "rainfall_model_mm": model_rainfall,
        "dataset_estimated_rainfall_mm": dataset_estimated_rainfall,
        "recommended_crop": recommended_crop,
        "top_crop_recommendations": top_recommendations,
    }


@app.get("/history")
def get_history(
    limit: int = 100,
    crop: Optional[str] = None,
):
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 1000")

    conn = get_connection()
    cursor = conn.cursor()

    base_query = """
        SELECT
            id,
            created_at,
            N,
            P,
            K,
            temperature,
            humidity,
            ph,
            wind_speed,
            rainfall_prediction,
            rainfall_prediction AS predicted_rainfall_mm,
            recommended_crop
        FROM predictions
    """

    params = []
    if crop and isinstance(crop, str):
        base_query += " WHERE LOWER(recommended_crop) = LOWER(?)"
        params.append(crop.strip())

    base_query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    cursor.execute(base_query, tuple(params))
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


@app.get("/history/export")
def export_history_csv(crop: Optional[str] = None):
    history_rows = get_history(limit=1000, crop=crop)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "id",
            "created_at",
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "wind_speed",
            "predicted_rainfall_mm",
            "recommended_crop",
        ]
    )

    for row in history_rows:
        writer.writerow(
            [
                row.get("id"),
                row.get("created_at"),
                row.get("N"),
                row.get("P"),
                row.get("K"),
                row.get("temperature"),
                row.get("humidity"),
                row.get("ph"),
                row.get("wind_speed"),
                row.get("predicted_rainfall_mm"),
                row.get("recommended_crop"),
            ]
        )

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=prediction_history.csv"},
    )


@app.delete("/history")
def clear_history():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    deleted_rows = cursor.rowcount
    conn.commit()
    conn.close()
    return {"deleted_rows": deleted_rows}


@app.get("/analytics")
def get_analytics():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT COUNT(*), COALESCE(AVG(rainfall_prediction), 0)
        FROM predictions
        """
    )
    total_predictions, avg_rainfall = cursor.fetchone()

    cursor.execute(
        """
        SELECT recommended_crop, COUNT(*) AS crop_count
        FROM predictions
        GROUP BY recommended_crop
        ORDER BY crop_count DESC
        LIMIT 5
        """
    )
    top_crops = [{"crop": row[0], "count": row[1]} for row in cursor.fetchall()]

    cursor.execute(
        """
        SELECT id, rainfall_prediction, recommended_crop
        FROM predictions
        ORDER BY id DESC
        LIMIT 20
        """
    )
    latest_rows = cursor.fetchall()
    conn.close()

    return {
        "total_predictions": total_predictions,
        "average_rainfall_mm": round(float(avg_rainfall), 2),
        "top_crops": top_crops,
        "latest": [
            {"id": row[0], "predicted_rainfall_mm": row[1], "recommended_crop": row[2]}
            for row in reversed(latest_rows)
        ],
    }
