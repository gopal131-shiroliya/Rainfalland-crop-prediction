import csv
import random
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent / "nawsp"
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
CROP_DATA_PATH = DATA_DIR / "Crop_recommendation.csv"


@st.cache_resource
def load_models():
    return (
        joblib.load(MODEL_DIR / "label_encoder.pkl"),
        joblib.load(MODEL_DIR / "rainfall_model.pkl"),
        joblib.load(MODEL_DIR / "crop_model.pkl"),
    )


@st.cache_data
def load_crop_rows():
    rows = []
    with CROP_DATA_PATH.open("r", newline="", encoding="utf-8") as file_obj:
        for row in csv.DictReader(file_obj):
            rows.append(
                {
                    "N": float(row["N"]), "P": float(row["P"]), "K": float(row["K"]),
                    "temperature": float(row["temperature"]), "humidity": float(row["humidity"]),
                    "ph": float(row["ph"]), "rainfall": float(row["rainfall"]),
                    "label": row["label"],
                }
            )
    return rows


def estimate_rainfall_from_dataset(rows, temperature, humidity, k=7):
    ranked = sorted(
        rows,
        key=lambda row: (row["temperature"] - temperature) ** 2
        + (row["humidity"] - humidity) ** 2,
    )
    neighbours = ranked[: max(1, min(k, len(ranked)))]
    rainfall = sum(row["rainfall"] for row in neighbours) / len(neighbours)
    return round(float(rainfall), 2), neighbours


def get_crop_recommendations(label_encoder, crop_model, crop_input, top_n=3):
    primary_encoded = int(crop_model.predict(crop_input)[0])
    primary_crop = str(label_encoder.inverse_transform([primary_encoded])[0])
    recommendations = [{"crop": primary_crop, "confidence": 1.0}]

    if hasattr(crop_model, "predict_proba") and hasattr(crop_model, "classes_"):
        probabilities = crop_model.predict_proba(crop_input)[0]
        recommendations = []
        for idx in np.argsort(probabilities)[::-1][:top_n]:
            crop = str(label_encoder.inverse_transform([int(crop_model.classes_[idx])])[0])
            recommendations.append({"crop": crop, "confidence": float(probabilities[idx])})
    return primary_crop, recommendations


def choose_sample(rows):
    sample = random.choice(rows)
    st.session_state.inputs = {
        "N": sample["N"], "P": sample["P"], "K": sample["K"],
        "temperature": sample["temperature"], "humidity": sample["humidity"],
        "ph": sample["ph"], "wind_speed": 10.0,
    }
    st.session_state.sample_label = sample["label"].title()


st.set_page_config(page_title="AgriSight | Crop Intelligence", page_icon="🌱", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background: #f4f8f2; color: #18352a; }
      .block-container { max-width: 1180px; padding-top: 2.1rem; padding-bottom: 3rem; }
      #MainMenu, footer, header { visibility: hidden; }
      .hero { padding: 2rem 2.2rem; border-radius: 24px; color: white;
        background: linear-gradient(120deg, #104b38, #1f8a5b 58%, #55a66d);
        box-shadow: 0 16px 35px rgba(20, 82, 55, .22); margin-bottom: 1.35rem; }
      .hero h1 { margin: 0; font-size: clamp(2rem, 4vw, 3.25rem); letter-spacing: -.04em; }
      .hero p { margin: .55rem 0 0; opacity: .9; font-size: 1.05rem; }
      .tag { display: inline-block; border: 1px solid rgba(255,255,255,.35); border-radius: 20px;
        padding: .27rem .75rem; margin-bottom: .8rem; font-size: .82rem; font-weight: 600; }
      .section-title { margin: .35rem 0 .15rem; font-size: 1.2rem; font-weight: 700; color: #163c2d; }
      .section-note { color: #638071; margin-bottom: .8rem; font-size: .92rem; }
      [data-testid="stForm"] { border: 1px solid #dbe7dc; border-radius: 20px; padding: 1.2rem;
        background: white; box-shadow: 0 8px 24px rgba(35, 76, 55, .06); }
      [data-testid="stMetric"] { background: white; border: 1px solid #dbe7dc; border-radius: 16px;
        padding: .95rem 1rem; box-shadow: 0 6px 16px rgba(35, 76, 55, .05); }
      [data-testid="stMetricLabel"] { color: #638071; }
      .result-card { background: #123f31; color: white; border-radius: 20px; padding: 1.5rem;
        min-height: 245px; box-shadow: 0 14px 28px rgba(18,63,49,.18); }
      .result-card h2 { margin: 0 0 .3rem; font-size: 1.05rem; font-weight: 500; opacity: .82; }
      .result-card .crop { font-size: 2.55rem; font-weight: 750; margin: .1rem 0 .6rem; }
      .result-card .rain { color: #b8f0c8; font-size: 1.12rem; }
      .result-card .small { opacity: .77; margin-top: 1rem; font-size: .86rem; }
      .rec { display:flex; justify-content:space-between; gap:1rem; padding:.62rem 0;
        border-bottom:1px solid #dfe9e2; color:#27483a; }
      .rec:last-child { border-bottom:0; }
      .source { background:#e8f4e9; border:1px solid #cde4d2; border-radius:14px; padding:.85rem 1rem;
        color:#27543b; font-size:.91rem; }
      .stButton > button, [data-testid="stFormSubmitButton"] > button { border: 0; border-radius: 10px;
        background: #16754d; color: white; font-weight: 650; }
      .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover { background:#0f5f3e; color:white; }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    label_encoder, rainfall_model, crop_model = load_models()
    crop_rows = load_crop_rows()
except Exception as exc:
    st.error(f"App setup failed: {exc}")
    st.stop()

if "inputs" not in st.session_state:
    choose_sample(crop_rows)
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown(
    """<section class="hero"><div class="tag">ML-POWERED AGRICULTURE DECISION SUPPORT</div>
    <h1>AgriSight Intelligence</h1><p>Turn soil and weather conditions into confident crop recommendations.</p></section>""",
    unsafe_allow_html=True,
)

top_a, top_b, top_c, top_d = st.columns(4)
top_a.metric("Training records", f"{len(crop_rows):,}")
top_b.metric("Soil nutrients", "N · P · K")
top_c.metric("Crop classes", len(label_encoder.classes_))
top_d.metric("Prediction engine", "Ready", delta="Online")

st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
left, right = st.columns([1.28, 0.72], gap="large")

with left:
    title_col, sample_col = st.columns([3, 1])
    with title_col:
        st.markdown("<div class='section-title'>Field conditions</div><div class='section-note'>Enter soil nutrients and local weather observations.</div>", unsafe_allow_html=True)
    with sample_col:
        if st.button("↻ Dataset sample", use_container_width=True):
            choose_sample(crop_rows)
            st.rerun()

    values = st.session_state.inputs
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            n_value = st.number_input("Nitrogen (N)", min_value=0.0, value=float(values["N"]), step=1.0)
            temperature = st.number_input("Temperature (°C)", min_value=-20.0, max_value=70.0, value=float(values["temperature"]), step=0.1)
        with col2:
            p_value = st.number_input("Phosphorus (P)", min_value=0.0, value=float(values["P"]), step=1.0)
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=float(values["humidity"]), step=0.1)
        with col3:
            k_value = st.number_input("Potassium (K)", min_value=0.0, value=float(values["K"]), step=1.0)
            ph_value = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=float(values["ph"]), step=0.1)
        wind_speed = st.number_input("Wind speed (km/h)", min_value=0.0, value=float(values["wind_speed"]), step=0.5)
        submitted = st.form_submit_button("Generate recommendation  →", use_container_width=True)

    if submitted:
        model_rainfall = round(max(0.0, float(rainfall_model.predict(np.array([[temperature, humidity, wind_speed]]))[0])), 2)
        estimated_rainfall, neighbours = estimate_rainfall_from_dataset(crop_rows, temperature, humidity)
        crop_input = np.array([[n_value, p_value, k_value, temperature, humidity, ph_value, estimated_rainfall]])
        crop, recommendations = get_crop_recommendations(label_encoder, crop_model, crop_input)
        st.session_state.prediction = {
            "crop": crop, "recommendations": recommendations, "estimated_rainfall": estimated_rainfall,
            "model_rainfall": model_rainfall, "neighbours": neighbours,
        }
        st.session_state.history.append(
            {
                "ID": len(st.session_state.history) + 1,
                "Created at": datetime.now().strftime("%d %b, %H:%M"),
                "N": round(n_value, 2), "P": round(p_value, 2), "K": round(k_value, 2),
                "Temperature": round(temperature, 2), "Humidity": round(humidity, 2),
                "pH": round(ph_value, 2), "Wind": round(wind_speed, 2),
                "Rainfall": estimated_rainfall, "Crop": crop.title(),
            }
        )

with right:
    prediction = st.session_state.prediction
    if prediction:
        st.markdown(
            f"""<div class="result-card"><h2>Recommended crop</h2><div class="crop">{prediction['crop'].title()}</div>
            <div class="rain">Estimated rainfall · {prediction['estimated_rainfall']:.2f} mm</div>
            <div class="small">Raw rainfall-model output: {prediction['model_rainfall']:.2f} mm<br>Based on soil, weather and nearest dataset conditions.</div></div>""",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='section-title' style='margin-top:1.1rem'>Top matches</div>", unsafe_allow_html=True)
        for item in prediction["recommendations"]:
            st.markdown(f"<div class='rec'><span>{item['crop'].title()}</span><strong>{item['confidence'] * 100:.1f}%</strong></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-card'><h2>Your result will appear here</h2><div class='crop' style='font-size:1.65rem'>Ready to predict</div><div class='small'>Use a real dataset sample or enter field values, then generate a recommendation.</div></div>", unsafe_allow_html=True)

st.markdown("<div style='height:1.4rem'></div><div class='section-title'>Live analytics</div><div class='section-note'>Every saved prediction updates these dashboard insights.</div>", unsafe_allow_html=True)
history_df = pd.DataFrame(st.session_state.history)
kpi1, kpi2, kpi3 = st.columns(3)
if history_df.empty:
    kpi1.metric("Saved predictions", "0")
    kpi2.metric("Average rainfall", "--")
    kpi3.metric("Top crop", "--")
    st.info("Generate and save a prediction to activate analytics and charts.")
else:
    top_crop = history_df["Crop"].mode().iat[0]
    kpi1.metric("Saved predictions", len(history_df))
    kpi2.metric("Average rainfall", f"{history_df['Rainfall'].mean():.2f} mm")
    kpi3.metric("Top crop", top_crop)
    chart_a, chart_b = st.columns(2, gap="large")
    with chart_a:
        st.caption("Rainfall trend by saved prediction")
        st.line_chart(history_df.set_index("ID")[["Rainfall"]], color="#1f7a4d", height=250)
    with chart_b:
        st.caption("Most recommended crops")
        crop_counts = history_df["Crop"].value_counts().rename("Predictions")
        st.bar_chart(crop_counts, color="#167a9f", height=250)

st.markdown("<div style='height:1.2rem'></div><div class='section-title'>Prediction history</div><div class='section-note'>Filter, export, or clear the predictions created in this session.</div>", unsafe_allow_html=True)
history_controls = st.columns([1, 1.3, 1, 1])
with history_controls[0]:
    limit = st.selectbox("Show rows", [10, 25, 50, 100], index=1)
with history_controls[1]:
    crop_filter = st.text_input("Filter crop", placeholder="e.g. Rice")
with history_controls[2]:
    if not history_df.empty:
        st.download_button("Download CSV", history_df.to_csv(index=False).encode("utf-8"), "prediction_history.csv", "text/csv", use_container_width=True)
with history_controls[3]:
    if st.button("Clear history", use_container_width=True, disabled=history_df.empty):
        st.session_state.history = []
        st.session_state.prediction = None
        st.rerun()

if history_df.empty:
    st.markdown("<div class='source'>No saved predictions yet. Click <b>Generate recommendation</b> to create your first history entry.</div>", unsafe_allow_html=True)
else:
    filtered_history = history_df
    if crop_filter.strip():
        filtered_history = history_df[history_df["Crop"].str.contains(crop_filter.strip(), case=False, na=False)]
    st.dataframe(filtered_history.tail(limit).iloc[::-1], use_container_width=True, hide_index=True)

st.markdown("<div style='height:1.15rem'></div>", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["Dataset evidence", "How the model works"])
with tab1:
    st.markdown("<div class='source'>The app reads the included Crop Recommendation dataset at runtime. The sample button fills the form with an actual row from that dataset.</div>", unsafe_allow_html=True)
    preview = pd.DataFrame(crop_rows)[["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]].sample(8, random_state=7)
    st.dataframe(preview, use_container_width=True, hide_index=True)
with tab2:
    st.markdown("""<div class='source'><b>1.</b> A rainfall model evaluates temperature, humidity and wind speed.<br>
    <b>2.</b> The 7 nearest climate records in the real dataset provide a crop-compatible rainfall estimate.<br>
    <b>3.</b> The crop classifier ranks the most suitable crops using N, P, K, temperature, humidity, pH and rainfall.</div>""", unsafe_allow_html=True)
