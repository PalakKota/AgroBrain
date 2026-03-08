import streamlit as st
import pandas as pd
import joblib
import requests
from datetime import datetime
import random
from groq import Groq

API_KEY = st.secrets["OPENWEATHER_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
# =========================
# CONFIG
# =========================

st.set_page_config(page_title="AgroBrain", layout="centered")



# =========================
# LOAD MODELS
# =========================

crop_model = joblib.load("models/crop_model.pkl")
le_crop = joblib.load("models/crop_label_encoder.pkl")

disease_model = joblib.load("models/disease_model.pkl")
le_disease = joblib.load("models/disease_label_encoder.pkl")

irrig_model = joblib.load("models/irrigation_model.pkl")
le_irrig_crop = joblib.load("models/irrig_crop_encoder.pkl")
le_soil = joblib.load("models/soil_encoder.pkl")
le_irrig = joblib.load("models/irrig_label_encoder.pkl")

# =========================
# SHC CATEGORY → NUMERIC
# =========================

def convert_shc_to_numeric(level, nutrient):

    fixed_values = {
        "N": {"Low": 30, "Medium": 65, "High": 110},
        "P": {"Low": 20, "Medium": 50, "High": 100},
        "K": {"Low": 40, "Medium": 80, "High": 150}
    }

    return fixed_values[nutrient][level]

# =========================
# WEATHER FUNCTION
# =========================

def get_weather_from_location(location):

    forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?q={location}&appid={API_KEY}&units=metric"
    forecast_res = requests.get(forecast_url).json()

    if "list" not in forecast_res:
        return None

    temp_values = []
    humidity_values = []
    five_day_rain_total = 0

    for entry in forecast_res["list"]:

        temp_values.append(entry["main"]["temp"])
        humidity_values.append(entry["main"]["humidity"])

        rain = entry.get("rain", {}).get("3h", 0)
        five_day_rain_total += rain

    # 5-day averages
    avg_temp = round(sum(temp_values) / len(temp_values), 2)
    avg_humidity = round(sum(humidity_values) / len(humidity_values), 2)

    avg_daily_rain_5day = round(five_day_rain_total / 5, 2)

    # 🔥 Estimated 30-day rainfall projection
    estimated_30_day_rain = round(avg_daily_rain_5day * 30, 2)

    return {
        "avg_temperature": avg_temp,
        "avg_humidity": avg_humidity,
        "avg_rainfall_5day": avg_daily_rain_5day,
        "estimated_30_day_rainfall": estimated_30_day_rain
    }

# =========================
# ML FUNCTIONS
# =========================

def recommend_crop(N, P, K, temp, humidity, ph, rainfall):

    data = pd.DataFrame([{
        "N": N,
        "P": P,
        "K": K,
        "temperature": temp,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    probs = crop_model.predict_proba(data)[0]
    classes = le_crop.classes_

    ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)

    return ranked


def disease_risk(temp, humidity, rainfall, ph):

    data = pd.DataFrame([{
        "temperature": temp,
        "humidity": humidity,
        "rainfall": rainfall,
        "soil_pH": ph
    }])

    pred = disease_model.predict(data)[0]
    return le_disease.inverse_transform([pred])[0]


def irrigation_advice(crop, temp, humidity, rainfall, soil_condition):

    crop_enc = le_irrig_crop.transform([crop])[0]
    soil_enc = le_soil.transform([soil_condition])[0]

    data = pd.DataFrame([{
        "crop": crop_enc,
        "temperature": temp,
        "humidity": humidity,
        "rainfall": rainfall,
        "soil_condition": soil_enc
    }])

    pred = irrig_model.predict(data)[0]
    return le_irrig.inverse_transform([pred])[0]



# =========================
# 🔥 AI EXPLANATION (GROQ VERSION)
# =========================

client = Groq(api_key=GROQ_API_KEY)
@st.cache_data(show_spinner=False)
def generate_ai_explanation(top_crop, disease, irrigation, inputs, monthly_rainfall):

    prompt = f"""
    You are an expert agricultural advisor.

    Explain clearly:

    1. Why {top_crop} is suitable.
    2. Disease prevention measures for risk level: {disease}.
    3. Irrigation advice: {irrigation}.
    4. Rainfall caution considering 30-day rainfall of {monthly_rainfall} mm.

    Farm Data:
    Nitrogen: {inputs['N']}
    Phosphorus: {inputs['P']}
    Potassium: {inputs['K']}
    Soil pH: {inputs['ph']}
    Soil Condition: {inputs['soil_condition']}
    Average Temperature: {inputs['temperature']} °C
    Average Humidity: {inputs['humidity']} %
    Average Daily Rainfall (5-day): {inputs['rainfall']} mm

    Keep explanation farmer-friendly and 6-8 sentences with emojis.
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # 🔥 You can change to 8b if needed
            messages=[
                {"role": "system", "content": "You are a professional agricultural AI advisor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI service error: {str(e)}"

# =========================
# UI
# =========================

st.title("🌾 AgroBrain ")
st.subheader("AI-Enhanced Smart Farming Advisory System")

st.markdown("---")

# LOCATION
st.header("🌍 Location-Based Weather Intelligence")

location = st.text_input("Enter City Name")

avg_temperature = None
avg_humidity = None
avg_rainfall = None
estimated_30_day_rain = None

if location:

    weather = get_weather_from_location(location)

    if weather:

        avg_temperature = weather["avg_temperature"]
        avg_humidity = weather["avg_humidity"]
        avg_rainfall = weather["avg_rainfall_5day"]
        estimated_30_day_rain = weather["estimated_30_day_rainfall"]

        st.success(f"Weather Forecast Analyzed for {location}")

        st.write(f"🌡 5-Day Avg Temperature: {avg_temperature} °C")
        st.write(f"💧 5-Day Avg Humidity: {avg_humidity} %")
        st.write(f"🌧 Avg Rainfall per Day (5-Day): {avg_rainfall} mm")
        st.write(f"📅 Estimated 30-Day Rainfall Projection: {estimated_30_day_rain} mm")

    else:
        st.error("Unable to fetch weather data. Check location or API key.")

st.markdown("---")

# SOIL INPUT (SHC STYLE)
st.header("🌱 Soil Health Card Inputs")

N_level = st.selectbox("Nitrogen Level (SHC)", ["Low", "Medium", "High"])
P_level = st.selectbox("Phosphorus Level (SHC)", ["Low", "Medium", "High"])
K_level = st.selectbox("Potassium Level (SHC)", ["Low", "Medium", "High"])

ph = st.slider("Soil pH", 3.5, 9.0, 6.5)
soil_condition = st.selectbox("Soil Condition", ["Dry", "Moist", "Wet"])

st.markdown("---")

# ANALYZE
if st.button("Analyze Farm Conditions"):

    if avg_temperature is None:
        st.error("Please enter a valid location first.")
    else:

        # Convert SHC categories to numeric
        N = convert_shc_to_numeric(N_level, "N")
        P = convert_shc_to_numeric(P_level, "P")
        K = convert_shc_to_numeric(K_level, "K")

        inputs = {
            "N": N,
            "P": P,
            "K": K,
            "temperature": avg_temperature,
            "humidity": avg_humidity,
            "ph": ph,
            "rainfall": avg_rainfall,
            "soil_condition": soil_condition
        }

        ranked_crops = recommend_crop(
            N, P, K,
            avg_temperature,
            avg_humidity,
            ph,
            avg_rainfall
        )

        top_crop = ranked_crops[0][0]
        second_crop = ranked_crops[1][0]

        st.subheader("🌱 Top 5 Crop Recommendations")

        top_5_crops = ranked_crops[:5]

        for i, (crop, prob) in enumerate(top_5_crops, 1):
            st.write(f"{i}. **{crop.capitalize()}** — {prob*100:.2f}% suitability")

        st.markdown("---")

        st.subheader("📋 Advisory for Top 2 Crops")

        for crop_name in [top_crop, second_crop]:

            disease = disease_risk(
                avg_temperature,
                avg_humidity,
                avg_rainfall,
                ph
            )

            irrigation = irrigation_advice(
                crop_name,
                avg_temperature,
                avg_humidity,
                avg_rainfall,
                soil_condition
            )

            st.markdown(f"### 🌾 {crop_name.capitalize()}")
            st.write(f"🦠 Disease Risk: **{disease}**")
            st.write(f"💧 Irrigation Advice: **{irrigation}**")
            st.markdown("---")

        st.subheader("🤖 AI Advisory")
        explanation = generate_ai_explanation(
                top_crop,
                disease,
                irrigation,
                inputs,
                estimated_30_day_rain
            )

        st.write(explanation)

