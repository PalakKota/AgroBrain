import streamlit as st
import pandas as pd
import joblib
import requests
from datetime import datetime
from groq import Groq
import plotly.express as px

# =========================
# CONFIG
# =========================

st.set_page_config(page_title="AgroBrain", layout="centered")

OPENWEATHER_API_KEY = ""
GROQ_API_KEY = ""

client = Groq(api_key=GROQ_API_KEY)

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
# SHC CONVERSION
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

    url = f"https://api.openweathermap.org/data/2.5/forecast?q={location}&appid={OPENWEATHER_API_KEY}&units=metric"

    res = requests.get(url).json()

    if "list" not in res:
        return None

    temps = []
    hums = []
    rain_total = 0

    for entry in res["list"]:

        temps.append(entry["main"]["temp"])
        hums.append(entry["main"]["humidity"])

        rain_total += entry.get("rain", {}).get("3h", 0)

    avg_temp = round(sum(temps) / len(temps), 2)
    avg_hum = round(sum(hums) / len(hums), 2)

    avg_rain = round(rain_total / 5, 2)
    rain_30 = round(avg_rain * 30, 2)

    return avg_temp, avg_hum, avg_rain, rain_30

# =========================
# ML FUNCTIONS
# =========================

def recommend_crop(N, P, K, temp, humidity, ph, rainfall):

    df = pd.DataFrame([{
        "N": N,
        "P": P,
        "K": K,
        "temperature": temp,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    probs = crop_model.predict_proba(df)[0]
    classes = le_crop.classes_

    ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)

    return ranked


def disease_risk(temp, humidity, rainfall, ph):

    df = pd.DataFrame([{
        "temperature": temp,
        "humidity": humidity,
        "rainfall": rainfall,
        "soil_pH": ph
    }])

    pred = disease_model.predict(df)[0]

    return le_disease.inverse_transform([pred])[0]


def irrigation_advice(crop, temp, humidity, rainfall, soil_condition):

    crop_enc = le_irrig_crop.transform([crop])[0]
    soil_enc = le_soil.transform([soil_condition])[0]

    df = pd.DataFrame([{
        "crop": crop_enc,
        "temperature": temp,
        "humidity": humidity,
        "rainfall": rainfall,
        "soil_condition": soil_enc
    }])

    pred = irrig_model.predict(df)[0]

    return le_irrig.inverse_transform([pred])[0]

# =========================
# CROP CALENDAR
# =========================

crop_calendar = {

    "rice":120,
    "maize":100,
    "chickpea":100,
    "kidneybeans":95,
    "pigeonpeas":160,
    "mothbeans":75,
    "mungbean":65,
    "blackgram":75,
    "lentil":105,

    "pomegranate":210,
    "banana":300,
    "mango":365,
    "grapes":240,
    "watermelon":90,
    "muskmelon":85,
    "apple":365,
    "orange":300,
    "papaya":270,
    "coconut":365,

    "cotton":160,
    "jute":130,
    "coffee":365
}

def crop_timeline(crop):

    crop = crop.lower().strip()

    today = datetime.today()

    if crop not in crop_calendar:
        return None

    duration = crop_calendar[crop]

    harvest = today + pd.Timedelta(days=duration)

    return today.date(), harvest.date(), duration

# =========================
# AI EXPLANATION
# =========================

def generate_ai_explanation(top_crop, disease, irrigation):

    prompt = f"""
Explain why {top_crop} is suitable for the farm.

Include:

• soil reasoning  
• irrigation reasoning  
• disease prevention advice  

Keep response simple for farmers.
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role":"user","content":prompt}]
    )

    return res.choices[0].message.content

# =========================
# UI
# =========================

st.title("🌾 AgroBrain")
st.subheader("Explainable AI Smart Farming Advisory System")

# LOCATION

location = st.text_input("Enter City")

avg_temp = avg_hum = avg_rain = rain_30 = None

if location:

    weather = get_weather_from_location(location)

    if weather:

        avg_temp, avg_hum, avg_rain, rain_30 = weather

        st.success(f"Weather loaded for {location}")

        st.write("Temperature:", avg_temp)
        st.write("Humidity:", avg_hum)
        st.write("Rainfall:", avg_rain)

# SOIL INPUT

N_level = st.selectbox("Nitrogen", ["Low","Medium","High"])
P_level = st.selectbox("Phosphorus", ["Low","Medium","High"])
K_level = st.selectbox("Potassium", ["Low","Medium","High"])

ph = st.slider("Soil pH",3.5,9.0,6.5)
soil_condition = st.selectbox("Soil Condition",["Dry","Moist","Wet"])

# =========================
# ANALYZE
# =========================

if st.button("Analyze Farm Conditions"):

    if avg_temp is None:

        st.error("Enter location first.")

    else:

        N = convert_shc_to_numeric(N_level,"N")
        P = convert_shc_to_numeric(P_level,"P")
        K = convert_shc_to_numeric(K_level,"K")

        ranked = recommend_crop(
            N,P,K,
            avg_temp,
            avg_hum,
            ph,
            avg_rain
        )

        st.subheader("Top Crop Recommendations")

        top5 = ranked[:5]

        for crop,prob in top5:
            st.write(crop,f"{prob*100:.2f}%")

        # =========================
        # VISUALIZATION
        # =========================

        chart_df = pd.DataFrame({
            "Crop":[c[0] for c in top5],
            "Suitability":[c[1]*100 for c in top5]
        })

        fig = px.bar(
            chart_df,
            x="Crop",
            y="Suitability",
            color="Suitability",
            text="Suitability",
            title="Crop Suitability Score"
        )

        st.plotly_chart(fig)

        top_crop,top_prob = ranked[0]
        second_crop,second_prob = ranked[1]

        prob_diff = (top_prob-second_prob)*100

        # =========================
        # AGROBRAIN INSIGHTS
        # =========================

        st.subheader("What AgroBrain Discovered")

        if prob_diff < 10:

            st.warning(
                f"{second_crop.capitalize()} is also viable "
                f"(only {prob_diff:.1f}% lower suitability)"
            )

            alt_prompt = f"""
                The recommended crop is {top_crop} ({top_prob*100:.1f}%).
                
                Second crop is {second_crop} ({second_prob*100:.1f}%).
                
                Explain how farmers can successfully grow {second_crop}
                even though it is slightly less suitable.
                
                Include:
                soil adjustments
                irrigation strategy
                fertilizer recommendations
                """

            alt_res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role":"user","content":alt_prompt}]
            )

            st.write("### Another Crop Worth Considering")

            st.write(alt_res.choices[0].message.content)

        else:

            st.success("Current conditions strongly favor the recommended crop.")

        # =========================
        # CROP ADVISORY
        # =========================

        st.subheader("Field Action Plan")

        disease = disease_risk(
            avg_temp,
            avg_hum,
            avg_rain,
            ph
        )

        irrigation = irrigation_advice(
            top_crop,
            avg_temp,
            avg_hum,
            avg_rain,
            soil_condition
        )

        st.write("Disease Risk:",disease)
        st.write("Irrigation Advice:",irrigation)

        # =========================
        # CROP TIMELINE
        # =========================

        st.subheader("Crop Planning Timeline")

        timeline = crop_timeline(top_crop)
        
        if timeline:
        
            plant, harvest, days = timeline
        
            col1, col2, col3 = st.columns(3)
        
            col1.metric("Planting Date", plant.strftime("%d %b %Y"))
            col2.metric("Expected Harvest", harvest.strftime("%d %b %Y"))
            col3.metric("Growth Duration", f"{days} days")
        
        else:
        
            st.info("Timeline data not available for this crop.")

        # =========================
        # AI ADVISORY
        # =========================

        st.subheader("🌾 AgroBrain Knows Best")

        explanation = generate_ai_explanation(
            top_crop,
            disease,
            irrigation
        )

        st.write(explanation)