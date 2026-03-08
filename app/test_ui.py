import streamlit as st
import pandas as pd
import joblib
import requests
from groq import Groq

API_KEY = st.secrets["OPENWEATHER_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

client = Groq(api_key=GROQ_API_KEY)

# PAGE CONFIG
st.set_page_config(page_title="AgroBrain", layout="wide", page_icon="🌾")

# -----------------------------
# CUSTOM CSS
# -----------------------------

st.markdown("""
<style>

.main {background-color:#f5f7fa;}

.section-box{
background:white;
padding:20px;
border-radius:12px;
box-shadow:0px 3px 10px rgba(0,0,0,0.1);
margin-bottom:20px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODELS
# -----------------------------

crop_model = joblib.load("models/crop_model.pkl")
le_crop = joblib.load("models/crop_label_encoder.pkl")

disease_model = joblib.load("models/disease_model.pkl")
le_disease = joblib.load("models/disease_label_encoder.pkl")

irrig_model = joblib.load("models/irrigation_model.pkl")
le_irrig_crop = joblib.load("models/irrig_crop_encoder.pkl")
le_soil = joblib.load("models/soil_encoder.pkl")
le_irrig = joblib.load("models/irrig_label_encoder.pkl")

# -----------------------------
# SHC CONVERSION
# -----------------------------

def convert_shc_to_numeric(level, nutrient):

    fixed_values = {
        "N": {"Low": 30, "Medium": 65, "High": 110},
        "P": {"Low": 20, "Medium": 50, "High": 100},
        "K": {"Low": 40, "Medium": 80, "High": 150}
    }

    return fixed_values[nutrient][level]

# -----------------------------
# WEATHER
# -----------------------------

def get_weather_from_location(location):

    url=f"https://api.openweathermap.org/data/2.5/forecast?q={location}&appid={API_KEY}&units=metric"

    data=requests.get(url).json()

    if "list" not in data:
        return None

    temp=[]
    humidity=[]
    rain_total=0

    for entry in data["list"]:
        temp.append(entry["main"]["temp"])
        humidity.append(entry["main"]["humidity"])
        rain_total+=entry.get("rain",{}).get("3h",0)

    avg_temp=round(sum(temp)/len(temp),2)
    avg_humidity=round(sum(humidity)/len(humidity),2)

    rain_5day=round(rain_total/5,2)

    monthly_rain=round(rain_5day*30,2)

    return avg_temp,avg_humidity,rain_5day,monthly_rain

# -----------------------------
# ML FUNCTIONS
# -----------------------------

def recommend_crop(N,P,K,temp,humidity,ph,rainfall):

    data=pd.DataFrame([{
        "N":N,
        "P":P,
        "K":K,
        "temperature":temp,
        "humidity":humidity,
        "ph":ph,
        "rainfall":rainfall
    }])

    probs=crop_model.predict_proba(data)[0]
    classes=le_crop.classes_

    ranked=sorted(zip(classes,probs),key=lambda x:x[1],reverse=True)

    return ranked

def disease_risk(temp,humidity,rainfall,ph):

    data=pd.DataFrame([{
        "temperature":temp,
        "humidity":humidity,
        "rainfall":rainfall,
        "soil_pH":ph
    }])

    pred=disease_model.predict(data)[0]

    return le_disease.inverse_transform([pred])[0]

def irrigation_advice(crop,temp,humidity,rainfall,soil_condition):

    crop_enc=le_irrig_crop.transform([crop])[0]
    soil_enc=le_soil.transform([soil_condition])[0]

    data=pd.DataFrame([{
        "crop":crop_enc,
        "temperature":temp,
        "humidity":humidity,
        "rainfall":rainfall,
        "soil_condition":soil_enc
    }])

    pred=irrig_model.predict(data)[0]

    return le_irrig.inverse_transform([pred])[0]

# -----------------------------
# AI EXPLANATION
# -----------------------------

def generate_ai_explanation(top_crop,disease,irrigation,inputs,monthly_rain):

    prompt=f"""
Explain clearly for a farmer:

1. Why {top_crop} is suitable
2. Disease prevention for {disease}
3. Irrigation advice: {irrigation}
4. Rainfall caution considering {monthly_rain} mm rain

Farm Data:
Nitrogen {inputs['N']}
Phosphorus {inputs['P']}
Potassium {inputs['K']}
Temperature {inputs['temperature']}
Humidity {inputs['humidity']}
pH {inputs['ph']}
Rainfall {inputs['rainfall']}

Write 6 simple sentences.
"""

    try:

        response=client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role":"user","content":prompt}]
        )

        return response.choices[0].message.content

    except Exception as e:

        return str(e)

# -----------------------------
# HEADER
# -----------------------------

st.title("🌾 AgroBrain")
st.subheader("AI Precision Farming Advisory System")

# -----------------------------
# ABOUT
# -----------------------------

st.header("About the System")

st.write("""
AgroBrain is an AI-powered precision agriculture advisory system that helps farmers make smarter crop decisions using machine learning.

The platform integrates:

• Soil Health Card nutrient analysis  
• Weather intelligence  
• Crop suitability prediction  
• Disease risk estimation  
• Irrigation scheduling  

Using these insights, AgroBrain recommends the most suitable crops and provides intelligent farming advice to improve yield and sustainability.
""")

# -----------------------------
# HOW TO USE
# -----------------------------

st.header("How to Use AgroBrain")

st.write("""
1️⃣ Enter your **farm location** to fetch weather data  
2️⃣ Select **Soil Health Card values (N,P,K)**  
3️⃣ Set soil pH and soil condition  
4️⃣ Click **Analyze Farm Conditions**  
5️⃣ View crop recommendations, disease risks, irrigation advice and AI insights
""")

st.markdown("---")

# -----------------------------
# WEATHER INPUT
# -----------------------------

st.header("Weather Intelligence")

location=st.text_input("Enter City")

avg_temperature=None

if location:

    weather=get_weather_from_location(location)

    if weather:

        avg_temperature,avg_humidity,avg_rainfall,estimated_30_day_rain=weather

        col1,col2,col3,col4=st.columns(4)

        col1.metric("Temperature",f"{avg_temperature} °C")
        col2.metric("Humidity",f"{avg_humidity}%")
        col3.metric("Rainfall",f"{avg_rainfall} mm/day")
        col4.metric("30 Day Rain",f"{estimated_30_day_rain} mm")

# -----------------------------
# SOIL INPUT
# -----------------------------

st.header("Soil Health Card")

col1,col2,col3=st.columns(3)

with col1:
    N_level=st.selectbox("Nitrogen",["Low","Medium","High"])

with col2:
    P_level=st.selectbox("Phosphorus",["Low","Medium","High"])

with col3:
    K_level=st.selectbox("Potassium",["Low","Medium","High"])

ph=st.slider("Soil pH",3.5,9.0,6.5)

soil_condition=st.selectbox("Soil Condition",["Dry","Moist","Wet"])

st.markdown("---")

# -----------------------------
# ANALYZE
# -----------------------------

if st.button("Analyze Farm Conditions"):

    if avg_temperature is None:

        st.error("Enter location first")

    else:

        N=convert_shc_to_numeric(N_level,"N")
        P=convert_shc_to_numeric(P_level,"P")
        K=convert_shc_to_numeric(K_level,"K")

        ranked_crops=recommend_crop(
            N,P,K,
            avg_temperature,
            avg_humidity,
            ph,
            avg_rainfall
        )

        st.header("Top Crop Recommendations")

        top5=ranked_crops[:5]

        for crop,prob in top5:

            percent=prob*100

            st.write(f"**{crop.capitalize()} — {percent:.2f}% suitability**")

            st.progress(prob)

        top_crop=ranked_crops[0][0]
        second_crop=ranked_crops[1][0]

        st.markdown("---")

        st.header("Advisory for Top Crops")

        for crop_name in [top_crop,second_crop]:

            disease=disease_risk(
                avg_temperature,
                avg_humidity,
                avg_rainfall,
                ph
            )

            irrigation=irrigation_advice(
                crop_name,
                avg_temperature,
                avg_humidity,
                avg_rainfall,
                soil_condition
            )

            st.subheader(crop_name.capitalize())

            st.write(f"Disease Risk: **{disease}**")

            st.write(f"Irrigation Advice: **{irrigation}**")

        st.markdown("---")

        inputs={
        "N":N,
        "P":P,
        "K":K,
        "temperature":avg_temperature,
        "humidity":avg_humidity,
        "ph":ph,
        "rainfall":avg_rainfall
        }

        st.header("AI Advisory")

        explanation=generate_ai_explanation(
            top_crop,
            disease,
            irrigation,
            inputs,
            estimated_30_day_rain
        )

        st.write(explanation)