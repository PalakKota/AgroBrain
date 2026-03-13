import streamlit as st
import pandas as pd
import joblib
import requests
from datetime import datetime
from groq import Groq
import plotly.express as px
import plotly.graph_objects as go

# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="AgroBrain",
    layout="wide",
    page_icon="🌾",
    initial_sidebar_state="collapsed"
)

API_KEY = st.secrets["OPENWEATHER_API_KEY"]

# =========================
# SESSION STATE
# =========================

if "chat_history"       not in st.session_state: st.session_state.chat_history       = []
if "analysis_done"      not in st.session_state: st.session_state.analysis_done      = False
if "farm_context"       not in st.session_state: st.session_state.farm_context       = {}
if "last_results"       not in st.session_state: st.session_state.last_results       = None
if "last_inputs_key"    not in st.session_state: st.session_state.last_inputs_key    = None

# =========================
# TERRACOTTA DUSK THEME — single fixed palette
# =========================

T = {
    # Backgrounds — deep warm earth
    "bg_deep":       "#5c2509",
    "bg_card":       "#1c1009",
    "bg_card2":      "#231409",
    "bg_card3":      "#2a1a0e",
    "border":        "#3d2010",
    "border_bright": "#6b3820",

    # Accent family — fire / ember / terracotta
    "accent":        "#ff7c3a",   # primary terracotta-orange
    "accent2":       "#ffad6a",   # soft amber-orange
    "accent3":       "#ffd166",   # warm yellow-gold
    "accent_red":    "#ff4d4d",   # error / high risk
    "accent_blue":   "#5ec4ff",   # info / medium
    "accent_green":  "#7ecb8a",   # success / low risk

    # Text
    "text_primary":  "#f5e6d8",
    "text_muted":    "#8a6050",
    "text_dim":      "#4a3020",

    # Glow & gradients
    "glow_rgba":     "rgba(255,124,58,0.12)",
    "glow_rgba2":    "rgba(255,173,106,0.06)",
    "hero_grad":     "linear-gradient(135deg, #1e0e06 0%, #2a1206 40%, #160b04 100%)",

    # Plotly
    "bar_grad":      [[0, "#3d2010"], [0.5, "#c45020"], [1.0, "#ff7c3a"]],
    "gantt_fill":    "rgba(255,124,58,0.20)",
    "gantt_line":    "#ff7c3a",
    "plot_grid":     "#3d2010",
    "plot_text":     "#8a6050",
    "plot_title":    "#f5e6d8",

    # Gauge step colours (proper rgba for Plotly)
    "gauge_poor":    "rgba(255,77,77,0.18)",
    "gauge_fair":    "rgba(94,196,255,0.15)",
    "gauge_good":    "rgba(255,173,106,0.15)",
    "gauge_great":   "rgba(255,124,58,0.18)",

    "stapp_bg":      "#120a06",
}

# =========================
# INJECT CSS — Terracotta Dusk
# =========================

def inject_css():
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {{
    --bg-deep:      {T['bg_deep']};
    --bg-card:      {T['bg_card']};
    --bg-card2:     {T['bg_card2']};
    --bg-card3:     {T['bg_card3']};
    --border:       {T['border']};
    --border-bright:{T['border_bright']};
    --accent:       {T['accent']};
    --accent2:      {T['accent2']};
    --accent3:      {T['accent3']};
    --accent-red:   {T['accent_red']};
    --accent-blue:  {T['accent_blue']};
    --accent-green: {T['accent_green']};
    --text-primary: {T['text_primary']};
    --text-muted:   {T['text_muted']};
    --text-dim:     {T['text_dim']};
    --glow:         {T['glow_rgba']};
    --glow2:        {T['glow_rgba2']};
}}

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary) !important;
}}
.stApp {{
    background-color: {T['stapp_bg']} !important;
    background-image: radial-gradient(ellipse at top right, rgba(255,124,58,0.04) 0%, transparent 60%),
                      radial-gradient(ellipse at bottom left, rgba(255,77,77,0.03) 0%, transparent 60%);
}}
.block-container {{
    padding: 2rem 3rem !important;
    max-width: 1400px !important;
    background: transparent !important;
}}

/* ---- HERO ---- */
.hero-header {{
    background: {T['hero_grad']};
    border: 1px solid var(--border-bright);
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}}
.hero-header::before {{
    content: '';
    position: absolute; top: -80px; right: -80px;
    width: 380px; height: 380px;
    background: radial-gradient(circle, {T['glow_rgba']} 0%, transparent 65%);
    border-radius: 50%;
}}
.hero-header::after {{
    content: '';
    position: absolute; bottom: -60px; left: 25%;
    width: 300px; height: 220px;
    background: radial-gradient(ellipse, {T['glow_rgba2']} 0%, transparent 65%);
}}
.hero-title {{
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem; font-weight: 800; letter-spacing: -1px;
    background: linear-gradient(90deg, {T['accent']} 0%, {T['accent2']} 55%, {T['accent3']} 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin: 0 0 0.5rem 0; line-height: 1.1;
}}
.hero-subtitle {{
    font-size: 1.1rem; color: var(--text-muted); font-weight: 300;
    letter-spacing: 0.02em; margin: 0;
}}
.hero-badge {{
    display: inline-block;
    background: {T['glow_rgba']};
    border: 1px solid var(--accent);
    color: var(--accent);
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; padding: 0.3rem 0.85rem;
    border-radius: 100px; margin-bottom: 1.2rem; opacity: 0.9;
}}

/* ---- SECTION LABELS ---- */
.section-label {{
    font-family: 'Syne', sans-serif; font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--accent); margin-bottom: 0.4rem;
}}
.section-title {{
    font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 700;
    color: var(--text-primary); margin: 0 0 1.5rem 0; line-height: 1.2;
}}

/* ---- STAT TILES ---- */
.stat-tile {{
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.4rem 1.6rem; text-align: center;
    position: relative; overflow: hidden;
    transition: border-color 0.2s;
}}
.stat-tile:hover {{ border-color: var(--border-bright); }}
.stat-tile-glow {{
    position: absolute; top: -20px; left: 50%; transform: translateX(-50%);
    width: 80px; height: 60px;
    background: radial-gradient(circle, {T['glow_rgba']} 0%, transparent 70%);
}}
.stat-tile-icon {{ font-size: 1.6rem; margin-bottom: 0.5rem; opacity: 1 !important; }}
.stat-tile-value {{
    font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 700;
    color: var(--accent); line-height: 1; margin-bottom: 0.3rem;
}}
.stat-tile-label {{ font-size: 0.78rem; color: var(--text-muted); font-weight: 400; letter-spacing: 0.04em; }}

/* ---- CROP RANK ---- */
.crop-rank-item {{
    display: flex; align-items: center; background: var(--bg-card2);
    border: 1px solid var(--border); border-radius: 12px;
    padding: 1rem 1.4rem; margin-bottom: 0.7rem; gap: 1rem;
    position: relative; overflow: hidden; transition: all 0.2s;
}}
.crop-rank-item:hover {{ border-color: var(--accent); transform: translateX(4px); }}
.crop-rank-item.top {{ border-color: var(--accent); background: {T['glow_rgba']}; }}
.crop-rank-badge {{ font-family: 'Syne', sans-serif; font-size: 0.85rem; font-weight: 700; color: var(--text-dim); min-width: 28px; }}
.crop-rank-badge.top {{ color: var(--accent); }}
.crop-rank-name {{ font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 600; color: var(--text-primary); flex: 1; text-transform: capitalize; }}
.crop-rank-pct {{ font-family: 'Syne', sans-serif; font-size: 0.9rem; font-weight: 700; color: var(--accent2); }}
.crop-rank-bar-bg {{ position: absolute; bottom: 0; left: 0; height: 2px; width: 100%; background: var(--border); }}
.crop-rank-bar-fill {{ height: 2px; background: linear-gradient(90deg, var(--accent), var(--accent2)); }}

/* ---- ADVISORY CARD ---- */
.advisory-card {{
    background: var(--bg-card2); border: 1px solid var(--border);
    border-radius: 16px; padding: 1.8rem; margin-bottom: 1rem;
}}
.advisory-card.primary {{ border-color: var(--accent); }}
.advisory-card-head {{
    font-family: 'Syne', sans-serif; font-size: 1.25rem; font-weight: 700;
    color: var(--accent); margin-bottom: 1.2rem; text-transform: capitalize;
}}
.advisory-pill {{
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: {T['glow_rgba']}; border: 1px solid var(--border);
    border-radius: 100px; padding: 0.5rem 1rem; margin: 0.3rem 0.3rem 0 0;
    font-size: 0.88rem; color: var(--text-primary);
}}
.advisory-pill .pill-label {{ color: var(--text-muted); font-size: 0.78rem; }}
.advisory-pill .pill-value {{ font-weight: 600; }}

/* ---- TIMELINE ---- */
.timeline-card {{
    background: var(--bg-card); border: 1px solid var(--accent);
    border-radius: 16px; padding: 2rem; text-align: center; opacity: 0.9;
}}
.timeline-card h4 {{
    font-family: 'Syne', sans-serif; font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.15em; text-transform: uppercase; color: var(--text-muted); margin: 0 0 0.4rem 0;
}}
.timeline-card .t-val {{
    font-family: 'Syne', sans-serif; font-size: 1.35rem; font-weight: 700; color: var(--text-primary);
}}

/* ---- AI BOX ---- */
.ai-box {{
    background: var(--bg-card2); border: 1px solid var(--accent);
    border-radius: 16px; padding: 2rem 2.4rem; position: relative; overflow: hidden;
}}
.ai-box::before {{
    content: '"'; position: absolute; top: -10px; left: 20px;
    font-size: 10rem; color: {T['glow_rgba']}; font-family: Georgia, serif; line-height: 1;
}}
.ai-box-head {{ display: flex; align-items: center; gap: 0.7rem; margin-bottom: 1.2rem; }}
.ai-box-icon {{
    width: 36px; height: 36px; background: {T['glow_rgba']};
    border: 1px solid var(--accent); border-radius: 8px;
    display: flex; align-items: center; justify-content: center; font-size: 1rem;
    opacity: 1 !important;
}}
.ai-box-title {{ font-family: 'Syne', sans-serif; font-size: 0.95rem; font-weight: 700; color: var(--accent); }}
.ai-box-body {{ font-size: 0.95rem; line-height: 1.8; color: var(--text-primary); opacity: 0.9; }}

/* ---- INSIGHT BANNERS ---- */
.insight-banner {{
    border-radius: 14px; padding: 1.4rem 1.8rem;
    display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 1rem;
    background: {T['glow_rgba']}; border: 1px solid var(--accent);
}}
.insight-icon {{ font-size: 1.4rem; margin-top: 0.1rem; opacity: 1 !important; }}
.insight-text {{ font-size: 0.92rem; line-height: 1.6; color: var(--text-primary); }}
.insight-text strong {{ color: var(--accent); }}

/* ---- COUNTERFACTUAL ---- */
.counterfactual-label {{
    font-family: 'Syne', sans-serif; font-size: 0.82rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--accent); margin-bottom: 0.8rem;
}}
.counterfactual-box {{
    background: var(--bg-card2); border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 0 14px 14px 0; padding: 1.6rem 1.8rem;
    font-size: 0.92rem; line-height: 1.8; color: var(--text-primary); opacity: 0.92;
}}

/* ---- MINI BADGE ---- */
.mini-badge {{
    background: var(--bg-card); border-radius: 12px; padding: 1rem; text-align: center;
}}
.mini-badge-val {{ font-family: 'Syne', sans-serif; font-size: 1.5rem; font-weight: 700; }}
.mini-badge-unit {{ font-size: 0.72rem; color: var(--text-dim); }}

/* ---- DIVIDER ---- */
.fancy-divider {{
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-bright), transparent);
    margin: 2.5rem 0;
}}

/* ---- COMPARE BARS ---- */
.compare-row {{ display:flex; align-items:center; gap:0.8rem; margin-bottom:1rem; }}
.compare-label {{ font-size:0.8rem; color:var(--text-muted); min-width:100px;
    text-transform:uppercase; letter-spacing:0.06em; font-weight:500; }}
.compare-bar-wrap {{ flex:1; background:var(--border); border-radius:4px; height:10px; position:relative; overflow:hidden; }}
.compare-bar-ideal {{ height:10px; border-radius:4px; background:var(--accent); opacity:0.22;
    position:absolute; top:0; left:0; }}
.compare-bar-your {{ height:10px; border-radius:4px; position:absolute; top:0; left:0; }}
.compare-val {{ font-family:'Syne',sans-serif; font-size:0.82rem; font-weight:700; min-width:80px; text-align:right; }}

/* ---- CHAT ---- */
.chat-section-wrap {{
    background: var(--bg-card);
    border: 1px solid var(--border-bright);
    border-radius: 20px;
    padding: 2rem 2.4rem;
    margin-top: 1rem;
}}
.chat-section-head {{
    font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700;
    color: var(--accent); margin-bottom: 0.4rem;
    display: flex; align-items: center; gap: 0.6rem;
}}
.chat-section-sub {{
    font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1.4rem;
}}
.chat-wrap {{
    background: var(--bg-card2); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.4rem; margin-bottom: 1rem;
    max-height: 440px; overflow-y: auto;
}}
.chat-msg {{ display:flex; gap:0.8rem; margin-bottom:1.2rem; align-items:flex-start; }}
.chat-msg.user {{ flex-direction:row-reverse; }}
.chat-bubble {{
    max-width:78%; padding:0.85rem 1.1rem; border-radius:14px;
    font-size:0.9rem; line-height:1.6;
}}
.chat-bubble.user {{
    background: linear-gradient(135deg, {T['accent']} 0%, {T['accent2']} 100%);
    color: {T['bg_deep']};
    border-radius: 14px 14px 4px 14px; font-weight: 500;
}}
.chat-bubble.ai {{
    background: var(--bg-card3); border: 1px solid var(--border);
    color: var(--text-primary); border-radius: 14px 14px 14px 4px;
}}
.chat-avatar {{
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
    border: 1px solid var(--border); background: var(--bg-card2);
    opacity: 1 !important;
}}
.sug-btn-row {{ display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem; }}

/* ---- INPUTS ---- */
.stTextInput input {{
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: 10px !important; color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.95rem !important;
    padding: 0.6rem 1rem !important;
}}
.stTextInput input:focus {{
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px {T['glow_rgba']} !important;
}}
div[data-baseweb="select"] > div {{
    background: var(--bg-card) !important; border-color: var(--border) !important;
    color: var(--text-primary) !important; border-radius: 10px !important;
}}
div[data-baseweb="popover"] {{ background: var(--bg-card) !important; border-color: var(--border) !important; }}
.stTextInput label, .stSelectbox label, .stSlider label {{
    color: var(--text-muted) !important; font-size: 0.82rem !important; font-weight: 500 !important;
    letter-spacing: 0.04em !important; text-transform: uppercase !important;
    font-family: 'DM Sans', sans-serif !important;
}}

/* ---- MAIN BUTTON ---- */
.stButton > button {{
    background: linear-gradient(135deg, {T['accent']}22 0%, {T['accent']}11 100%) !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 1rem !important; letter-spacing: 0.05em !important;
    padding: 0.75rem 2.5rem !important; width: 100% !important;
    transition: all 0.25s !important;
    box-shadow: 0 4px 20px {T['glow_rgba']} !important;
}}
.stButton > button:hover {{
    background: var(--accent) !important; color: var(--bg-deep) !important;
    box-shadow: 0 6px 28px {T['glow_rgba']} !important; transform: translateY(-1px) !important;
}}

/* ---- EXPANDER ---- */
.streamlit-expanderHeader,
div[data-testid="stExpander"] > details > summary,
div[data-testid="stExpander"] > details {{
    background: {T['bg_card']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 10px !important;
    color: {T['text_muted']} !important;
    font-family: 'DM Sans', sans-serif !important;
}}
div[data-testid="stExpander"] > details > summary:hover {{
    background: {T['bg_card2']} !important;
    color: {T['text_primary']} !important;
}}
div[data-testid="stExpander"] > details > summary p,
div[data-testid="stExpander"] > details > summary span {{
    color: {T['text_muted']} !important;
}}
div[data-testid="stExpander"] > details[open] > summary {{
    border-radius: 10px 10px 0 0 !important;
    border-bottom: 1px solid {T['border']} !important;
}}
div[data-testid="stExpander"] > details > div {{
    background: {T['bg_card']} !important;
    border: 1px solid {T['border']} !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    color: {T['text_primary']} !important;
}}
div[data-testid="stExpander"] > details > div p,
div[data-testid="stExpander"] > details > div li {{
    color: {T['text_primary']} !important;
}}

/* ---- HIDE CHROME ---- */
#MainMenu, footer, header {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}

/* ---- SCROLLBAR ---- */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: var(--bg-deep); }}
::-webkit-scrollbar-thumb {{ background: var(--border-bright); border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--accent); }}

/* ---- SLIDER ---- */
div[data-testid="stSlider"] > div > div > div {{
    background: var(--accent) !important;
}}
div[data-testid="stSlider"] > div > div {{
    background: var(--border) !important;
}}
</style>
""", unsafe_allow_html=True)

inject_css()

# =========================
# LOAD MODELS
# =========================

crop_model    = joblib.load("models/crop_model.pkl")
le_crop       = joblib.load("models/crop_label_encoder.pkl")
disease_model = joblib.load("models/disease_model.pkl")
le_disease    = joblib.load("models/disease_label_encoder.pkl")
irrig_model   = joblib.load("models/irrigation_model.pkl")
le_irrig_crop = joblib.load("models/irrig_crop_encoder.pkl")
le_soil       = joblib.load("models/soil_encoder.pkl")
le_irrig      = joblib.load("models/irrig_label_encoder.pkl")

# =========================
# HELPERS
# =========================

def convert_shc_to_numeric(level, nutrient):
    fixed = {"N":{"Low":30,"Medium":65,"High":110},"P":{"Low":20,"Medium":50,"High":100},"K":{"Low":40,"Medium":80,"High":150}}
    return fixed[nutrient][level]

def get_weather_from_location(location):
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={location}&appid={API_KEY}&units=metric"
    res = requests.get(url).json()
    if "list" not in res: return None
    temps, humids, rain_total = [], [], 0
    for entry in res["list"]:
        temps.append(entry["main"]["temp"])
        humids.append(entry["main"]["humidity"])
        rain_total += entry.get("rain", {}).get("3h", 0)
    daily = round(rain_total / 5, 2)
    return {
        "avg_temperature":           round(sum(temps)  / len(temps),  2),
        "avg_humidity":              round(sum(humids) / len(humids), 2),
        "avg_rainfall_5day":         daily,
        "estimated_30_day_rainfall": round(daily * 30, 2)
    }

def recommend_crop(N, P, K, temp, humidity, ph, rainfall):
    data  = pd.DataFrame([{"N":N,"P":P,"K":K,"temperature":temp,"humidity":humidity,"ph":ph,"rainfall":rainfall}])
    probs = crop_model.predict_proba(data)[0]
    return sorted(zip(le_crop.classes_, probs), key=lambda x: x[1], reverse=True)

def disease_risk(temp, humidity, rainfall, ph):
    data = pd.DataFrame([{"temperature":temp,"humidity":humidity,"rainfall":rainfall,"soil_pH":ph}])
    return le_disease.inverse_transform([disease_model.predict(data)[0]])[0]

def irrigation_advice(crop, temp, humidity, rainfall, soil_condition):
    crop_enc = le_irrig_crop.transform([crop])[0]
    soil_enc = le_soil.transform([soil_condition])[0]
    data = pd.DataFrame([{"crop":crop_enc,"temperature":temp,"humidity":humidity,"rainfall":rainfall,"soil_condition":soil_enc}])
    return le_irrig.inverse_transform([irrig_model.predict(data)[0]])[0]


GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

@st.cache_data(show_spinner=False)
def generate_ai_explanation(top_crop, disease, irrigation, inputs_tuple, monthly_rainfall):
    inputs = dict(inputs_tuple)
    prompt = f"""You are an expert agricultural advisor. Explain clearly:
1. Why {top_crop} is suitable.
2. Disease prevention for risk: {disease}.
3. Irrigation advice: {irrigation}.
4. Rainfall caution for 30-day rainfall of {monthly_rainfall} mm.
Farm Data — N:{inputs['N']} P:{inputs['P']} K:{inputs['K']} pH:{inputs['ph']}
Soil:{inputs['soil_condition']} Temp:{inputs['temperature']}°C Humidity:{inputs['humidity']}%
Rainfall(5d):{inputs['rainfall']}mm
Keep it farmer-friendly, 6-8 sentences, use emojis."""
    try:
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role":"system","content":"You are a professional agricultural AI advisor."},
                {"role":"user","content":prompt}
            ],
            temperature=0.6,
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"AI service error: {str(e)}"

crop_calendar = {
    "rice":120,"maize":100,"chickpea":100,"kidneybeans":95,"pigeonpeas":160,
    "mothbeans":75,"mungbean":65,"blackgram":75,"lentil":105,"pomegranate":210,
    "banana":300,"mango":365,"grapes":240,"watermelon":90,"muskmelon":85,
    "apple":365,"orange":300,"papaya":270,"coconut":365,"cotton":160,"jute":130,"coffee":365
}

def crop_timeline(crop):
    crop = crop.lower().strip()
    if crop not in crop_calendar: return None
    today    = datetime.today()
    duration = crop_calendar[crop]
    return today.date(), (today + pd.Timedelta(days=duration)).date(), duration

def styled_fig(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans', color=T['plot_text'], size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        title_font=dict(family='Syne', color=T['plot_title'], size=14),
    )
    fig.update_xaxes(gridcolor=T['plot_grid'], color=T['plot_text'], showline=False)
    fig.update_yaxes(gridcolor=T['plot_grid'], color=T['plot_text'], showline=False)
    return fig

# =========================
# SOIL HEALTH SCORE
# =========================

def compute_soil_score(N_level, P_level, K_level, ph):
    level_pts = {"Low":10,"Medium":25,"High":33}
    score = level_pts[N_level] + level_pts[P_level] + level_pts[K_level]
    if 6.0 <= ph <= 7.5:        score += 1
    elif not (5.5 <= ph <= 8.0): score -= 5
    score = max(0, min(100, score))
    if score >= 80:   return score, "Excellent 🌟", T["accent"]
    elif score >= 60: return score, "Good 👍",      T["accent2"]
    elif score >= 40: return score, "Fair ⚠️",      T["accent_blue"]
    else:             return score, "Poor ❌",       T["accent_red"]

IDEAL_RANGES = {
    "rice":{"N":(80,120),"P":(40,70),"K":(60,100),"ph":(5.5,7.0)},
    "maize":{"N":(60,100),"P":(30,60),"K":(50,90),"ph":(5.8,7.0)},
    "chickpea":{"N":(20,45),"P":(40,80),"K":(60,100),"ph":(6.0,8.0)},
    "kidneybeans":{"N":(20,40),"P":(40,80),"K":(50,100),"ph":(6.0,7.5)},
    "pigeonpeas":{"N":(20,40),"P":(40,80),"K":(50,100),"ph":(5.5,7.0)},
    "mothbeans":{"N":(20,40),"P":(30,60),"K":(40,80),"ph":(7.0,8.5)},
    "mungbean":{"N":(20,40),"P":(40,70),"K":(40,80),"ph":(6.0,7.5)},
    "blackgram":{"N":(20,40),"P":(40,70),"K":(40,80),"ph":(6.0,7.5)},
    "lentil":{"N":(20,40),"P":(40,80),"K":(40,80),"ph":(6.0,8.0)},
    "pomegranate":{"N":(60,90),"P":(40,70),"K":(80,120),"ph":(5.5,7.5)},
    "banana":{"N":(80,120),"P":(40,80),"K":(120,160),"ph":(6.0,7.5)},
    "mango":{"N":(60,90),"P":(30,60),"K":(60,100),"ph":(5.5,7.5)},
    "grapes":{"N":(60,90),"P":(50,80),"K":(80,120),"ph":(6.0,7.0)},
    "watermelon":{"N":(80,110),"P":(40,70),"K":(60,100),"ph":(6.0,7.0)},
    "muskmelon":{"N":(80,110),"P":(40,70),"K":(60,100),"ph":(6.0,7.0)},
    "apple":{"N":(60,90),"P":(40,70),"K":(60,100),"ph":(6.0,7.0)},
    "orange":{"N":(60,100),"P":(40,70),"K":(80,120),"ph":(6.0,7.5)},
    "papaya":{"N":(80,110),"P":(40,70),"K":(80,120),"ph":(6.0,7.0)},
    "coconut":{"N":(60,100),"P":(30,60),"K":(80,140),"ph":(5.5,8.0)},
    "cotton":{"N":(80,120),"P":(40,80),"K":(60,100),"ph":(6.0,8.0)},
    "jute":{"N":(60,100),"P":(30,60),"K":(40,80),"ph":(6.0,7.5)},
    "coffee":{"N":(80,120),"P":(30,60),"K":(60,100),"ph":(5.0,6.5)},
}

def get_ideal(crop, nutrient):
    defaults = {"N":(65,90),"P":(40,70),"K":(60,100),"ph":(6.0,7.5)}
    r = IDEAL_RANGES.get(crop.lower().strip(), defaults).get(nutrient, defaults[nutrient])
    return (r[0] + r[1]) / 2

# =========================
# AI CHAT
# =========================

def chat_with_agrobrain(user_msg, history, farm_ctx):
    context = f"""You are AgroBrain, an expert AI farming assistant. The farmer ran an analysis.
Farm context: Location={farm_ctx.get('location','?')}, Top crop={farm_ctx.get('top_crop','?')},
Disease risk={farm_ctx.get('disease','?')}, Irrigation={farm_ctx.get('irrigation','?')},
N={farm_ctx.get('N','?')} P={farm_ctx.get('P','?')} K={farm_ctx.get('K','?')} pH={farm_ctx.get('ph','?')},
Temp={farm_ctx.get('temperature','?')}C Humidity={farm_ctx.get('humidity','?')}%.
Answer farmer questions about their specific farm. Be concise, practical, and use emojis."""
    messages = [{"role":"system","content":context}]
    for h in history[-6:]:
        messages.append({"role":"user","content":h["user"]})
        messages.append({"role":"assistant","content":h["ai"]})
    messages.append({"role":"user","content":user_msg})
    try:
        r = client.chat.completions.create(model="llama-3.1-8b-instant", messages=messages, temperature=0.7, max_tokens=300)
        return r.choices[0].message.content
    except Exception as e:
        return f"Sorry, error: {str(e)}"

# =========================
# HERO
# =========================

st.markdown(f"""
<div class="hero-header">
  <div class="hero-badge">🌱 AI-Powered Platform</div>
  <div class="hero-title">AgroBrain</div>
  <div class="hero-subtitle">Smart Farming Advisory System · Soil · Weather · Crops · Intelligence</div>
</div>
""", unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    with st.expander("🌱 About the System"):
        st.markdown("""AgroBrain analyzes **soil nutrients (N, P, K)**, pH, moisture, and **live weather** to deliver:
- 🌾 Ranked crop recommendations &nbsp; · &nbsp; 🦠 Disease risk prediction
- 💧 Irrigation strategy &nbsp; · &nbsp; 🤖 AI guidance & counterfactual strategies
- 📊 Soil Health Score &nbsp; · &nbsp; 💬 Ask AgroBrain chat""")
with col_b:
    with st.expander("🧑‍🌾 How to Use"):
        st.markdown("""1. Enter your **city name** to pull live weather
2. Select **N, P, K levels** from your Soil Health Card
3. Set **soil pH** and **moisture condition**
4. Hit **Analyze Farm Conditions** — then chat with AgroBrain!""")

st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

# =========================
# STEP 01 — WEATHER
# =========================

st.markdown('<div class="section-label">Step 01</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Location & Weather Intelligence</div>', unsafe_allow_html=True)

avg_temperature = avg_humidity = avg_rainfall = estimated_30_day_rain = None

loc_col, _ = st.columns([2, 1])
with loc_col:
    location = st.text_input("Enter City Name", placeholder="e.g. Chennai, Mumbai, Delhi...")

if location:
    with st.spinner("Fetching weather forecast..."):
        weather = get_weather_from_location(location)
    if weather:
        avg_temperature       = weather["avg_temperature"]
        avg_humidity          = weather["avg_humidity"]
        avg_rainfall          = weather["avg_rainfall_5day"]
        estimated_30_day_rain = weather["estimated_30_day_rainfall"]

        rain_label    = "No rain expected"    if avg_rainfall == 0          else f"{avg_rainfall} mm/day"
        rain_30_label = "No significant rain" if estimated_30_day_rain == 0 else f"{estimated_30_day_rain} mm"

        st.markdown(f"""
        <div style="display:flex; gap:1rem; flex-wrap:wrap; margin-top:1.2rem;">
            <div class="stat-tile" style="flex:1; min-width:130px;">
                <div class="stat-tile-glow"></div>
                <div class="stat-tile-icon">🌡</div>
                <div class="stat-tile-value">{avg_temperature}°C</div>
                <div class="stat-tile-label">5-Day Avg Temp</div>
            </div>
            <div class="stat-tile" style="flex:1; min-width:130px;">
                <div class="stat-tile-glow"></div>
                <div class="stat-tile-icon">💧</div>
                <div class="stat-tile-value">{avg_humidity}%</div>
                <div class="stat-tile-label">5-Day Avg Humidity</div>
            </div>
            <div class="stat-tile" style="flex:1; min-width:130px;">
                <div class="stat-tile-glow"></div>
                <div class="stat-tile-icon">🌧</div>
                <div class="stat-tile-value">{rain_label}</div>
                <div class="stat-tile-label">Daily Rainfall (5-day)</div>
            </div>
            <div class="stat-tile" style="flex:1; min-width:130px;">
                <div class="stat-tile-glow"></div>
                <div class="stat-tile-icon">📅</div>
                <div class="stat-tile-value">{rain_30_label}</div>
                <div class="stat-tile-label">30-Day Projection</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("⚠️ Unable to fetch weather data. Check city name or API key.")

st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

# =========================
# STEP 02 — SOIL
# =========================

st.markdown('<div class="section-label">Step 02</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Soil Health Card Inputs</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1: N_level = st.selectbox("Nitrogen Level (SHC)",   ["Low", "Medium", "High"])
with c2: P_level = st.selectbox("Phosphorus Level (SHC)", ["Low", "Medium", "High"])
with c3: K_level = st.selectbox("Potassium Level (SHC)",  ["Low", "Medium", "High"])

ph_col, soil_col = st.columns(2)
with ph_col:   ph             = st.slider("Soil pH", 3.5, 9.0, 6.5)
with soil_col: soil_condition = st.selectbox("Soil Moisture Condition", ["Dry", "Moist", "Wet"])

N_val = convert_shc_to_numeric(N_level, "N")
P_val = convert_shc_to_numeric(P_level, "P")
K_val = convert_shc_to_numeric(K_level, "K")

level_colors = {"Low": T["accent_red"], "Medium": T["accent_blue"], "High": T["accent"]}

def mini_badge(label, level, value, unit="kg/ha"):
    c = level_colors[level]
    return f"""
    <div class="mini-badge" style="border:1px solid {c}44; border-top:2px solid {c};">
      <div style="font-size:0.7rem; color:var(--text-muted); letter-spacing:0.1em;
           text-transform:uppercase; margin-bottom:0.4rem;">{label}</div>
      <div class="mini-badge-val" style="color:{c};">{value}</div>
      <div class="mini-badge-unit">{unit} · {level}</div>
    </div>"""

ph_level = "Medium" if 5.5 <= ph <= 7.5 else ("Low" if ph < 5.5 else "High")
cn, cp, ck, cph = st.columns(4)
with cn:  st.markdown(mini_badge("Nitrogen",   N_level,  N_val),           unsafe_allow_html=True)
with cp:  st.markdown(mini_badge("Phosphorus", P_level,  P_val),           unsafe_allow_html=True)
with ck:  st.markdown(mini_badge("Potassium",  K_level,  K_val),           unsafe_allow_html=True)
with cph: st.markdown(mini_badge("Soil pH",    ph_level, ph, "pH level"),  unsafe_allow_html=True)

st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

btn_col, _ = st.columns([1, 2])
with btn_col:
    analyze = st.button("⚡ Analyze Farm Conditions")

# =========================
# RESULTS — cached in session_state
# =========================

# Build a key from current inputs to detect changes
current_inputs_key = (location, N_level, P_level, K_level, ph, soil_condition)

# When Analyze is clicked: run models, store ALL results in session_state
if analyze:
    if avg_temperature is None:
        st.error("⚠️ Please enter a valid city name first to fetch weather data.")
    else:
        N = convert_shc_to_numeric(N_level, "N")
        P = convert_shc_to_numeric(P_level, "P")
        K = convert_shc_to_numeric(K_level, "K")

        inputs = {"N":N,"P":P,"K":K,"temperature":avg_temperature,"humidity":avg_humidity,
                  "ph":ph,"rainfall":avg_rainfall,"soil_condition":soil_condition}

        ranked_crops             = recommend_crop(N, P, K, avg_temperature, avg_humidity, ph, avg_rainfall)
        top_crop,    top_prob    = ranked_crops[0]
        second_crop, second_prob = ranked_crops[1]
        prob_diff                = (top_prob - second_prob) * 100
        top_5_crops              = ranked_crops[:5]
        disease                  = disease_risk(avg_temperature, avg_humidity, avg_rainfall, ph)
        irrig_top                = irrigation_advice(top_crop,    avg_temperature, avg_humidity, avg_rainfall, soil_condition)
        irrig_sec                = irrigation_advice(second_crop, avg_temperature, avg_humidity, avg_rainfall, soil_condition)

        with st.spinner("🤖 Generating AI advisory..."):
            explanation = generate_ai_explanation(
                top_crop, disease, irrig_top, tuple(sorted(inputs.items())), estimated_30_day_rain
            )

        # Counterfactual if needed
        alt_text = None
        if prob_diff < 10:
            with st.spinner("🌿 Analyzing alternative crop strategy..."):
                alt_prompt = f"""Recommended: {top_crop} ({top_prob*100:.1f}%). Second: {second_crop} ({second_prob*100:.1f}%).
Explain how to grow {second_crop} successfully. Include soil adjustments, irrigation, fertilizer. Concise & practical."""
                alt_res  = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role":"user","content":alt_prompt}])
                alt_text = alt_res.choices[0].message.content

        # Store everything
        st.session_state.last_results = {
            "inputs": inputs, "N": N, "P": P, "K": K,
            "ranked_crops": ranked_crops, "top_crop": top_crop, "top_prob": top_prob,
            "second_crop": second_crop, "second_prob": second_prob, "prob_diff": prob_diff,
            "top_5_crops": top_5_crops, "disease": disease,
            "irrig_top": irrig_top, "irrig_sec": irrig_sec,
            "explanation": explanation, "alt_text": alt_text,
            "estimated_30_day_rain": estimated_30_day_rain,
        }
        st.session_state.farm_context = {
            "location": location, "top_crop": top_crop, "disease": disease,
            "irrigation": irrig_top, "N": N, "P": P, "K": K, "ph": ph,
            "temperature": avg_temperature, "humidity": avg_humidity,
            "rainfall": avg_rainfall, "soil_condition": soil_condition
        }
        st.session_state.last_inputs_key = current_inputs_key
        st.session_state.analysis_done = True
        # Clear chat when new analysis is run
        st.session_state.chat_history = []

# =========================
# RENDER RESULTS (from session_state — persists across chat reruns)
# =========================

if st.session_state.last_results is not None:
    R = st.session_state.last_results
    top_crop    = R["top_crop"];    top_prob    = R["top_prob"]
    second_crop = R["second_crop"]; second_prob = R["second_prob"]
    prob_diff   = R["prob_diff"];   top_5_crops = R["top_5_crops"]
    disease     = R["disease"];     explanation = R["explanation"]
    irrig_top   = R["irrig_top"];   irrig_sec   = R["irrig_sec"]
    alt_text    = R["alt_text"];    N = R["N"]; P = R["P"]; K = R["K"]
    inputs      = R["inputs"]
    estimated_30_day_rain = R["estimated_30_day_rain"]

    # ---- ph from farm_context (needed for soil score) ----
    ph_val = st.session_state.farm_context.get("ph", 6.5)
    ph_level_r = "Medium" if 5.5 <= ph_val <= 7.5 else ("Low" if ph_val < 5.5 else "High")
    N_level_r  = "Low" if N <= 40 else ("Medium" if N <= 80 else "High")
    P_level_r  = "Low" if P <= 30 else ("Medium" if P <= 70 else "High")
    K_level_r  = "Low" if K <= 55 else ("Medium" if K <= 100 else "High")

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ---- SOIL HEALTH SCORE ----
    st.markdown('<div class="section-label">Soil Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Soil Health Score & Crop Comparison</div>', unsafe_allow_html=True)

    soil_score, soil_grade, score_color = compute_soil_score(N_level_r, P_level_r, K_level_r, ph_val)
    score_col, compare_col = st.columns([1, 2], gap="large")

    with score_col:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=soil_score,
            domain={"x":[0,1],"y":[0,1]},
            gauge={
                "axis":{"range":[0,100],"tickcolor":T["plot_text"],"tickfont":{"color":T["plot_text"]}},
                "bar":{"color":score_color,"thickness":0.25},
                "bgcolor":"rgba(0,0,0,0)","bordercolor":T["border"],
                "steps":[
                    {"range":[0,40],  "color":T["gauge_poor"]},
                    {"range":[40,60], "color":T["gauge_fair"]},
                    {"range":[60,80], "color":T["gauge_good"]},
                    {"range":[80,100],"color":T["gauge_great"]},
                ],
                "threshold":{"line":{"color":score_color,"width":3},"thickness":0.75,"value":soil_score}
            },
            number={"font":{"family":"Syne","color":score_color,"size":42},"suffix":"/100"},
            title={"text":f"Soil Health<br><span style='font-size:0.9em;color:{score_color}'>{soil_grade}</span>",
                   "font":{"family":"Syne","color":T["plot_title"],"size":14}}
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=260,
            margin=dict(l=20,r=20,t=30,b=10), font=dict(color=T["plot_text"]))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with compare_col:
        st.markdown(f'<div style="font-family:Syne,sans-serif;font-size:0.72rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:{T["accent"]};margin-bottom:1rem;">Your Soil vs Ideal for {top_crop.capitalize()}</div>', unsafe_allow_html=True)
        max_vals  = {"N":120,"P":100,"K":150,"ph":9.0}
        your_vals = {"N":N,"P":P,"K":K,"ph":ph_val}
        ideal_vals= {"N":get_ideal(top_crop,"N"),"P":get_ideal(top_crop,"P"),"K":get_ideal(top_crop,"K"),"ph":get_ideal(top_crop,"ph")}
        labels    = {"N":"Nitrogen","P":"Phosphorus","K":"Potassium","ph":"Soil pH"}
        for nutrient, label in labels.items():
            your_v  = your_vals[nutrient]
            ideal_v = ideal_vals[nutrient]
            max_v   = max_vals[nutrient]
            your_pct  = min(100, int(your_v  / max_v * 100))
            ideal_pct = min(100, int(ideal_v / max_v * 100))
            diff      = your_v - ideal_v
            diff_str  = f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}"
            diff_color= T["accent_green"] if abs(diff) < ideal_v*0.4 else T["accent_red"]
            st.markdown(f"""
            <div class="compare-row">
              <div class="compare-label">{label}</div>
              <div class="compare-bar-wrap">
                <div class="compare-bar-ideal" style="width:{ideal_pct}%;"></div>
                <div class="compare-bar-your" style="width:{your_pct}%;background:{T['accent']};opacity:0.85;"></div>
              </div>
              <div class="compare-val">
                <span style="color:{T['text_primary']}">{your_v}</span>
                <span style="font-size:0.7rem;color:{diff_color};margin-left:4px;">({diff_str})</span>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown(f'<div style="margin-top:0.8rem;font-size:0.78rem;color:var(--text-muted);"><span style="display:inline-block;width:12px;height:12px;background:{T["accent"]};opacity:0.85;border-radius:2px;margin-right:6px;"></span>Your soil &nbsp;&nbsp;<span style="display:inline-block;width:12px;height:12px;background:{T["accent"]};opacity:0.22;border-radius:2px;margin-right:6px;"></span>Ideal for {top_crop.capitalize()}</div>', unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ---- CROP RECOMMENDATIONS ----
    st.markdown('<div class="section-label">Analysis Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Top 5 Crop Recommendations</div>', unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1.2], gap="large")
    with left_col:
        for i, (crop, prob) in enumerate(top_5_crops):
            is_top  = i == 0
            bar_pct = int(prob * 100 / top_prob)
            st.markdown(f"""
            <div class="crop-rank-item {'top' if is_top else ''}">
                <span class="crop-rank-badge {'top' if is_top else ''}">#{'★' if is_top else i+1}</span>
                <span class="crop-rank-name">{crop}</span>
                <span class="crop-rank-pct">{prob*100:.1f}%</span>
                <div class="crop-rank-bar-bg"><div class="crop-rank-bar-fill" style="width:{bar_pct}%"></div></div>
            </div>""", unsafe_allow_html=True)

    with right_col:
        chart_df = pd.DataFrame({
            "Crop":        [c[0].capitalize() for c in top_5_crops],
            "Suitability": [round(c[1]*100, 2) for c in top_5_crops]
        })
        fig = px.bar(chart_df, x="Crop", y="Suitability", color="Suitability",
            color_continuous_scale=T["bar_grad"], text="Suitability", title="Crop Suitability Score (%)")
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', textfont_color=T["plot_title"])
        fig = styled_fig(fig)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ---- ADVISORY ----
    st.markdown('<div class="section-label">Crop Advisory</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Top 2 Crop Advisory</div>', unsafe_allow_html=True)

    disease_color = {"High": T["accent_red"], "Medium": T["accent_blue"], "Low": T["accent_green"]}
    d_color       = disease_color.get(disease, T["accent_blue"])

    adv_col1, adv_col2 = st.columns(2, gap="large")
    for col, crop_name, irrig, is_primary in [
        (adv_col1, top_crop,    irrig_top, True),
        (adv_col2, second_crop, irrig_sec, False)
    ]:
        with col:
            tag  = "⭐ Primary Pick" if is_primary else "2nd Choice"
            prob = top_prob if is_primary else second_prob
            st.markdown(f"""
            <div class="advisory-card {'primary' if is_primary else ''}">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
                <div class="advisory-card-head">{crop_name.capitalize()}</div>
                <span style="font-size:0.72rem; background:{T['glow_rgba']}; border:1px solid {T['accent']};
                    color:{T['accent']}; padding:0.25rem 0.7rem; border-radius:100px; font-weight:600;">{tag}</span>
              </div>
              <div>
                <div class="advisory-pill">
                  <span class="pill-label">Disease Risk</span>
                  <span class="pill-value" style="color:{d_color};">{disease}</span>
                </div>
                <div class="advisory-pill">
                  <span class="pill-label">Irrigation</span>
                  <span class="pill-value">{irrig}</span>
                </div>
                <div class="advisory-pill">
                  <span class="pill-label">Suitability</span>
                  <span class="pill-value">{prob*100:.1f}%</span>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ---- TIMELINE ----
    st.markdown('<div class="section-label">Planning</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Crop Planning Timeline</div>', unsafe_allow_html=True)

    timeline = crop_timeline(top_crop)
    if timeline:
        plant, harvest, days = timeline
        t1, t2, t3 = st.columns(3)
        for tcol, icon, label, val in [
            (t1, "🌱", "Planting Date",    plant.strftime("%d %b %Y")),
            (t2, "🌾", "Expected Harvest", harvest.strftime("%d %b %Y")),
            (t3, "⏱", "Growth Duration",  f"{days} days"),
        ]:
            with tcol:
                st.markdown(f"""<div class="timeline-card">
                    <div style="font-size:2rem; margin-bottom:0.5rem; opacity:1;">{icon}</div>
                    <h4>{label}</h4><div class="t-val">{val}</div>
                </div>""", unsafe_allow_html=True)

        fig_t = go.Figure()
        fig_t.add_trace(go.Bar(
            x=[days], y=[top_crop.capitalize()], base=[0], orientation='h',
            marker=dict(color=T["gantt_fill"], line=dict(color=T["gantt_line"], width=1.5)),
            width=0.4,
            hovertemplate=f"Plant: {plant}<br>Harvest: {harvest}<br>Duration: {days} days<extra></extra>"
        ))
        fig_t.update_layout(
            title=f"Growth Window — {top_crop.capitalize()}", xaxis_title="Days from Today", height=140,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='DM Sans', color=T["plot_text"]),
            margin=dict(l=10, r=10, t=40, b=10),
            title_font=dict(family='Syne', color=T["plot_title"], size=13),
            xaxis=dict(gridcolor=T["plot_grid"], color=T["plot_text"]),
            yaxis=dict(gridcolor=T["plot_grid"], color=T["plot_text"]),
            showlegend=False,
        )
        st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("Timeline data not available for this crop.")

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ---- AI EXPLANATION ----
    st.markdown('<div class="section-label">AI Advisory</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">AgroBrain Knows Best</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ai-box">
      <div class="ai-box-head">
        <div class="ai-box-icon">🤖</div>
        <div class="ai-box-title">AgroBrain AI — Farming Advisor</div>
      </div>
      <div class="ai-box-body">{explanation}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ---- INSIGHTS / COUNTERFACTUAL ----
    st.markdown('<div class="section-label">Smart Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">What AgroBrain Discovered</div>', unsafe_allow_html=True)

    if prob_diff < 10 and alt_text:
        st.markdown(f"""
        <div class="insight-banner">
          <div class="insight-icon">⚠️</div>
          <div class="insight-text">
            <strong>{second_crop.capitalize()}</strong> is also a viable option —
            only <strong>{prob_diff:.1f}%</strong> lower suitability than the top pick.
            Conditions are close enough that both crops are worth considering.
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="margin-top:1rem;">
          <div class="counterfactual-label">🌿 Alternative Crop Strategy — {second_crop.capitalize()}</div>
          <div class="counterfactual-box">{alt_text}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="insight-banner">
          <div class="insight-icon">✅</div>
          <div class="insight-text">
            Current soil and weather conditions <strong>strongly favour {top_crop.capitalize()}</strong>
            as the recommended crop — with a {prob_diff:.1f}% lead over the next best option.
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

# =========================
# ASK AGROBRAIN CHAT
# — shown only when analysis exists; results above always rendered from session_state
# =========================

if st.session_state.analysis_done:
    top_crop_ctx = st.session_state.farm_context.get("top_crop", "your crop")

    st.markdown(f"""
    <div class="chat-section-wrap">
      <div class="chat-section-head">💬 Ask AgroBrain</div>
      <div class="chat-section-sub">
        Chat about your {top_crop_ctx.capitalize()} farm — fertilizer, pests, market timing, organics, or anything else.
      </div>
    """, unsafe_allow_html=True)

    # Chat history display
    if st.session_state.chat_history:
        chat_html = '<div class="chat-wrap">'
        for turn in st.session_state.chat_history:
            chat_html += f"""
            <div class="chat-msg user">
                <div class="chat-bubble user">{turn["user"]}</div>
                <div class="chat-avatar">👨‍🌾</div>
            </div>
            <div class="chat-msg">
                <div class="chat-avatar">🤖</div>
                <div class="chat-bubble ai">{turn["ai"]}</div>
            </div>"""
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close chat-section-wrap

    # Suggested questions
    suggestions = [
        f"How much fertilizer for {top_crop_ctx}?",
        "What pests should I watch for this season?",
        "Best time to sell my harvest?",
        "Any organic alternatives to chemical fertilizers?",
    ]
    st.markdown(f'<div style="font-size:0.78rem;color:{T["text_muted"]};margin:0.8rem 0 0.4rem;">💡 Suggested questions:</div>', unsafe_allow_html=True)
    sug_cols = st.columns(2)
    for i, sug in enumerate(suggestions):
        with sug_cols[i % 2]:
            if st.button(sug, key=f"sug_{i}"):
                with st.spinner("🤖 AgroBrain is thinking..."):
                    ai_reply = chat_with_agrobrain(sug, st.session_state.chat_history, st.session_state.farm_context)
                st.session_state.chat_history.append({"user": sug, "ai": ai_reply})
                st.rerun()

    chat_col, send_col = st.columns([5, 1])
    with chat_col:
        user_question = st.text_input("Message AgroBrain...", key="chat_input", label_visibility="collapsed", placeholder="Ask anything about your farm...")
    with send_col:
        send = st.button("Send ➤", key="chat_send")

    if send and user_question.strip():
        with st.spinner("🤖 AgroBrain is thinking..."):
            ai_reply = chat_with_agrobrain(user_question, st.session_state.chat_history, st.session_state.farm_context)
        st.session_state.chat_history.append({"user": user_question, "ai": ai_reply})
        st.rerun()

    if st.session_state.chat_history:
        _, clr_col, _ = st.columns([3, 1, 3])
        with clr_col:
            if st.button("🗑 Clear Chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

# =========================
# FOOTER
# =========================

st.markdown(f"""
<div style="text-align:center; padding: 1.5rem 0;">
    <span style="font-family:'Syne',sans-serif; font-size:0.75rem; color:{T['text_dim']};
        letter-spacing:0.1em; text-transform:uppercase;">
        AgroBrain · AI-Enhanced Smart Farming Advisory
    </span>
</div>
""", unsafe_allow_html=True)
