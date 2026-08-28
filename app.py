import streamlit as st
import pandas as pd
import numpy as np
import shap
import math
from xgboost import XGBClassifier

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CreditSight",
    page_icon="\u25C8",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# THEME STATE
# =========================================================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

DARK = st.session_state.dark_mode

# =========================================================
# DESIGN TOKENS
# "Signal" system — near-black glass base, one saturated
# lime accent (the "signal"), violet as its gradient partner.
# =========================================================
if DARK:
    BG          = "#06070C"
    BG_SOFT     = "#0A0C15"
    GLASS       = "rgba(255,255,255,0.035)"
    GLASS_HOVER = "rgba(255,255,255,0.055)"
    CARD_BORDER = "rgba(255,255,255,0.09)"
    DIVIDER     = "rgba(255,255,255,0.07)"
    TEXT        = "#F4F5FA"
    SUBTEXT     = "#9599B5"
    MUTE        = "#5D6180"
    LIME        = "#C6FF5E"
    LIME_GLOW   = "rgba(198,255,94,0.35)"
    LIME_SOFT   = "rgba(198,255,94,0.12)"
    VIOLET      = "#9B8CFF"
    VIOLET_SOFT = "rgba(155,140,255,0.14)"
    TRACK       = "rgba(255,255,255,0.07)"
    RISK_LOW    = "#6EE7B7"
    RISK_LOW_SOFT  = "rgba(110,231,183,0.14)"
    RISK_MED    = "#FFC168"
    RISK_MED_SOFT  = "rgba(255,193,104,0.14)"
    RISK_HIGH   = "#FF6B6B"
    RISK_HIGH_SOFT = "rgba(255,107,107,0.14)"
    BTN_TEXT    = "#06070C"
    INPUT_BG    = "rgba(255,255,255,0.045)"
else:
    BG          = "#F3F4F8"
    BG_SOFT     = "#EAEBF2"
    GLASS       = "rgba(255,255,255,0.75)"
    GLASS_HOVER = "rgba(255,255,255,0.92)"
    CARD_BORDER = "rgba(20,22,40,0.09)"
    DIVIDER     = "rgba(20,22,40,0.08)"
    TEXT        = "#12131F"
    SUBTEXT     = "#54586E"
    MUTE        = "#8A8DA3"
    LIME        = "#5C8A1E"
    LIME_GLOW   = "rgba(92,138,30,0.20)"
    LIME_SOFT   = "rgba(92,138,30,0.10)"
    VIOLET      = "#5A4CD6"
    VIOLET_SOFT = "rgba(90,76,214,0.10)"
    TRACK       = "rgba(20,22,40,0.08)"
    RISK_LOW    = "#1E9E73"
    RISK_LOW_SOFT  = "rgba(30,158,115,0.12)"
    RISK_MED    = "#B4790C"
    RISK_MED_SOFT  = "rgba(180,121,12,0.12)"
    RISK_HIGH   = "#D23C50"
    RISK_HIGH_SOFT = "rgba(210,60,80,0.12)"
    BTN_TEXT    = "#FFFFFF"
    INPUT_BG    = "rgba(20,22,40,0.04)"

# =========================================================
# GLOBAL CSS
# =========================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

.stApp {{
    background:
        radial-gradient(ellipse 60% 45% at 12% -8%, {LIME_GLOW} 0%, transparent 60%),
        radial-gradient(ellipse 55% 40% at 100% 8%, {VIOLET_SOFT} 0%, transparent 55%),
        {BG};
    color: {TEXT};
}}

#MainMenu, footer, header {{visibility: hidden;}}

.block-container {{
    padding-top: 1.1rem;
    padding-bottom: 3rem;
    padding-left: clamp(1rem, 4vw, 3.6rem);
    padding-right: clamp(1rem, 4vw, 3.6rem);
    max-width: 1480px;
}}

h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif; }}

/* ---------- Glass card base ---------- */
.glass {{
    background: {GLASS};
    border: 1px solid {CARD_BORDER};
    border-radius: 20px;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
}}

/* ---------- Top bar ---------- */
.topbar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.8rem 1.3rem;
    margin-bottom: 1.6rem;
}}
.brand {{ display: flex; align-items: center; gap: 0.65rem; }}
.brand-mark {{
    width: 34px; height: 34px;
    border-radius: 10px;
    background: linear-gradient(135deg, {LIME} 0%, {VIOLET} 140%);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 0 22px {LIME_GLOW};
}}
.brand-name {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    color: {TEXT};
    letter-spacing: -0.01em;
}}
.brand-tag {{
    color: {MUTE};
    font-size: 0.72rem;
    font-weight: 500;
    display: block;
    margin-top: -1px;
}}

/* ---------- Hero ---------- */
.hero-card {{
    padding: clamp(1.8rem, 3.5vw, 3rem) clamp(1.8rem, 4vw, 3.2rem);
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
}}
.hero-eyebrow {{
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {LIME};
    background: {LIME_SOFT};
    border: 1px solid {LIME_GLOW};
    padding: 0.32rem 0.75rem;
    border-radius: 999px;
    margin-bottom: 1.1rem;
}}
.eyebrow-dot {{
    width: 6px; height: 6px; border-radius: 50%;
    background: {LIME};
    box-shadow: 0 0 8px {LIME};
}}
.hero-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(1.9rem, 3.6vw, 3rem);
    font-weight: 700;
    color: {TEXT};
    line-height: 1.1;
    letter-spacing: -0.02em;
    margin-bottom: 0.9rem;
    max-width: 760px;
}}
.hero-title .accent {{ color: {LIME}; }}
.hero-sub {{
    color: {SUBTEXT};
    font-size: clamp(0.92rem, 1.1vw, 1.05rem);
    max-width: 560px;
    line-height: 1.6;
    margin-bottom: 1.8rem;
}}
.stat-row {{
    display: flex;
    gap: clamp(1.2rem, 3vw, 2.6rem);
    flex-wrap: wrap;
}}
.stat-item {{ display: flex; flex-direction: column; }}
.stat-num {{
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    font-size: clamp(1.25rem, 1.8vw, 1.55rem);
    color: {TEXT};
}}
.stat-label {{
    color: {MUTE};
    font-size: 0.7rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 2px;
}}

/* ---------- Pulse waveform (signature element) ---------- */
.pulse-holder {{
    position: absolute;
    right: -20px; bottom: -10px;
    width: min(46%, 480px);
    height: 130px;
    opacity: 0.9;
    pointer-events: none;
}}
@media (max-width: 900px) {{ .pulse-holder {{ display: none; }} }}
.pulse-path {{
    stroke-dasharray: 1000;
    stroke-dashoffset: 1000;
    animation: draw 3.2s ease-out forwards, glow-pulse 2.4s ease-in-out infinite 3.2s;
}}
@keyframes draw {{ to {{ stroke-dashoffset: 0; }} }}
@keyframes glow-pulse {{
    0%, 100% {{ filter: drop-shadow(0 0 5px {LIME_GLOW}); }}
    50% {{ filter: drop-shadow(0 0 14px {LIME_GLOW}); }}
}}
@media (prefers-reduced-motion: reduce) {{
    .pulse-path {{ animation: none; stroke-dashoffset: 0; }}
}}

/* ---------- Section cards ---------- */
.section-card {{
    padding: 1.7rem 1.8rem;
    margin-bottom: 1.3rem;
}}
.card-eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.66rem;
    font-weight: 600;
    color: {VIOLET};
    margin-bottom: 0.5rem;
}}
.card-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: {TEXT};
    margin-bottom: 0.25rem;
}}
.card-desc {{
    font-size: 0.85rem;
    color: {SUBTEXT};
    margin-bottom: 1.3rem;
}}
.fieldset-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: {MUTE};
    margin: 0.3rem 0 0.6rem 0;
    padding-top: 0.9rem;
    border-top: 1px solid {DIVIDER};
}}

/* ---------- Streamlit widget overrides ---------- */
.stSlider label, .stNumberInput label, .stSelectbox label {{
    color: {TEXT} !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    font-family: 'Inter', sans-serif !important;
}}
.stNumberInput input, div[data-baseweb="select"] > div {{
    background-color: {INPUT_BG} !important;
    color: {TEXT} !important;
    border: 1px solid {CARD_BORDER} !important;
    border-radius: 10px !important;
}}
div[data-baseweb="popover"] li {{
    background-color: {BG_SOFT} !important;
    color: {TEXT} !important;
}}
.stSlider [data-baseweb="slider"] > div > div {{ background: {TRACK} !important; }}
.stSlider [data-baseweb="slider"] div[role="slider"] {{
    background-color: {LIME} !important;
    box-shadow: 0 0 0 5px {LIME_SOFT} !important;
}}

/* ---------- Predict button ---------- */
div.stButton > button {{
    background: linear-gradient(100deg, {LIME} 0%, #A8E84A 100%);
    color: {BTN_TEXT};
    border: none;
    border-radius: 12px;
    padding: 0.78rem 1.6rem;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 0.96rem;
    width: 100%;
    box-shadow: 0 0 0 1px rgba(255,255,255,0.06) inset, 0 10px 30px -10px {LIME_GLOW};
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
div.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 0 0 1px rgba(255,255,255,0.1) inset, 0 14px 34px -8px {LIME_GLOW};
}}
div.stButton > button:active {{ transform: translateY(0px); }}

.theme-btn button {{
    background: {GLASS} !important;
    backdrop-filter: blur(18px);
    color: {TEXT} !important;
    border: 1px solid {CARD_BORDER} !important;
    box-shadow: none !important;
    font-size: 1.05rem !important;
    border-radius: 12px !important;
    width: 44px !important;
    height: 44px !important;
    min-width: 44px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    line-height: 1 !important;
}}
.theme-btn {{ display: flex; justify-content: flex-end; }}
.theme-btn > div {{ width: auto !important; }}

/* ---------- Empty state (slim, full-width prompt bar) ---------- */
.empty-state-slim {{
    display: flex;
    align-items: center;
    gap: 0.9rem;
    padding: 1.1rem 1.5rem;
}}

/* ---------- Gauge ---------- */
.gauge-wrap {{
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 0.3rem 0 0.1rem 0;
}}
.gauge-svg-holder {{ position: relative; width: 236px; height: 138px; }}
.gauge-center {{
    position: absolute; top: 60px; left: 0; right: 0; text-align: center;
}}
.gauge-score {{
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    font-size: 2.9rem;
    line-height: 1;
}}
.gauge-outof {{
    color: {MUTE};
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 3px;
}}
.risk-pill {{
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.36rem 1rem;
    border-radius: 999px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 0.8rem;
    margin-top: 1rem;
}}
.risk-pill .dot {{ width: 7px; height: 7px; border-radius: 50%; }}
.proba-line {{
    color: {SUBTEXT};
    font-size: 0.83rem;
    margin-top: 0.65rem;
    text-align: center;
}}
.proba-line b {{ color: {TEXT}; font-family: 'IBM Plex Mono', monospace; }}

/* ---------- Signal meter ---------- */
.signal-row {{
    display: grid;
    grid-template-columns: minmax(0,1.3fr) minmax(0,1fr) auto;
    align-items: center;
    gap: 0.9rem;
    padding: 0.72rem 0;
    border-bottom: 1px solid {DIVIDER};
}}
.signal-row:last-child {{ border-bottom: none; }}
.signal-name {{ font-weight: 600; font-size: 0.87rem; color: {TEXT}; }}
.signal-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem; color: {MUTE};
    display: block; margin-top: 1px;
}}
.signal-track {{ display: flex; height: 8px; width: 100%; min-width: 90px; }}
.track-half {{ flex: 1; display: flex; position: relative; }}
.track-half.left {{ justify-content: flex-end; }}
.track-half.right {{ justify-content: flex-start; }}
.track-half::before {{
    content: ''; position: absolute; top: 0; bottom: 0; width: 100%;
    background: {TRACK};
}}
.track-half.left::before {{ right: 0; border-radius: 6px 0 0 6px; }}
.track-half.right::before {{ left: 0; border-radius: 0 6px 6px 0; }}
.bar-fill {{ height: 8px; z-index: 1; position: relative; }}
.bar-fill.left {{ border-radius: 6px 2px 2px 6px; }}
.bar-fill.right {{ border-radius: 2px 6px 6px 2px; }}
.signal-tag {{
    font-size: 0.71rem; font-weight: 700;
    padding: 0.28rem 0.6rem; border-radius: 8px;
    white-space: nowrap; justify-self: end;
}}

.legend {{
    display: flex; gap: 1.4rem; margin-bottom: 1.15rem;
    font-size: 0.75rem; color: {SUBTEXT};
}}
.legend span {{ display: flex; align-items: center; gap: 0.4rem; }}
.legend i {{ width: 8px; height: 8px; border-radius: 3px; display: inline-block; }}

/* ---------- Disclaimer + footer ---------- */
.disclaimer {{
    padding: 1.05rem 1.35rem;
    font-size: 0.79rem;
    color: {SUBTEXT};
    line-height: 1.55;
    margin-top: 1.6rem;
}}
.disclaimer b {{ color: {TEXT}; font-family: 'Space Grotesk', sans-serif; }}
.footer-note {{
    text-align: center;
    color: {MUTE};
    font-size: 0.75rem;
    margin-top: 2rem;
    padding-top: 1.2rem;
    border-top: 1px solid {DIVIDER};
}}

@media (max-width: 768px) {{
    .section-card, .hero-card {{ padding: 1.3rem 1.2rem; }}
    .stat-row {{ gap: 1.1rem; }}
}}
</style>
""", unsafe_allow_html=True)

# =========================================================
# TOP BAR
# =========================================================
top_l, top_r = st.columns([10, 1])
with top_l:
    st.markdown(f"""
    <div class="topbar glass">
        <div class="brand">
            <div class="brand-mark">
                <svg width="17" height="17" viewBox="0 0 17 17" fill="none">
                    <rect x="1" y="8.5" width="3.2" height="7.5" rx="1" fill="{BG}"/>
                    <rect x="6.9" y="3.8" width="3.2" height="12.2" rx="1" fill="{BG}"/>
                    <rect x="12.8" y="0.5" width="3.2" height="15.5" rx="1" fill="{BG}"/>
                </svg>
            </div>
            <div>
                <div class="brand-name">CreditSight</div>
                <span class="brand-tag">Alternative Credit Scoring</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with top_r:
    st.markdown('<div class="theme-btn">', unsafe_allow_html=True)
    st.button("\u2600\ufe0f" if DARK else "\U0001F319", on_click=toggle_theme, help="Switch to light mode" if DARK else "Switch to dark mode")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# HERO
# =========================================================
st.markdown(f"""
<div class="hero-card glass">
    <div class="hero-eyebrow"><span class="eyebrow-dot"></span>Behavioral Credit Model</div>
    <div class="hero-title">Creditworthiness, read<br>from <span class="accent">signal</span>, not paperwork.</div>
    <div class="hero-sub">CreditSight scores thin-file and first-time borrowers using payment
    regularity, spending discipline, and income stability \u2014 the behavioral signals
    traditional bureaus never see.</div>
    <div class="stat-row">
        <div class="stat-item"><div class="stat-num">0.83</div><div class="stat-label">Model AUC</div></div>
        <div class="stat-item"><div class="stat-num">13</div><div class="stat-label">Behavioral signals</div></div>
        <div class="stat-item"><div class="stat-num">190M+</div><div class="stat-label">Unbanked, India</div></div>
        <div class="stat-item"><div class="stat-num">SHAP</div><div class="stat-label">Explainable output</div></div>
    </div>
    <div class="pulse-holder">
        <svg viewBox="0 0 480 130" width="100%" height="100%" preserveAspectRatio="xMaxYMid meet">
            <path class="pulse-path" d="M0,65 L70,65 L95,65 L112,20 L132,110 L152,40 L168,65 L200,65
                     L230,65 L252,30 L272,95 L292,50 L310,65 L480,65"
                  fill="none" stroke="{LIME}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("xgb_model.json")
    explainer = shap.TreeExplainer(model)
    return model, explainer

model, explainer = load_model()

# Hardcoded from your model's actual trained column order (confirmed
# directly from the XGBoost mismatch error) rather than read dynamically
# from get_booster().feature_names, since that comes back empty on some
# XGBoost versions after load_model() and silently falls back to a guess.
FEATURE_ORDER = [
    "spending_discipline_ratio", "age", "payment_irregularity_minor",
    "income_obligation_ratio", "income_level", "financial_activity_count",
    "payment_irregularity_severe", "property_loan_count", "payment_irregularity_moderate",
    "dependents_count", "income_unreported", "payment_regularity_score", "income_stability",
]

FEATURE_LABELS = {
    "spending_discipline_ratio": "Credit utilization ratio",
    "age": "Age",
    "payment_irregularity_minor": "Minor late payments (30\u201359d)",
    "payment_irregularity_moderate": "Moderate late payments (60\u201389d)",
    "payment_irregularity_severe": "Severe late payments (90+d)",
    "income_obligation_ratio": "Debt-to-income ratio",
    "income_level": "Monthly income",
    "income_unreported": "Income undocumented",
    "financial_activity_count": "Active credit lines / loans",
    "property_loan_count": "Property loans",
    "dependents_count": "Dependents",
    "payment_regularity_score": "Payment regularity score",
    "income_stability": "Income stability tier",
}

# =========================================================
# FORM — full width
# =========================================================
st.markdown('<div class="section-card glass">', unsafe_allow_html=True)
st.markdown('<div class="card-eyebrow">01 \u2014 Input</div>', unsafe_allow_html=True)
st.markdown('<div class="card-title">Your financial behavior</div>', unsafe_allow_html=True)
st.markdown('<div class="card-desc">Nothing here is stored \u2014 the score is computed live, in this browser session only.</div>', unsafe_allow_html=True)

st.markdown('<div class="fieldset-label">Personal</div>', unsafe_allow_html=True)
p1, p2, p3, p4 = st.columns(4)
with p1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
with p2:
    income_level = st.number_input("Monthly income (\u20b9)", min_value=0, value=25000, step=1000)
with p3:
    dependents_count = st.number_input("Dependents", min_value=0, max_value=10, value=0)
with p4:
    income_unreported = st.selectbox("Income documented?", ["Yes", "No"])

st.markdown('<div class="fieldset-label">Credit behavior</div>', unsafe_allow_html=True)
b1, b2, b3, b4 = st.columns(4)
with b1:
    spending_discipline_ratio = st.slider("Credit utilization ratio", 0.0, 1.5, 0.3, help="Portion of available credit currently in use.")
with b2:
    income_obligation_ratio = st.slider("Debt-to-income ratio", 0.0, 2.0, 0.3, help="Monthly debt obligations \u00f7 monthly income.")
with b3:
    financial_activity_count = st.number_input("Active credit lines / loans", min_value=0, max_value=30, value=3)
with b4:
    property_loan_count = st.number_input("Property / real estate loans", min_value=0, max_value=10, value=0)

st.markdown('<div class="fieldset-label">Payment history</div>', unsafe_allow_html=True)
l1, l2, l3 = st.columns(3)
with l1:
    payment_irregularity_minor = st.number_input("30\u201359 days late", min_value=0, max_value=20, value=0)
with l2:
    payment_irregularity_moderate = st.number_input("60\u201389 days late", min_value=0, max_value=20, value=0)
with l3:
    payment_irregularity_severe = st.number_input("90+ days late", min_value=0, max_value=20, value=0)

st.markdown('</div>', unsafe_allow_html=True)
predict_clicked = st.button("Calculate my score \u2192")

# =========================================================
# GAUGE SVG BUILDER
# =========================================================
def gauge_svg(score, color):
    r = 86
    cx, cy = 118, 112
    circumference = math.pi * r
    frac = max(0, min(score, 100)) / 100
    offset = circumference * (1 - frac)
    return f"""
    <div class="gauge-svg-holder">
    <svg width="236" height="138" viewBox="0 0 236 138">
        <defs>
            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="4" result="blur"/>
                <feMerge>
                    <feMergeNode in="blur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
        </defs>
        <path d="M {cx-r} {cy} A {r} {r} 0 0 1 {cx+r} {cy}"
              fill="none" stroke="{TRACK}" stroke-width="14" stroke-linecap="round"/>
        <path d="M {cx-r} {cy} A {r} {r} 0 0 1 {cx+r} {cy}"
              fill="none" stroke="{color}" stroke-width="14" stroke-linecap="round"
              stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}"
              filter="url(#glow)"/>
    </svg>
    <div class="gauge-center">
        <div class="gauge-score" style="color:{color};">{score}</div>
        <div class="gauge-outof">out of 100</div>
    </div>
    </div>
    """

# =========================================================
# RESULT — full width, below the form
# =========================================================
if not predict_clicked:
    st.markdown(f"""
    <div class="section-card glass empty-state-slim">
        <svg width="26" height="26" viewBox="0 0 26 26" fill="none" style="flex-shrink:0;">
            <circle cx="13" cy="13" r="11.5" stroke="{LIME}" stroke-width="1.5" stroke-dasharray="3 4.2" opacity="0.7"/>
            <path d="M13 7v6l4 2.5" stroke="{LIME}" stroke-width="1.6" stroke-linecap="round"/>
        </svg>
        <div>
            <div class="card-title" style="font-size:1rem; margin-bottom:0.1rem;">Your result appears here</div>
            <div class="card-desc" style="margin-bottom:0;">Fill in the form above and calculate to see your score and the signals behind it.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    payment_regularity_score = (
        payment_irregularity_minor * 1
        + payment_irregularity_moderate * 2
        + payment_irregularity_severe * 3
    )
    if income_obligation_ratio <= 0.3:
        income_stability = 0
    elif income_obligation_ratio <= 0.6:
        income_stability = 1
    else:
        income_stability = 2
    income_unreported_val = 1 if income_unreported == "No" else 0

    row = pd.DataFrame([{
        "spending_discipline_ratio": spending_discipline_ratio,
        "age": age,
        "payment_irregularity_minor": payment_irregularity_minor,
        "payment_irregularity_moderate": payment_irregularity_moderate,
        "payment_irregularity_severe": payment_irregularity_severe,
        "income_obligation_ratio": income_obligation_ratio,
        "income_level": income_level,
        "income_unreported": income_unreported_val,
        "financial_activity_count": financial_activity_count,
        "property_loan_count": property_loan_count,
        "dependents_count": dependents_count,
        "payment_regularity_score": payment_regularity_score,
        "income_stability": income_stability,
    }])[FEATURE_ORDER]

    proba_default = model.predict_proba(row)[0, 1]
    credit_score = int(round((1 - proba_default) * 100))

    if credit_score >= 70:
        risk_label, risk_color, risk_soft = "Low Risk", RISK_LOW, RISK_LOW_SOFT
    elif credit_score >= 45:
        risk_label, risk_color, risk_soft = "Moderate Risk", RISK_MED, RISK_MED_SOFT
    else:
        risk_label, risk_color, risk_soft = "High Risk", RISK_HIGH, RISK_HIGH_SOFT

    sv = explainer(row)

    gauge_col, signal_col = st.columns([1, 1.7], gap="large")

    with gauge_col:
        st.markdown('<div class="section-card glass gauge-wrap" style="height:100%;">', unsafe_allow_html=True)
        st.markdown('<div class="card-eyebrow" style="align-self:flex-start;">02 \u2014 Result</div>', unsafe_allow_html=True)
        st.markdown(gauge_svg(credit_score, risk_color), unsafe_allow_html=True)
        st.markdown(f'<span class="risk-pill" style="background:{risk_soft}; color:{risk_color};"><span class="dot" style="background:{risk_color};"></span>{risk_label}</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="proba-line">Estimated probability of default: <b>{proba_default*100:.1f}%</b></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with signal_col:
        values = sv[0].values
        data_vals = sv[0].data
        contributions = list(zip(FEATURE_ORDER, values, data_vals))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        top_factors = contributions[:6]
        max_abs = max(abs(v) for _, v, _ in top_factors) or 1.0

        st.markdown('<div class="section-card glass">', unsafe_allow_html=True)
        st.markdown('<div class="card-title" style="font-size:1.02rem;">Signal breakdown</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="legend">
            <span><i style="background:{RISK_LOW};"></i>Decreases risk</span>
            <span><i style="background:{RISK_HIGH};"></i>Increases risk</span>
        </div>
        """, unsafe_allow_html=True)

        for name, shap_val, raw_val in top_factors:
            up = shap_val > 0
            color = RISK_HIGH if up else RISK_LOW
            pct = (abs(shap_val) / max_abs) * 100
            label = FEATURE_LABELS.get(name, name)
            left_bar = f'<div class="bar-fill left" style="width:{pct if not up else 0:.0f}%; background:{RISK_LOW};"></div>' if not up else ""
            right_bar = f'<div class="bar-fill right" style="width:{pct if up else 0:.0f}%; background:{RISK_HIGH};"></div>' if up else ""
            st.markdown(f"""
            <div class="signal-row">
                <div>
                    <div class="signal-name">{label}</div>
                    <span class="signal-value">value: {raw_val:.2f}</span>
                </div>
                <div class="signal-track">
                    <div class="track-half left">{left_bar}</div>
                    <div class="track-half right">{right_bar}</div>
                </div>
                <div class="signal-tag" style="background:{color}22; color:{color};">{"+ risk" if up else "\u2212 risk"}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# DISCLAIMER + FOOTER
# =========================================================
st.markdown(f"""
<div class="disclaimer glass">
    <b>Disclaimer \u2014</b> CreditSight is an academic mini-project demonstrating alternative
    credit scoring on publicly available, reframed data. It is not a licensed credit
    bureau product, does not access real financial accounts, and its output should not
    inform actual lending or borrowing decisions. All calculations run locally in this
    session; no input is stored or transmitted.
</div>
""", unsafe_allow_html=True)

st.markdown(f'<div class="footer-note">CreditSight \u2014 Alternative Credit Scoring for the Unbanked \u00b7 Mini Project \u00b7 Hindustan College of Science and Technology \u00b7 Oorvi Kulshreshtha & Kunal Rathore</div>', unsafe_allow_html=True)