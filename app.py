import streamlit as st
import pandas as pd
import numpy as np
import shap
import math
import urllib.parse
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
# Flat near-black base, single warm gold accent, subtle
# scattered fintech-glyph texture behind everything.
# =========================================================
if DARK:
    BG          = "#08090C"
    BG_SOFT     = "#0C0D12"
    GLASS       = "rgba(20,21,28,0.72)"
    GLASS_HOVER = "rgba(26,27,35,0.85)"
    CARD_BORDER = "rgba(255,255,255,0.08)"
    DIVIDER     = "rgba(255,255,255,0.07)"
    TEXT        = "#F5F4F0"
    SUBTEXT     = "#9C9CA8"
    MUTE        = "#5F5F6C"
    LIME        = "#14C9A6"
    LIME_2      = "#5EEAD4"
    LIME_GLOW   = "rgba(20,201,166,0.35)"
    LIME_SOFT   = "rgba(20,201,166,0.13)"
    VIOLET      = "#14C9A6"
    VIOLET_SOFT = "rgba(20,201,166,0.10)"
    TRACK       = "rgba(255,255,255,0.07)"
    RISK_LOW    = "#5FD98E"
    RISK_LOW_SOFT  = "rgba(95,217,142,0.14)"
    RISK_MED    = "#E8A33D"
    RISK_MED_SOFT  = "rgba(232,163,61,0.14)"
    RISK_HIGH   = "#F1596E"
    RISK_HIGH_SOFT = "rgba(241,89,110,0.14)"
    BTN_TEXT    = "#062420"
    INPUT_BG    = "rgba(255,255,255,0.045)"
else:
    BG          = "#F5F4EF"
    BG_SOFT     = "#ECEAE1"
    GLASS       = "rgba(255,255,255,0.86)"
    GLASS_HOVER = "rgba(255,255,255,0.96)"
    CARD_BORDER = "rgba(20,22,40,0.09)"
    DIVIDER     = "rgba(20,22,40,0.08)"
    TEXT        = "#17150E"
    SUBTEXT     = "#54586E"
    MUTE        = "#8A8DA3"
    LIME        = "#0B7C74"
    LIME_2      = "#12A594"
    LIME_GLOW   = "rgba(11,124,116,0.20)"
    LIME_SOFT   = "rgba(11,124,116,0.10)"
    VIOLET      = "#0B7C74"
    VIOLET_SOFT = "rgba(11,124,116,0.08)"
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
# BACKGROUND TEXTURE — a fine dot-grid, very low opacity.
# Generic "data surface" motif (not tied to any specific
# reference) that reads as structure, not decoration.
# =========================================================
_tex_rgb = "255,255,255" if DARK else "20,22,40"
_tex_svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='28' height='28'>
<circle cx='2' cy='2' r='1.1' fill='rgba({_tex_rgb},0.06)'/>
</svg>"""
BG_TEXTURE_URI = "data:image/svg+xml," + urllib.parse.quote(_tex_svg)

# =========================================================
# GLOBAL CSS
# =========================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600;700&display=swap');

/* Neutralize Streamlit's own theme accent (default red
   #FF4B4B) wherever BaseWeb components reference it via CSS
   custom property, e.g. the tab active-indicator bar. */
:root, .stApp {{
    --primary-color: {LIME} !important;
}}

/* Confirmed via DevTools inspection: Streamlit's newer Tabs
   uses React Aria internally, not BaseWeb — the active-tab
   underline is this exact class, not any data-baseweb attr. */
.react-aria-SelectionIndicator {{
    background: transparent !important;
    display: none !important;
    height: 0 !important;
    width: 0 !important;
}}

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

.stApp {{
    background-image:
        url("{BG_TEXTURE_URI}"),
        radial-gradient(ellipse 60% 45% at 12% -8%, {LIME_GLOW} 0%, transparent 62%),
        radial-gradient(ellipse 55% 40% at 100% 8%, {VIOLET_SOFT} 0%, transparent 58%);
    background-repeat: repeat, no-repeat, no-repeat;
    background-size: 28px 28px, auto, auto;
    background-position: 0 0, 12% -8%, 100% 8%;
    background-color: {BG};
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

/* ---------- Card base (flat, near-solid — not heavy glass) ---------- */
.glass {{
    background: {GLASS};
    border: 1px solid {CARD_BORDER};
    border-radius: 20px;
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
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
    background: linear-gradient(135deg, {LIME} 0%, {LIME_2} 140%);
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
    font-size: clamp(2.1rem, 4.1vw, 3.35rem);
    font-weight: 800;
    color: {TEXT};
    line-height: 1.08;
    letter-spacing: -0.025em;
    margin-bottom: 0.9rem;
    max-width: 780px;
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
    transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
}}
.section-card:hover {{
    transform: translateY(-3px);
    border-color: {LIME_GLOW};
    box-shadow: 0 16px 40px -20px rgba(0,0,0,0.5);
}}
.hero-card {{ transition: border-color 0.3s ease; }}
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
    display: flex;
    align-items: center;
    gap: 0.45rem;
}}
.fieldset-label svg {{ flex-shrink: 0; opacity: 0.85; }}

/* ---------- Streamlit widget overrides ---------- */
.stSlider label, .stNumberInput label, .stSelectbox label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {{
    color: {TEXT} !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    font-family: 'Inter', sans-serif !important;
}}
.stNumberInput input, div[data-baseweb="select"] > div,
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
    background-color: {INPUT_BG} !important;
    color: {TEXT} !important;
    border: 1px solid {CARD_BORDER} !important;
    border-radius: 10px !important;
}}
/* Universal fallback: whatever wrapper class Streamlit actually
   uses, the visible text on any number/text input must always
   be readable against the theme, not a stale hardcoded color. */
input[type="number"], input[type="text"] {{
    background-color: {INPUT_BG} !important;
    color: {TEXT} !important;
    -webkit-text-fill-color: {TEXT} !important;
}}
div[data-baseweb="select"] * {{
    color: {TEXT} !important;
}}
div[data-baseweb="popover"] li {{
    background-color: {BG_SOFT} !important;
    color: {TEXT} !important;
}}
.stSlider [data-baseweb="slider"] > div > div,
div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {{
    background: {TRACK} !important;
}}
.stSlider [data-baseweb="slider"] div[role="slider"],
div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
    background-color: {LIME} !important;
    box-shadow: 0 0 0 5px {LIME_SOFT} !important;
}}

/* ---------- Predict button ---------- */
div.stButton > button {{
    background: linear-gradient(100deg, {LIME} 0%, {LIME_2} 100%);
    color: {BTN_TEXT};
    border: none;
    border-radius: 999px;
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
    backdrop-filter: blur(6px);
    color: {LIME} !important;
    border: 1px solid {CARD_BORDER} !important;
    box-shadow: none !important;
    font-size: 1.05rem !important;
    border-radius: 999px !important;
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

/* ---------- Download buttons (secondary, outline pill) ---------- */
div[data-testid="stDownloadButton"] > button {{
    background: transparent !important;
    color: {LIME} !important;
    border: 1.5px solid {LIME_GLOW} !important;
    border-radius: 999px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1.3rem !important;
    box-shadow: none !important;
    transition: all 0.18s ease !important;
}}
div[data-testid="stDownloadButton"] > button:hover {{
    background: {LIME_SOFT} !important;
    border-color: {LIME} !important;
}}

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

/* ---------- What-if simulator ---------- */
.whatif-result {{
    display: flex;
    align-items: center;
    gap: 1.6rem;
    margin-top: 0.9rem;
    padding-top: 1rem;
    border-top: 1px solid {DIVIDER};
    flex-wrap: wrap;
}}
.whatif-item {{ display: flex; flex-direction: column; }}
.whatif-arrow {{
    font-size: 1.3rem;
    color: {MUTE};
    font-family: 'Space Grotesk', sans-serif;
}}

/* ---------- Batch tier / approval bars ---------- */
.tier-bar {{
    display: flex;
    height: 10px;
    width: 100%;
    border-radius: 6px;
    overflow: hidden;
    background: {TRACK};
    margin: 0.9rem 0 0.5rem 0;
}}
.tier-seg {{ height: 10px; }}

/* ---------- Streamlit tabs override (pill nav, gradient active state) ---------- */
div[data-testid="stTabs"] {{ margin-top: 0.4rem; }}
div[data-testid="stTabs"] [data-baseweb="tab-list"],
div[data-testid="stTabs"] [role="tablist"] {{
    gap: 0.7rem !important;
    background: transparent !important;
    border-bottom: none !important;
    flex-wrap: wrap !important;
    overflow: visible !important;
    padding: 0 0 1.6rem 0 !important;
}}
div[data-testid="stTabs"] [data-baseweb="tab"],
div[data-testid="stTabs"] [role="tab"] {{
    background: {GLASS} !important;
    border: 1px solid {CARD_BORDER} !important;
    border-radius: 999px !important;
    color: {SUBTEXT} !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.85rem 2rem !important;
    min-width: max-content !important;
    white-space: nowrap !important;
    outline: none !important;
    box-shadow: none !important;
    transition: all 0.18s ease !important;
}}
div[data-testid="stTabs"] [data-baseweb="tab"] p,
div[data-testid="stTabs"] [role="tab"] p {{
    display: flex !important;
    align-items: center !important;
    gap: 0.55rem !important;
    margin: 0 !important;
    white-space: nowrap !important;
}}
div[data-testid="stTabs"] [data-baseweb="tab"]:focus,
div[data-testid="stTabs"] [data-baseweb="tab"]:focus-visible,
div[data-testid="stTabs"] [data-baseweb="tab"]:active,
div[data-testid="stTabs"] [role="tab"]:focus,
div[data-testid="stTabs"] [role="tab"]:focus-visible,
div[data-testid="stTabs"] [role="tab"]:active {{
    outline: none !important;
    box-shadow: none !important;
}}
div[data-testid="stTabs"] [aria-selected="true"] {{
    background: linear-gradient(100deg, {LIME} 0%, {LIME_2} 100%) !important;
    color: {BTN_TEXT} !important;
    border: 1px solid transparent !important;
    font-weight: 700 !important;
    box-shadow: 0 8px 22px -10px {LIME_GLOW} !important;
}}
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
    background: transparent !important;
    background-color: transparent !important;
    border-bottom: none !important;
    box-shadow: none !important;
    height: 0 !important;
    display: none !important;
}}
div[data-testid="stTabs"] [data-baseweb="tab"] [data-baseweb="tab-highlight"] {{
    display: none !important;
    height: 0 !important;
}}
div[data-testid="stTabs"] [data-baseweb="tab"]::after,
div[data-testid="stTabs"] [data-baseweb="tab"]::before,
div[data-testid="stTabs"] [role="tab"]::after,
div[data-testid="stTabs"] [role="tab"]::before {{
    display: none !important;
    content: none !important;
}}
div[data-testid="stTabs"] [data-baseweb="tab-border"] {{ display: none !important; }}
div[data-testid="stTabs"] [data-baseweb="tab-panel"] {{ padding-top: 0.5rem !important; }}
/* Brute-force fallback: hide any element in the tab bar that
   isn't literally a tab button, whatever its real attribute
   name turns out to be (catches the active-indicator bar). */
div[data-testid="stTabs"] [data-baseweb="tab-list"] > *:not([data-baseweb="tab"]):not([role="tab"]) {{
    display: none !important;
    height: 0 !important;
}}

/* ---------- Affordability estimator ---------- */
.afford-label {{
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {MUTE};
    margin-bottom: 0.3rem;
}}
.afford-num {{
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    font-size: 1.35rem;
    color: {TEXT};
}}
.afford-num span {{
    font-size: 0.68rem;
    font-weight: 500;
    color: {MUTE};
    font-family: 'Inter', sans-serif;
}}
.afford-note {{
    font-size: 0.75rem;
    color: {MUTE};
    margin-top: 1.1rem;
    padding-top: 0.9rem;
    border-top: 1px solid {DIVIDER};
}}

/* ---------- Regulatory context ---------- */
.reg-note {{
    font-size: 0.85rem;
    color: {SUBTEXT};
    line-height: 1.55;
}}
.reg-list {{
    margin: 0.9rem 0 0 0;
    padding-left: 1.1rem;
    font-size: 0.85rem;
    color: {SUBTEXT};
    line-height: 1.7;
}}
.reg-list li {{ margin-bottom: 0.5rem; }}
.reg-list b {{ color: {TEXT}; font-weight: 600; }}

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
    .afford-num {{ font-size: 1.15rem; }}
    .whatif-result {{ gap: 1rem; }}
    .gauge-svg-holder {{ width: 200px; height: 118px; }}
    div[data-testid="stTabs"] [data-baseweb="tab"] {{ padding: 0.65rem 1.3rem !important; font-size: 0.82rem !important; }}
}}
@media (max-width: 520px) {{
    .signal-row {{
        grid-template-columns: 1fr;
        row-gap: 0.4rem;
    }}
    .signal-tag {{ justify-self: start; }}
    .whatif-item {{ min-width: 90px; }}
    .brand-tag {{ display: none; }}
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
    "spending_discipline_ratio": "Credit limit used",
    "age": "Age",
    "payment_irregularity_minor": "Paid 1\u20132 months late",
    "payment_irregularity_moderate": "Paid 2\u20133 months late",
    "payment_irregularity_severe": "Paid 3+ months late",
    "income_obligation_ratio": "Income spent on debt",
    "income_level": "Monthly income",
    "income_unreported": "No income proof",
    "financial_activity_count": "Loans / credit cards held",
    "property_loan_count": "Home or property loans",
    "dependents_count": "People depending on you",
    "payment_regularity_score": "Overall payment regularity",
    "income_stability": "Income stability level",
}

# =========================================================
# TABS — separate individual vs. institutional flows so
# neither page gets cluttered with the other's controls
# =========================================================
tab1, tab2 = st.tabs(["\U0001F9CD\u2002Individual Score", "\U0001F3E6\u2002Portfolio Review (Bank / NBFC)"])

with tab1:
    # =========================================================
    # FORM — full width
    # =========================================================
    st.markdown('<div class="section-card glass">', unsafe_allow_html=True)
    st.markdown('<div class="card-eyebrow">01 \u2014 Input</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Your financial behavior</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-desc">Nothing here is stored \u2014 the score is computed live, in this browser session only.</div>', unsafe_allow_html=True)

    st.markdown('''<div class="fieldset-label"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.5-7 8-7s8 3 8 7"/></svg>Personal</div>''', unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, help="Your age in years.")
    with p2:
        income_level = st.number_input("Monthly income (\u20b9)", min_value=0, value=25000, step=1000, help="Your total monthly income before any deductions.")
    with p3:
        dependents_count = st.number_input("People who depend on you", min_value=0, max_value=10, value=0, help="Children, spouse, parents, or anyone who relies on your income for support.")
    with p4:
        income_unreported = st.selectbox("Do you have income proof?", ["Yes", "No"], help="Select 'No' if your income is informal (e.g. cash-based, gig work) and you don't have documents like salary slips, bank statements, or tax returns to show it.")

    st.markdown('''<div class="fieldset-label"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="5" width="20" height="14" rx="2"/><line x1="2" y1="10" x2="22" y2="10"/></svg>Credit behavior</div>''', unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        spending_discipline_pct = st.slider("Credit limit you're using", 0, 150, 30, format="%d%%", help="If your credit card limit is \u20b91,00,000 and you've spent \u20b930,000 on it, you're using 30%. Generally, staying under 30% is seen as healthy. Going over 100% means you've exceeded your limit.")
        spending_discipline_ratio = spending_discipline_pct / 100
    with b2:
        income_obligation_pct = st.slider("Income spent repaying debt", 0, 200, 30, format="%d%%", help="Add up all your monthly loan/EMI/credit card payments and divide by your monthly income. E.g. \u20b925,000 income and \u20b97,500 in EMIs = 30%. Going over 100% means your debt payments exceed your income.")
        income_obligation_ratio = income_obligation_pct / 100
    with b3:
        financial_activity_count = st.number_input("Loans or credit cards you have", min_value=0, max_value=30, value=3, help="Count personal loans, credit cards, car loans, and any other loans that are currently open (not yet fully paid off).")
    with b4:
        property_loan_count = st.number_input("Home or property loans", min_value=0, max_value=10, value=0, help="Number of home loans or loans taken against property that you currently have.")

    st.markdown('''<div class="fieldset-label"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/></svg>Payment history (last 2 years)</div>''', unsafe_allow_html=True)
    l1, l2, l3 = st.columns(3)
    with l1:
        payment_irregularity_minor = st.number_input("Paid 1\u20132 months late", min_value=0, max_value=20, value=0, help="Number of times you paid a bill, EMI, or credit card due about 1 to 2 months after the due date.")
    with l2:
        payment_irregularity_moderate = st.number_input("Paid 2\u20133 months late", min_value=0, max_value=20, value=0, help="Number of times a payment was about 2 to 3 months overdue.")
    with l3:
        payment_irregularity_severe = st.number_input("Paid 3+ months late", min_value=0, max_value=20, value=0, help="Number of times a payment was 3 or more months overdue. This is treated as a serious warning sign and affects your score the most.")

    st.markdown('</div>', unsafe_allow_html=True)

    if "score_calculated" not in st.session_state:
        st.session_state.score_calculated = False
    if st.button("Calculate my score \u2192"):
        st.session_state.score_calculated = True

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
            <style>
                @keyframes gaugeFill {{
                    from {{ stroke-dashoffset: {circumference:.2f}; }}
                    to {{ stroke-dashoffset: {offset:.2f}; }}
                }}
                .gauge-fill-path {{
                    animation: gaugeFill 1s cubic-bezier(0.22, 1, 0.36, 1) forwards;
                }}
                @media (prefers-reduced-motion: reduce) {{
                    .gauge-fill-path {{ animation: none; stroke-dashoffset: {offset:.2f}; }}
                }}
            </style>
            <path d="M {cx-r} {cy} A {r} {r} 0 0 1 {cx+r} {cy}"
                  fill="none" stroke="{TRACK}" stroke-width="14" stroke-linecap="round"/>
            <path class="gauge-fill-path" d="M {cx-r} {cy} A {r} {r} 0 0 1 {cx+r} {cy}"
                  fill="none" stroke="{color}" stroke-width="14" stroke-linecap="round"
                  stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{circumference:.2f}"
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
    if not st.session_state.score_calculated:
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

        # =====================================================
        # LOAN AFFORDABILITY ESTIMATOR — full width, uses the
        # already-computed score + income + obligation ratio
        # =====================================================
        if credit_score >= 70:
            safe_emi_ratio = 0.50
        elif credit_score >= 45:
            safe_emi_ratio = 0.35
        else:
            safe_emi_ratio = 0.20

        safe_emi_ceiling = income_level * safe_emi_ratio
        existing_obligation = income_level * income_obligation_ratio
        available_capacity = max(safe_emi_ceiling - existing_obligation, 0)

        illus_rate_annual = 0.12
        illus_tenure_months = 36
        r = illus_rate_annual / 12
        n = illus_tenure_months
        if available_capacity > 0:
            indicative_loan = available_capacity * ((1 + r) ** n - 1) / (r * (1 + r) ** n)
        else:
            indicative_loan = 0

        st.markdown('<div class="section-card glass">', unsafe_allow_html=True)
        st.markdown('<div class="card-eyebrow">03 \u2014 Affordability</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title" style="font-size:1.05rem;">Indicative loan affordability</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-desc">Based on your risk tier\u2019s safe EMI-to-income ceiling, minus obligations you\u2019ve already reported.</div>', unsafe_allow_html=True)

        a1, a2, a3, a4 = st.columns(4)
        with a1:
            st.markdown(f'<div class="afford-label">Safe EMI ceiling</div><div class="afford-num">\u20b9{safe_emi_ceiling:,.0f}<span>/mo</span></div>', unsafe_allow_html=True)
        with a2:
            st.markdown(f'<div class="afford-label">Already committed</div><div class="afford-num">\u20b9{existing_obligation:,.0f}<span>/mo</span></div>', unsafe_allow_html=True)
        with a3:
            st.markdown(f'<div class="afford-label">Available capacity</div><div class="afford-num" style="color:{LIME};">\u20b9{available_capacity:,.0f}<span>/mo</span></div>', unsafe_allow_html=True)
        with a4:
            st.markdown(f'<div class="afford-label">Indicative max loan</div><div class="afford-num" style="color:{LIME};">\u20b9{indicative_loan:,.0f}</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="afford-note">Illustrative only \u2014 assumes {illus_rate_annual*100:.0f}% p.a. interest over {illus_tenure_months} months. Not a loan offer or eligibility guarantee.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # =====================================================
        # WHAT-IF SIMULATOR — same score/row context as above,
        # lets the user drag the top levers and see the delta
        # =====================================================
        st.markdown('<div class="section-card glass">', unsafe_allow_html=True)
        st.markdown('<div class="card-eyebrow">04 \u2014 Explore</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title" style="font-size:1.05rem;">What would improve your score?</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-desc">Drag these to see how your score would change \u2014 nothing here affects the result above.</div>', unsafe_allow_html=True)

        w1, w2, w3 = st.columns(3)
        with w1:
            wi_spending_pct = st.slider("Credit limit used", 0, 150, int(round(spending_discipline_ratio * 100)), format="%d%%", key="wi_spending")
            wi_spending = wi_spending_pct / 100
        with w2:
            wi_obligation_pct = st.slider("Income spent on debt", 0, 200, int(round(income_obligation_ratio * 100)), format="%d%%", key="wi_obligation")
            wi_obligation = wi_obligation_pct / 100
        with w3:
            wi_severe = st.number_input("Times paid 3+ months late", min_value=0, max_value=20, value=int(payment_irregularity_severe), key="wi_severe")

        wi_regularity_score = (
            payment_irregularity_minor * 1
            + payment_irregularity_moderate * 2
            + wi_severe * 3
        )
        if wi_obligation <= 0.3:
            wi_stability = 0
        elif wi_obligation <= 0.6:
            wi_stability = 1
        else:
            wi_stability = 2

        wi_row = pd.DataFrame([{
            "spending_discipline_ratio": wi_spending,
            "age": age,
            "payment_irregularity_minor": payment_irregularity_minor,
            "payment_irregularity_moderate": payment_irregularity_moderate,
            "payment_irregularity_severe": wi_severe,
            "income_obligation_ratio": wi_obligation,
            "income_level": income_level,
            "income_unreported": income_unreported_val,
            "financial_activity_count": financial_activity_count,
            "property_loan_count": property_loan_count,
            "dependents_count": dependents_count,
            "payment_regularity_score": wi_regularity_score,
            "income_stability": wi_stability,
        }])[FEATURE_ORDER]

        wi_proba = model.predict_proba(wi_row)[0, 1]
        wi_score = int(round((1 - wi_proba) * 100))
        delta = wi_score - credit_score

        if delta > 0:
            delta_color, delta_text = RISK_LOW, f"+{delta}"
        elif delta < 0:
            delta_color, delta_text = RISK_HIGH, f"{delta}"
        else:
            delta_color, delta_text = MUTE, "0"

        st.markdown(f"""
        <div class="whatif-result">
            <div class="whatif-item">
                <div class="afford-label">Current score</div>
                <div class="afford-num">{credit_score}</div>
            </div>
            <div class="whatif-arrow">\u2192</div>
            <div class="whatif-item">
                <div class="afford-label">Simulated score</div>
                <div class="afford-num" style="color:{delta_color};">{wi_score}</div>
            </div>
            <div class="whatif-item">
                <div class="afford-label">Change</div>
                <div class="afford-num" style="color:{delta_color};">{delta_text}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:

    st.markdown('<div class="section-card glass">', unsafe_allow_html=True)
    st.markdown('<div class="card-eyebrow">Batch Review</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Score a portfolio of applicants</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-desc">Upload a CSV of applicants to score them all at once, set your own approval threshold, and export the results. Nothing is stored beyond this session.</div>', unsafe_allow_html=True)

    sample_df = pd.DataFrame([
        {"age": 34, "monthly_income": 28000, "dependents": 1, "income_documented": "Yes",
         "credit_utilization_ratio": 0.25, "debt_to_income_ratio": 0.30, "active_credit_lines": 3,
         "property_loans": 0, "late_30_59": 0, "late_60_89": 0, "late_90_plus": 0},
        {"age": 45, "monthly_income": 19000, "dependents": 3, "income_documented": "No",
         "credit_utilization_ratio": 0.62, "debt_to_income_ratio": 0.71, "active_credit_lines": 5,
         "property_loans": 1, "late_30_59": 2, "late_60_89": 1, "late_90_plus": 0},
    ])
    sample_csv = sample_df.to_csv(index=False).encode("utf-8")

    dl1, dl2 = st.columns([1, 3])
    with dl1:
        st.download_button("\u2B07 Download CSV template", data=sample_csv, file_name="creditsight_batch_template.csv", mime="text/csv", use_container_width=True)

    uploaded = st.file_uploader("Upload applicant CSV", type=None, label_visibility="collapsed", help="If your CSV appears greyed out on mobile, this now accepts any file type and checks the format after upload instead.")
    st.markdown('</div>', unsafe_allow_html=True)

    REQUIRED_COLS = [
        "age", "monthly_income", "dependents", "income_documented",
        "credit_utilization_ratio", "debt_to_income_ratio", "active_credit_lines",
        "property_loans", "late_30_59", "late_60_89", "late_90_plus",
    ]

    if uploaded is not None:
        if not uploaded.name.lower().endswith(".csv"):
            st.error("That doesn\u2019t look like a CSV file \u2014 please upload a file ending in .csv.")
            batch_df = None
        else:
            try:
                batch_df = pd.read_csv(uploaded)
            except Exception:
                batch_df = None
                st.error("Couldn\u2019t read that file \u2014 make sure it\u2019s a valid CSV.")

        if batch_df is not None:
            missing = [c for c in REQUIRED_COLS if c not in batch_df.columns]
            if missing:
                st.error(f"Missing required column(s): {', '.join(missing)}. Use the template above for the expected format.")
            else:
                work = batch_df.copy()
                work["payment_regularity_score"] = (
                    work["late_30_59"] * 1 + work["late_60_89"] * 2 + work["late_90_plus"] * 3
                )
                work["income_stability"] = pd.cut(
                    work["debt_to_income_ratio"],
                    bins=[-0.01, 0.3, 0.6, work["debt_to_income_ratio"].max() + 0.01],
                    labels=[0, 1, 2],
                ).astype(int)
                work["income_unreported"] = (work["income_documented"].astype(str).str.strip().str.lower() == "no").astype(int)

                model_input = pd.DataFrame({
                    "spending_discipline_ratio": work["credit_utilization_ratio"],
                    "age": work["age"],
                    "payment_irregularity_minor": work["late_30_59"],
                    "payment_irregularity_moderate": work["late_60_89"],
                    "payment_irregularity_severe": work["late_90_plus"],
                    "income_obligation_ratio": work["debt_to_income_ratio"],
                    "income_level": work["monthly_income"],
                    "income_unreported": work["income_unreported"],
                    "financial_activity_count": work["active_credit_lines"],
                    "property_loan_count": work["property_loans"],
                    "dependents_count": work["dependents"],
                    "payment_regularity_score": work["payment_regularity_score"],
                    "income_stability": work["income_stability"],
                })[FEATURE_ORDER]

                batch_proba = model.predict_proba(model_input)[:, 1]
                batch_score = ((1 - batch_proba) * 100).round().astype(int)

                results = pd.DataFrame({
                    "applicant_id": range(1, len(work) + 1),
                    "score": batch_score,
                    "default_probability": (batch_proba * 100).round(1),
                })
                results["risk_tier"] = pd.cut(
                    results["score"], bins=[-1, 44, 69, 100], labels=["High Risk", "Moderate Risk", "Low Risk"]
                )

                # --- summary stats ---
                st.markdown('<div class="section-card glass">', unsafe_allow_html=True)
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    st.markdown(f'<div class="afford-label">Applicants scored</div><div class="afford-num">{len(results)}</div>', unsafe_allow_html=True)
                with s2:
                    st.markdown(f'<div class="afford-label">Average score</div><div class="afford-num">{results["score"].mean():.0f}</div>', unsafe_allow_html=True)
                with s3:
                    low_n = int((results["risk_tier"] == "Low Risk").sum())
                    st.markdown(f'<div class="afford-label">Low risk</div><div class="afford-num" style="color:{RISK_LOW};">{low_n}</div>', unsafe_allow_html=True)
                with s4:
                    high_n = int((results["risk_tier"] == "High Risk").sum())
                    st.markdown(f'<div class="afford-label">High risk</div><div class="afford-num" style="color:{RISK_HIGH};">{high_n}</div>', unsafe_allow_html=True)

                mod_n = int((results["risk_tier"] == "Moderate Risk").sum())
                total_n = len(results) or 1
                st.markdown(f"""
                <div class="tier-bar">
                    <div class="tier-seg" style="width:{low_n/total_n*100:.1f}%; background:{RISK_LOW};"></div>
                    <div class="tier-seg" style="width:{mod_n/total_n*100:.1f}%; background:{RISK_MED};"></div>
                    <div class="tier-seg" style="width:{high_n/total_n*100:.1f}%; background:{RISK_HIGH};"></div>
                </div>
                <div class="legend" style="margin-top:0.5rem; margin-bottom:0;">
                    <span><i style="background:{RISK_LOW};"></i>Low ({low_n})</span>
                    <span><i style="background:{RISK_MED};"></i>Moderate ({mod_n})</span>
                    <span><i style="background:{RISK_HIGH};"></i>High ({high_n})</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # --- approval threshold ---
                st.markdown('<div class="section-card glass">', unsafe_allow_html=True)
                st.markdown('<div class="card-title" style="font-size:1.02rem;">Approval threshold</div>', unsafe_allow_html=True)
                st.markdown('<div class="card-desc">Set your own cutoff \u2014 applicants scoring at or above it are marked approved.</div>', unsafe_allow_html=True)
                threshold = st.slider("Minimum score to approve", 0, 100, 60, label_visibility="collapsed")

                results["decision"] = np.where(results["score"] >= threshold, "Approved", "Declined")
                approved_n = int((results["decision"] == "Approved").sum())
                declined_n = total_n - approved_n

                st.markdown(f"""
                <div class="tier-bar">
                    <div class="tier-seg" style="width:{approved_n/total_n*100:.1f}%; background:{LIME};"></div>
                    <div class="tier-seg" style="width:{declined_n/total_n*100:.1f}%; background:{RISK_HIGH};"></div>
                </div>
                <div class="legend" style="margin-top:0.5rem; margin-bottom:0;">
                    <span><i style="background:{LIME};"></i>Approved ({approved_n})</span>
                    <span><i style="background:{RISK_HIGH};"></i>Declined ({declined_n})</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # --- results table ---
                st.markdown('<div class="section-card glass">', unsafe_allow_html=True)
                st.markdown('<div class="card-title" style="font-size:1.02rem;">Results</div>', unsafe_allow_html=True)
                st.dataframe(
                    results[["applicant_id", "score", "default_probability", "risk_tier", "decision"]],
                    use_container_width=True,
                    hide_index=True,
                )
                out_csv = results.to_csv(index=False).encode("utf-8")
                st.download_button("\u2B07 Download scored results", data=out_csv, file_name="creditsight_batch_results.csv", mime="text/csv")
                st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# REGULATORY CONTEXT — static, always visible
# =========================================================
with st.expander("\U0001F4D8 Regulatory context \u2014 RBI Digital Lending framework"):
    st.markdown(f"""
    <div class="reg-note">
    On 8 May 2025, the RBI consolidated its digital lending rules \u2014 replacing the
    earlier 2022 guidelines and 2023 default-loss-guarantee framework \u2014 into a single
    rulebook: the <b style="color:{TEXT};">RBI (Digital Lending) Directions, 2025</b>.
    RBI has continued amending this framework through 2026, including a public DLA
    registry on its CIMS portal, a multi-lender LSP platform framework (effective
    November 2025), and further disclosure and collection-conduct updates reported
    through early-to-mid 2026. A few provisions relevant to a scoring tool like this one:
    </div>
    <ul class="reg-list">
        <li><b>Key Fact Statement (KFS)</b> \u2014 lenders must disclose the full cost of
        credit (APR, fees, penal charges) before a loan is signed, not after.</li>
        <li><b>Direct disbursal only</b> \u2014 loan funds must move straight between the
        regulated lender and the borrower's own account, with no pass-through
        intermediary accounts.</li>
        <li><b>Cooling-off period</b> \u2014 borrowers can exit a loan shortly after
        disbursal by repaying principal plus proportionate interest, with no penalty.</li>
        <li><b>FLDG cap</b> \u2014 default-guarantee arrangements between a lender and its
        sourcing partner are capped at 5% of the loan portfolio.</li>
        <li><b>Data minimisation</b> \u2014 only data necessary for the credit decision may
        be collected, with explicit borrower consent.</li>
        <li><b>Collection conduct</b> \u2014 recovery calls are restricted to 8am\u20137pm, and
        contacting a borrower's family or using social media to share default status
        is prohibited.</li>
    </ul>
    <div class="reg-note" style="margin-top:0.8rem;">
    CreditSight is not a Regulated Entity under this
    framework \u2014 shown here for context on how a real alternative-credit product would
    need to operate. This framework continues to evolve, so for the current
    consolidated text, see rbi.org.in.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# DISCLAIMER + FOOTER
# =========================================================
st.markdown(f"""
<div class="disclaimer glass">
    <b>Disclaimer \u2014</b> CreditSight is a demonstration of alternative credit scoring
    built on reframed, publicly available data. It is not a licensed credit bureau
    product, does not access real financial accounts, and its output should not inform
    actual lending or borrowing decisions. All calculations run locally in this session;
    no input is stored or transmitted.
</div>
""", unsafe_allow_html=True)

st.markdown(f'<div class="footer-note">CreditSight \u2014 Alternative Credit Scoring for the Unbanked</div>', unsafe_allow_html=True)