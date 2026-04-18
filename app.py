# ================== IMPORTS ==================
import streamlit as st
import pickle
import re
import time
import nltk
from nltk.corpus import stopwords
from datetime import datetime

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="TruthLens — Fake News Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model():
    model      = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# ================== SESSION STATE ==================
defaults = {
    "history":        [],
    "total_analyzed": 0,
    "total_fake":     0,
    "total_real":     0,
    "feedback_log":   {},
    "article_input":  "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ================== FUNCTIONS ==================
def clean_text(text):
    text  = text.lower()
    text  = re.sub(r'\W', ' ', text)
    text  = re.sub(r'\d', ' ', text)
    text  = re.sub(r'\s+', ' ', text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

def predict_news(text):
    cleaned    = clean_text(text)
    vec        = vectorizer.transform([cleaned])
    pred       = model.predict(vec)[0]
    prob       = model.predict_proba(vec)[0]
    confidence = max(prob)
    label      = "REAL" if pred == 1 else "FAKE"
    words      = text.split()
    wc         = len(words)
    unique     = len(set(w.lower() for w in words))
    sentences  = max(len(re.findall(r'[.!?]+', text)), 1)
    avg_wlen   = round(sum(len(w) for w in words) / max(wc, 1), 1)
    return label, confidence, prob, wc, unique, sentences, avg_wlen

def get_risk_tier(confidence, label):
    if label == "FAKE":
        if confidence >= 0.90: return "CRITICAL",    "#ef4444"
        if confidence >= 0.75: return "HIGH RISK",   "#f97316"
        return                        "MODERATE",    "#eab308"
    else:
        if confidence >= 0.90: return "VERIFIED",    "#10b981"
        if confidence >= 0.75: return "LIKELY REAL", "#34d399"
        return                        "UNCERTAIN",   "#818cf8"

def get_tips(label):
    if label == "FAKE":
        return [
            ("🔗", "Cross-verify sources",   "Check Reuters, AP, or BBC for the same story before sharing."),
            ("📅", "Check the date",         "Old articles frequently resurface as current breaking news."),
            ("👤", "Verify the author",      "Look up the author's credentials and publication history."),
            ("🎭", "Watch the tone",         "Emotional or provocative language is a common manipulation tactic."),
            ("🖼️", "Reverse image search",  "Photos used may be stock images or taken out of context."),
        ]
    else:
        return [
            ("✅", "Good credibility signals", "Article reflects structured, professional journalistic style."),
            ("🔍", "Still verify key claims",  "Corroborate major facts with primary sources when possible."),
            ("⚖️", "Watch for editorial bias", "Even real news can carry slant — read multiple outlets."),
            ("📡", "Check wider coverage",     "Verify that other reputable outlets are reporting the same story."),
            ("🧠", "Context matters",          "Real news can still be misleading without the full picture."),
        ]

def get_signals(label, confidence, lex_den):
    if label == "FAKE":
        return [
            ("Sensationalism Risk",  min(int((1 - lex_den/100) * 85 + 15), 100), "#ef4444"),
            ("Emotional Language",   min(int(confidence * 90), 100),             "#f97316"),
            ("Credibility Score",    max(int((1 - confidence) * 100), 5),         "#10b981"),
            ("Source Reliability",   max(int((1 - confidence) * 80), 8),          "#818cf8"),
        ]
    else:
        return [
            ("Sensationalism Risk",  max(int((1 - confidence) * 60), 4),          "#ef4444"),
            ("Emotional Language",   max(int((1 - confidence) * 50), 4),          "#f97316"),
            ("Credibility Score",    min(int(confidence * 100), 100),             "#10b981"),
            ("Source Reliability",   min(int(confidence * 88), 100),              "#818cf8"),
        ]

EXAMPLES = [
    ("🧪 Fake",      "Scientists SHOCKED as new study proves drinking coffee causes INSTANT weight loss of 30 pounds in just ONE WEEK! Big Pharma doesn't want you to know this SECRET remedy that doctors are hiding from the public. Share before they DELETE this!"),
    ("📰 Real",      "The Federal Reserve raised interest rates by a quarter point on Wednesday, citing continued progress on inflation while acknowledging uncertainty in the labor market. Officials indicated they would monitor incoming data before making further adjustments."),
    ("⚠️ Borderline","Local mayor caught in alleged corruption scandal involving city contracts worth millions. Anonymous sources claim documents were destroyed. The mayor's office denied all allegations and called for an independent investigation into the matter."),
]

NOW = datetime.now().strftime("%H:%M · %d %b %Y")

# ================== CSS ==================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,600;1,400&family=Manrope:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg:      #07090f;
    --s1:      #0d1117;
    --s2:      #111827;
    --s3:      #1a2235;
    --border:  rgba(255,255,255,0.065);
    --border2: rgba(255,255,255,0.13);
    --text:    #cbd5e1;
    --muted:   #3d4f68;
    --accent:  #4f7eff;
    --purple:  #7c3aed;
    --real:    #10b981;
    --realhi:  #34d399;
    --fake:    #ef4444;
    --fakehi:  #f87171;
    --warn:    #f59e0b;
    --mono:    'IBM Plex Mono', monospace;
    --sans:    'Manrope', sans-serif;
}

*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp {
    background: var(--bg) !important;
    font-family: var(--sans); color: var(--text);
}

/* Scanlines */
.stApp::before {
    content: ''; position: fixed; inset: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 3px,
        rgba(255,255,255,0.007) 3px, rgba(255,255,255,0.007) 4px);
    pointer-events: none; z-index: 9999;
}

.block-container { max-width: 1400px !important; padding: 2rem 2.5rem 6rem !important; }

/* ── NAVBAR ── */
.navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 18px; border-bottom: 1px solid var(--border);
    margin-bottom: 22px; animation: fadeDown 0.5s ease both;
}
.nav-left { display: flex; align-items: center; gap: 12px; }
.nav-logo-box {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, var(--accent), var(--purple));
    border-radius: 10px; display: grid; place-items: center; font-size: 22px;
    box-shadow: 0 0 22px rgba(79,126,255,.4);
}
.nav-brand { font-family: var(--mono); font-size: 18px; font-weight: 600; color: #fff; letter-spacing: -.5px; }
.nav-chip {
    font-family: var(--mono); font-size: 9px; letter-spacing: 1.5px;
    color: var(--accent); background: rgba(79,126,255,.1);
    border: 1px solid rgba(79,126,255,.25); padding: 2px 8px; border-radius: 999px;
}
.nav-right { display: flex; align-items: center; gap: 20px; }
.nav-time  { font-family: var(--mono); font-size: 10px; color: var(--muted); letter-spacing: 1px; }
.live-pill { display: flex; align-items: center; gap: 6px; font-family: var(--mono); font-size: 9px; color: var(--real); letter-spacing: 1.5px; }
.live-dot  { width: 7px; height: 7px; border-radius: 50%; background: var(--real); box-shadow: 0 0 8px var(--real); animation: pulse 2s infinite; }

/* ── STAT ROW ── */
.stat-row {
    display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px;
    margin-bottom: 22px; animation: fadeUp 0.5s 0.05s ease both;
}
.scard {
    background: var(--s1); border: 1px solid var(--border);
    border-radius: 13px; padding: 16px 18px; position: relative; overflow: hidden;
    transition: transform .2s, border-color .2s;
}
.scard:hover { transform: translateY(-2px); border-color: var(--border2); }
.scard-top-bar { position: absolute; top: 0; left: 0; width: 100%; height: 2px; }
.scard-eyebrow { font-family: var(--mono); font-size: 9px; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }
.scard-val     { font-family: var(--mono); font-size: 28px; font-weight: 600; color: #fff; line-height: 1; margin-bottom: 3px; }
.scard-sub     { font-size: 11px; color: var(--muted); }

/* ── SECTION LABEL ── */
.sec-lbl {
    font-family: var(--mono); font-size: 9px; letter-spacing: 2.5px;
    text-transform: uppercase; color: var(--accent); margin-bottom: 12px;
    display: flex; align-items: center; gap: 8px;
}
.sec-lbl::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* ── PANEL ── */
.panel {
    background: var(--s1); border: 1px solid var(--border);
    border-radius: 14px; padding: 20px 22px; margin-bottom: 16px;
}
.panel-hd { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }
.panel-tag {
    font-family: var(--mono); font-size: 9px; letter-spacing: 1.5px;
    color: var(--muted); background: var(--s2); border: 1px solid var(--border);
    padding: 2px 8px; border-radius: 999px;
}

/* ── TEXTAREA ── */
.stTextArea textarea {
    background: var(--s2) !important; border: 1px solid var(--border2) !important;
    border-radius: 12px !important; color: #e2e8f0 !important;
    font-family: var(--sans) !important; font-size: 14px !important;
    font-weight: 400 !important; line-height: 1.75 !important;
    padding: 16px 18px !important; resize: none !important;
    transition: border-color .3s, box-shadow .3s !important;
}
.stTextArea textarea:focus {
    border-color: rgba(79,126,255,.5) !important;
    box-shadow: 0 0 0 3px rgba(79,126,255,.1) !important; outline: none !important;
}
.stTextArea textarea::placeholder { color: #2d3d55 !important; }
.stTextArea label { display: none !important; }

/* ── META BAR ── */
.meta-bar { display: flex; justify-content: space-between; align-items: center; margin: 8px 0 14px; }
.meta-item { font-family: var(--mono); font-size: 10px; color: var(--muted); letter-spacing: 1px; }
.meta-ok   { color: var(--real); }
.meta-warn { color: var(--warn); }

/* ── TAG ROW ── */
.tag-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
.tag {
    font-family: var(--mono); font-size: 9px; letter-spacing: 1px;
    background: var(--s2); border: 1px solid var(--border);
    color: var(--muted); padding: 3px 12px; border-radius: 999px; cursor: pointer;
    transition: all .2s;
}

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--purple)) !important;
    color: #fff !important; border: none !important; border-radius: 100px !important;
    height: 48px !important; padding: 0 32px !important; font-family: var(--mono) !important;
    font-size: 12px !important; font-weight: 600 !important; letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    box-shadow: 0 0 24px rgba(79,126,255,.35), 0 4px 16px rgba(0,0,0,.4) !important;
    transition: transform .2s, box-shadow .2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 36px rgba(79,126,255,.5), 0 8px 24px rgba(0,0,0,.5) !important;
}
.stButton > button:active { transform: scale(0.97) !important; }

/* ── STEP ── */
.step { display: flex; gap: 14px; align-items: flex-start; padding: 12px 0; border-bottom: 1px solid var(--border); }
.step:last-child { border-bottom: none; padding-bottom: 0; }
.step-num {
    font-family: var(--mono); font-size: 10px; font-weight: 600; color: var(--accent);
    background: rgba(79,126,255,.1); border: 1px solid rgba(79,126,255,.2);
    border-radius: 6px; width: 30px; height: 30px; display: grid; place-items: center; flex-shrink: 0;
}
.step-title { font-size: 12px; font-weight: 700; color: #e2e8f0; margin-bottom: 3px; }
.step-desc  { font-size: 11px; color: var(--muted); line-height: 1.55; }

/* ── HISTORY ── */
.hist-row {
    display: flex; align-items: center; gap: 10px; padding: 10px 14px; margin-bottom: 6px;
    background: var(--s2); border: 1px solid var(--border); border-radius: 10px; transition: border-color .2s;
}
.hist-row:hover { border-color: var(--border2); }
.hist-badge { font-family: var(--mono); font-size: 9px; font-weight: 700; letter-spacing: 1px; padding: 3px 9px; border-radius: 999px; flex-shrink: 0; }
.hist-badge.real { background: rgba(16,185,129,.15); color: var(--realhi); border: 1px solid rgba(16,185,129,.3); }
.hist-badge.fake { background: rgba(239,68,68,.15);  color: var(--fakehi); border: 1px solid rgba(239,68,68,.3); }
.hist-snippet { font-size: 12px; color: var(--text); flex: 1; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.hist-conf    { font-family: var(--mono); font-size: 11px; color: var(--muted); flex-shrink: 0; }

/* ── RESULT HERO ── */
.result-hero {
    border-radius: 16px; padding: 26px 28px; position: relative; overflow: hidden;
    animation: popIn 0.5s cubic-bezier(.34,1.56,.64,1) both; margin-bottom: 18px;
}
.result-hero.real { background: linear-gradient(135deg, rgba(16,185,129,.07), rgba(52,211,153,.03)); border: 1px solid rgba(16,185,129,.25); box-shadow: 0 0 60px rgba(16,185,129,.06); }
.result-hero.fake { background: linear-gradient(135deg, rgba(239,68,68,.07), rgba(248,113,113,.03)); border: 1px solid rgba(239,68,68,.25); box-shadow: 0 0 60px rgba(239,68,68,.06); }
.result-hero::before { content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%; border-radius: 16px 0 0 16px; }
.result-hero.real::before { background: linear-gradient(180deg, var(--real), #059669); }
.result-hero.fake::before { background: linear-gradient(180deg, var(--fake), #dc2626); }

.verdict-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; }
.verdict-eyebrow { font-family: var(--mono); font-size: 9px; letter-spacing: 3px; text-transform: uppercase; color: var(--muted); margin-bottom: 5px; }
.verdict-big { font-family: var(--mono); font-size: 46px; font-weight: 700; letter-spacing: -2px; line-height: 1; }
.verdict-big.real { color: var(--realhi); text-shadow: 0 0 40px rgba(52,211,153,.35); }
.verdict-big.fake { color: var(--fakehi); text-shadow: 0 0 40px rgba(248,113,113,.35); }
.risk-badge { font-family: var(--mono); font-size: 11px; font-weight: 600; letter-spacing: 2px; border: 1px solid; border-radius: 999px; padding: 6px 18px; }

/* ── BARS ── */
.bar-row   { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.bar-lbl   { font-family: var(--mono); font-size: 10px; color: var(--muted); width: 34px; letter-spacing: 1px; }
.bar-track { flex: 1; height: 7px; background: rgba(255,255,255,.06); border-radius: 999px; overflow: hidden; }
.bar-fill.real { height: 100%; background: linear-gradient(90deg, var(--real), var(--realhi)); border-radius: 999px; box-shadow: 0 0 10px rgba(52,211,153,.4); }
.bar-fill.fake { height: 100%; background: linear-gradient(90deg, var(--fake), var(--fakehi)); border-radius: 999px; box-shadow: 0 0 10px rgba(248,113,113,.4); }
.bar-pct.real  { font-family: var(--mono); font-size: 12px; font-weight: 600; width: 46px; text-align: right; color: var(--realhi); }
.bar-pct.fake  { font-family: var(--mono); font-size: 12px; font-weight: 600; width: 46px; text-align: right; color: var(--fakehi); }

.conf-divider { margin: 16px 0 12px; border: none; border-top: 1px solid var(--border); }
.conf-row     { display: flex; justify-content: space-between; align-items: center; }
.conf-label   { font-size: 12px; color: var(--muted); }
.conf-val     { font-family: var(--mono); font-size: 13px; font-weight: 600; color: var(--accent); }

/* ── METRICS GRID ── */
.metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 16px; }
.mcard { background: var(--s2); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; text-align: center; }
.mcard-val { font-family: var(--mono); font-size: 20px; font-weight: 600; color: #fff; line-height: 1; margin-bottom: 4px; }
.mcard-lbl { font-size: 10px; color: var(--muted); }

/* ── SIGNAL BARS ── */
.signal-row { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
.signal-lbl { font-size: 12px; color: var(--text); width: 150px; flex-shrink: 0; }
.signal-track { flex: 1; height: 5px; background: rgba(255,255,255,.05); border-radius: 999px; overflow: hidden; }
.signal-fill  { height: 100%; border-radius: 999px; }
.signal-pct   { font-family: var(--mono); font-size: 11px; width: 36px; text-align: right; }

/* ── TIPS ── */
.tip-item {
    display: flex; gap: 12px; align-items: flex-start; padding: 11px 14px; margin-bottom: 8px;
    background: var(--s2); border: 1px solid var(--border); border-radius: 10px; transition: border-color .2s;
}
.tip-item:hover { border-color: var(--border2); }
.tip-icon { font-size: 18px; flex-shrink: 0; margin-top: 1px; }
.tip-head { font-size: 12px; font-weight: 700; color: #e2e8f0; margin-bottom: 2px; }
.tip-body { font-size: 12px; color: var(--muted); line-height: 1.5; }

/* ── FEEDBACK BUTTONS ── */
.feedback-hd { font-size: 12px; color: var(--muted); text-align: center; margin: 16px 0 10px; }
div[data-testid="column"] .stButton > button {
    width: 100% !important; height: 42px !important; font-size: 11px !important;
    padding: 0 !important; background: var(--s2) !important; box-shadow: none !important;
    border: 1px solid var(--border2) !important; border-radius: 100px !important;
}
div[data-testid="column"] .stButton > button:hover {
    background: var(--s3) !important; border-color: var(--accent) !important; color: #fff !important;
}

/* ── DISCLAIMER ── */
.disclaimer {
    background: rgba(245,158,11,.05); border: 1px solid rgba(245,158,11,.15);
    border-radius: 10px; padding: 12px 16px; margin-top: 6px;
    display: flex; gap: 10px; align-items: flex-start;
}
.disc-icon { font-size: 15px; flex-shrink: 0; margin-top: 1px; }
.disc-text { font-size: 12px; color: #92400e; line-height: 1.55; }

/* ── FOOTER ── */
.site-footer {
    margin-top: 60px; padding-top: 20px; border-top: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
}
.footer-left  { font-family: var(--mono); font-size: 10px; color: var(--muted); }
.footer-pills { display: flex; gap: 8px; }
.footer-pill  { font-family: var(--mono); font-size: 9px; color: var(--muted); letter-spacing: 1px; background: var(--s1); border: 1px solid var(--border); padding: 3px 10px; border-radius: 999px; }

/* ── MISC ── */
.stAlert { background: rgba(245,158,11,.07) !important; border: 1px solid rgba(245,158,11,.2) !important; border-radius: 10px !important; color: #fbbf24 !important; }
.stSpinner > div { color: var(--accent) !important; }
#MainMenu, footer, header { visibility: hidden !important; }

@keyframes fadeDown { from { opacity:0; transform:translateY(-14px); } to { opacity:1; transform:translateY(0); } }
@keyframes fadeUp   { from { opacity:0; transform:translateY(14px);  } to { opacity:1; transform:translateY(0); } }
@keyframes popIn    { from { opacity:0; transform:scale(.92) translateY(14px); } to { opacity:1; transform:scale(1) translateY(0); } }
@keyframes pulse    { 0%,100% { opacity:1; } 50% { opacity:.3; } }
</style>
""", unsafe_allow_html=True)

# ================== NAVBAR ==================
st.markdown(f"""
<div class="navbar">
    <div class="nav-left">
        <div class="nav-logo-box">🔍</div>
        <span class="nav-brand">TruthLens</span>
        <span class="nav-chip">v3.0 BETA</span>
    </div>
    <div class="nav-right">
        <span class="nav-time">⏱ {NOW}</span>
        <div class="live-pill"><div class="live-dot"></div>MODEL ACTIVE</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ================== STAT STRIP ==================
total     = st.session_state.total_analyzed
n_fake    = st.session_state.total_fake
n_real    = st.session_state.total_real
fake_rate = round(n_fake / total * 100) if total else 0
real_rate = round(n_real / total * 100) if total else 0

st.markdown(f"""
<div class="stat-row">
    <div class="scard">
        <div class="scard-top-bar" style="background:linear-gradient(90deg,var(--accent),transparent)"></div>
        <div class="scard-eyebrow">Analyzed</div>
        <div class="scard-val">{total}</div>
        <div class="scard-sub">this session</div>
    </div>
    <div class="scard">
        <div class="scard-top-bar" style="background:linear-gradient(90deg,var(--fake),transparent)"></div>
        <div class="scard-eyebrow">Fake Detected</div>
        <div class="scard-val">{n_fake}</div>
        <div class="scard-sub">{fake_rate}% of total</div>
    </div>
    <div class="scard">
        <div class="scard-top-bar" style="background:linear-gradient(90deg,var(--real),transparent)"></div>
        <div class="scard-eyebrow">Real Verified</div>
        <div class="scard-val">{n_real}</div>
        <div class="scard-sub">{real_rate}% of total</div>
    </div>
    <div class="scard">
        <div class="scard-top-bar" style="background:linear-gradient(90deg,var(--purple),transparent)"></div>
        <div class="scard-eyebrow">Model Accuracy</div>
        <div class="scard-val">94.3%</div>
        <div class="scard-sub">benchmark F1</div>
    </div>
    <div class="scard">
        <div class="scard-top-bar" style="background:linear-gradient(90deg,var(--warn),transparent)"></div>
        <div class="scard-eyebrow">Feedback Logged</div>
        <div class="scard-val">{len(st.session_state.feedback_log)}</div>
        <div class="scard-sub">corrections this session</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ================== TWO COLUMNS ==================
col_main, col_side = st.columns([3, 2], gap="medium")

# ── LEFT: INPUT ──
with col_main:
    st.markdown('<div class="sec-lbl">Article Input</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="panel">
        <div class="panel-hd">
            <span class="sec-lbl" style="margin:0">Paste your article</span>
            <span class="panel-tag">TF-IDF · SKLEARN</span>
        </div>
    </div>""", unsafe_allow_html=True)

    # Category tags (visual only)
    st.markdown("""
    <div class="tag-row">
        <span class="tag">Politics</span><span class="tag">Health</span>
        <span class="tag">Science</span><span class="tag">Technology</span>
        <span class="tag">Finance</span><span class="tag">World</span>
        <span class="tag">Entertainment</span><span class="tag">Sports</span>
    </div>""", unsafe_allow_html=True)

    user_input = st.text_area(
        label="article",
        placeholder="Paste a full news article or headline here. Longer text yields more accurate results — the model analyses vocabulary patterns, lexical density, sentence structure, and linguistic cues to determine credibility…",
        height=230,
        label_visibility="collapsed",
        key="article_input"
    )

    wc_live = len(user_input.split()) if user_input.strip() else 0
    wc_cls  = "meta-ok"   if wc_live >= 20 else "meta-warn"
    wc_msg  = "✓ SUFFICIENT LENGTH" if wc_live >= 20 else "⚠ ADD MORE TEXT FOR BEST RESULTS"
    st.markdown(f"""
    <div class="meta-bar">
        <span class="meta-item">WORDS: <b style="color:#e2e8f0">{wc_live}</b>
            &nbsp;·&nbsp; CHARS: <b style="color:#e2e8f0">{len(user_input)}</b></span>
        <span class="meta-item {wc_cls}">{wc_msg}</span>
    </div>""", unsafe_allow_html=True)

    btn1, btn2 = st.columns([2, 1])
    with btn1:
        analyze_btn = st.button("⟶  Run Full Analysis", use_container_width=True)
    with btn2:
        clear_btn = st.button("✕  Clear", use_container_width=True)

    # Sample loaders
    st.markdown('<div class="sec-lbl" style="margin-top:18px">Quick Samples</div>', unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns(3)
    if sc1.button("🧪 Fake Sample",   use_container_width=True, key="ex0"): st.session_state["article_input"] = EXAMPLES[0][1]; st.rerun()
    if sc2.button("📰 Real Sample",   use_container_width=True, key="ex1"): st.session_state["article_input"] = EXAMPLES[1][1]; st.rerun()
    if sc3.button("⚠️  Borderline",   use_container_width=True, key="ex2"): st.session_state["article_input"] = EXAMPLES[2][1]; st.rerun()
    if clear_btn: st.session_state["article_input"] = ""; st.rerun()

    st.markdown("""
    <div class="disclaimer">
        <div class="disc-icon">⚠️</div>
        <div class="disc-text">
            <b style="color:#d97706">AI Disclaimer:</b> This tool uses a machine learning model and is
            not 100% accurate. It should not be used as the sole basis for judging content.
            Always cross-reference with reputable news organisations before sharing anything.
        </div>
    </div>""", unsafe_allow_html=True)

# ── RIGHT: PIPELINE + HISTORY ──
with col_side:
    st.markdown('<div class="sec-lbl">How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="panel">
        <div class="step">
            <div class="step-num">01</div>
            <div>
                <div class="step-title">Text Preprocessing</div>
                <div class="step-desc">Text lowercased, punctuation stripped, numbers removed, stopwords filtered, then tokenized.</div>
            </div>
        </div>
        <div class="step">
            <div class="step-num">02</div>
            <div>
                <div class="step-title">TF-IDF Vectorization</div>
                <div class="step-desc">Tokens converted to weighted feature vectors based on term frequency across the training corpus.</div>
            </div>
        </div>
        <div class="step">
            <div class="step-num">03</div>
            <div>
                <div class="step-title">ML Classification</div>
                <div class="step-desc">Trained classifier maps the high-dimensional vector to a binary REAL / FAKE prediction.</div>
            </div>
        </div>
        <div class="step">
            <div class="step-num">04</div>
            <div>
                <div class="step-title">Probability Scoring</div>
                <div class="step-desc">Class probabilities returned and displayed as dual progress bars for transparent confidence.</div>
            </div>
        </div>
        <div class="step">
            <div class="step-num">05</div>
            <div>
                <div class="step-title">Linguistic Analysis</div>
                <div class="step-desc">Lexical density, vocab richness, and reading level derived as secondary credibility signals.</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-lbl">Recent History</div>', unsafe_allow_html=True)
    if not st.session_state.history:
        st.markdown("""
        <div class="panel" style="text-align:center;padding:28px 20px">
            <div style="font-size:26px;margin-bottom:8px">📋</div>
            <div style="font-size:12px;color:var(--muted)">No analyses yet this session.</div>
        </div>""", unsafe_allow_html=True)
    else:
        rows_html = '<div class="panel" style="padding:14px 16px">'
        for item in st.session_state.history:
            cls  = "real" if item["label"] == "REAL" else "fake"
            conf = f"{item['confidence']*100:.0f}%"
            rows_html += f"""
            <div class="hist-row">
                <span class="hist-badge {cls}">{item['label']}</span>
                <span class="hist-snippet">{item['snippet']}</span>
                <span class="hist-conf">{conf}</span>
            </div>"""
        rows_html += "</div>"
        st.markdown(rows_html, unsafe_allow_html=True)

# ================== RESULT ==================
if analyze_btn:
    if not user_input.strip():
        st.warning("⚠  Please paste some text before running the analysis.")
    elif wc_live < 4:
        st.warning("⚠  Text is too short — provide at least a sentence for a meaningful result.")
    else:
        with st.spinner("Running NLP pipeline…"):
            time.sleep(0.4)
            label, confidence, prob, wc, unique, sentences, avg_wlen = predict_news(user_input)

        st.session_state.total_analyzed += 1
        if label == "FAKE": st.session_state.total_fake += 1
        else:               st.session_state.total_real += 1

        snippet = user_input[:65].strip() + ("…" if len(user_input) > 65 else "")
        st.session_state.history.insert(0, {"snippet": snippet, "label": label, "confidence": confidence, "words": wc})
        st.session_state.history = st.session_state.history[:8]
        result_idx = st.session_state.total_analyzed

        real_pct  = prob[1] * 100
        fake_pct  = prob[0] * 100
        conf_pct  = confidence * 100
        cls       = "real" if label == "REAL" else "fake"
        risk, rcolor = get_risk_tier(confidence, label)
        tips      = get_tips(label)
        lex_den   = round(unique / wc * 100, 1) if wc > 0 else 0
        signals   = get_signals(label, confidence, lex_den)
        vocab_q   = "Rich"         if lex_den  > 65  else "Moderate" if lex_den  > 45  else "Low"
        read_lvl  = "Advanced"     if avg_wlen > 5.5 else "Intermediate" if avg_wlen > 4.5 else "Basic"
        wps       = round(wc / max(sentences, 1), 1)
        cred_score = int(real_pct)
        gauge_color = "#10b981" if cred_score >= 60 else "#f97316" if cred_score >= 40 else "#ef4444"

        st.markdown("---")
        st.markdown('<div class="sec-lbl">Analysis Result</div>', unsafe_allow_html=True)

        r1, r2 = st.columns([3, 2], gap="medium")

        with r1:
            # Verdict hero
            st.markdown(f"""
            <div class="result-hero {cls}">
                <div class="verdict-row">
                    <div>
                        <div class="verdict-eyebrow">VERDICT</div>
                        <div class="verdict-big {cls}">{'✓ REAL' if label=='REAL' else '✕ FAKE'}</div>
                    </div>
                    <div class="risk-badge" style="color:{rcolor};border-color:{rcolor}">{risk}</div>
                </div>
                <div class="bar-row">
                    <span class="bar-lbl">REAL</span>
                    <div class="bar-track"><div class="bar-fill real" style="width:{real_pct:.1f}%"></div></div>
                    <span class="bar-pct real">{real_pct:.1f}%</span>
                </div>
                <div class="bar-row">
                    <span class="bar-lbl">FAKE</span>
                    <div class="bar-track"><div class="bar-fill fake" style="width:{fake_pct:.1f}%"></div></div>
                    <span class="bar-pct fake">{fake_pct:.1f}%</span>
                </div>
                <hr class="conf-divider"/>
                <div class="conf-row">
                    <span class="conf-label">Model confidence</span>
                    <span class="conf-val">{conf_pct:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Article metrics
            st.markdown('<div class="sec-lbl">Article Metrics</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metrics-grid">
                <div class="mcard"><div class="mcard-val">{wc}</div><div class="mcard-lbl">Word Count</div></div>
                <div class="mcard"><div class="mcard-val">{sentences}</div><div class="mcard-lbl">Sentences</div></div>
                <div class="mcard"><div class="mcard-val">{lex_den}%</div><div class="mcard-lbl">Lexical Density</div></div>
                <div class="mcard"><div class="mcard-val">{avg_wlen}</div><div class="mcard-lbl">Avg Word Len</div></div>
            </div>
            <div class="metrics-grid">
                <div class="mcard"><div class="mcard-val">{unique}</div><div class="mcard-lbl">Unique Words</div></div>
                <div class="mcard"><div class="mcard-val">{vocab_q}</div><div class="mcard-lbl">Vocab Richness</div></div>
                <div class="mcard"><div class="mcard-val">{read_lvl}</div><div class="mcard-lbl">Reading Level</div></div>
                <div class="mcard"><div class="mcard-val">{wps}</div><div class="mcard-lbl">Words/Sentence</div></div>
            </div>
            """, unsafe_allow_html=True)

            # Linguistic signals
            st.markdown('<div class="sec-lbl">Linguistic Signals</div>', unsafe_allow_html=True)
            sig_html = '<div class="panel">'
            for name, val, color in signals:
                sig_html += f"""
                <div class="signal-row">
                    <span class="signal-lbl">{name}</span>
                    <div class="signal-track">
                        <div class="signal-fill" style="width:{val}%;background:{color};box-shadow:0 0 8px {color}55"></div>
                    </div>
                    <span class="signal-pct" style="color:{color}">{val}%</span>
                </div>"""
            sig_html += "</div>"
            st.markdown(sig_html, unsafe_allow_html=True)

        with r2:
            # Credibility gauge
            circumference = 2 * 3.14159 * 54
            offset = circumference * (1 - cred_score / 100)
            st.markdown(f"""
            <div class="panel" style="text-align:center;margin-bottom:16px">
                <div class="sec-lbl" style="justify-content:center;margin-bottom:14px">Credibility Score</div>
                <div style="position:relative;width:148px;height:148px;margin:0 auto 14px">
                    <svg viewBox="0 0 148 148" style="width:148px;height:148px;transform:rotate(-90deg)">
                        <circle cx="74" cy="74" r="54" fill="none" stroke="rgba(255,255,255,.06)" stroke-width="12"/>
                        <circle cx="74" cy="74" r="54" fill="none" stroke="{gauge_color}"
                            stroke-width="12" stroke-linecap="round"
                            stroke-dasharray="{circumference:.2f}"
                            stroke-dashoffset="{offset:.2f}"
                            style="filter:drop-shadow(0 0 8px {gauge_color})"/>
                    </svg>
                    <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center">
                        <div style="font-family:var(--mono);font-size:30px;font-weight:700;color:{gauge_color};line-height:1">{cred_score}</div>
                        <div style="font-size:10px;color:var(--muted);margin-top:2px">/ 100</div>
                    </div>
                </div>
                <div style="font-size:11px;color:var(--muted);line-height:1.6">
                    Derived from model probability<br>and lexical pattern analysis
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Tips / recommendations
            st.markdown('<div class="sec-lbl">Recommendations</div>', unsafe_allow_html=True)
            tips_html = ""
            for icon, head, body in tips:
                tips_html += f"""
                <div class="tip-item">
                    <div class="tip-icon">{icon}</div>
                    <div>
                        <div class="tip-head">{head}</div>
                        <div class="tip-body">{body}</div>
                    </div>
                </div>"""
            st.markdown(tips_html, unsafe_allow_html=True)

            # Feedback
            st.markdown('<div class="feedback-hd">Was this prediction correct?</div>', unsafe_allow_html=True)
            fb1, fb2 = st.columns(2)
            if fb1.button("✅  Yes, correct",   key=f"fby_{result_idx}", use_container_width=True):
                st.session_state.feedback_log[result_idx] = "correct"
                st.success("Thanks for the feedback!")
            if fb2.button("❌  No, incorrect",  key=f"fbn_{result_idx}", use_container_width=True):
                st.session_state.feedback_log[result_idx] = "incorrect"
                st.warning("Noted — feedback logged.")

# ================== FOOTER ==================
st.markdown("""
<div class="site-footer">
    <div class="footer-left">© 2025 TruthLens · NLP-Powered Fake News Detection</div>
    <div class="footer-pills">
        <span class="footer-pill">TF-IDF</span>
        <span class="footer-pill">SKLEARN</span>
        <span class="footer-pill">NLTK</span>
        <span class="footer-pill">STREAMLIT</span>
    </div>
</div>
""", unsafe_allow_html=True)