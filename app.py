# Sentiment Analysis with VC-Dimension Control
# Streamlit Application — Creamy 3D Luxury UI

import streamlit as st
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split

from sentiment.data_utils import load_and_prepare_data
from sentiment.train_utils import load_or_train_model, get_model_filenames
from sentiment.infer_utils import predict_texts
from sentiment.vc_utils import vc_generalization_bound
from sentiment.plot_utils import make_vc_figure, plot_confusion_matrix
from sentiment.hf_utils import load_hf_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import time

st.set_page_config(
    page_title="SentiFlow · VC Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700;900&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ══════════════════════════════════════════════
   ROOT TOKENS
══════════════════════════════════════════════ */
:root {
  --cream:       #faf6f0;
  --cream-dark:  #f2ebe0;
  --cream-deep:  #e8ddd0;
  --warm-white:  #fffdf9;
  --caramel:     #c8956c;
  --caramel-lt:  #e8b48a;
  --caramel-dk:  #9c6840;
  --espresso:    #3d2314;
  --mocha:       #6b4226;
  --latte:       #a07850;
  --sage:        #7a9e7e;
  --sage-lt:     #a8c5ab;
  --rose:        #c97878;
  --rose-lt:     #e8a8a8;
  --slate:       #6e6560;
  --slate-lt:    #9e9590;
  --gold:        #c8a84b;
  --shadow-sm:   0 2px 8px rgba(61,35,20,0.08);
  --shadow-md:   0 8px 32px rgba(61,35,20,0.12);
  --shadow-lg:   0 20px 60px rgba(61,35,20,0.16);
  --shadow-xl:   0 32px 80px rgba(61,35,20,0.2);
  --radius-sm:   10px;
  --radius-md:   16px;
  --radius-lg:   24px;
  --radius-xl:   32px;
}

/* ══════════════════════════════════════════════
   GLOBAL RESET & BASE
══════════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
  font-family: 'DM Sans', sans-serif;
  background: transparent;
  color: var(--espresso);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
  padding-top: 0 !important;
  padding-bottom: 3rem !important;
  max-width: 1140px;
}

/* ══════════════════════════════════════════════
   3D CREAMY BACKGROUND
══════════════════════════════════════════════ */
.stApp {
  background:
    radial-gradient(ellipse 120% 80% at 15% 10%, rgba(232,180,138,0.28) 0%, transparent 55%),
    radial-gradient(ellipse 80% 60% at 85% 5%,  rgba(201,149,108,0.18) 0%, transparent 50%),
    radial-gradient(ellipse 60% 70% at 90% 80%, rgba(122,158,126,0.14) 0%, transparent 50%),
    radial-gradient(ellipse 90% 50% at 5%  85%, rgba(200,168,75,0.12)  0%, transparent 55%),
    radial-gradient(ellipse 70% 90% at 50% 50%, rgba(250,246,240,0.95) 0%, transparent 100%),
    linear-gradient(160deg, #f7f0e6 0%, #faf6f0 35%, #f5ede0 65%, #f0e8d8 100%) !important;
  min-height: 100vh;
}

/* Floating 3D orbs */
.stApp::before {
  content: '';
  position: fixed;
  top: -120px; left: -80px;
  width: 500px; height: 500px;
  background: radial-gradient(circle, rgba(200,149,108,0.22) 0%, rgba(232,180,138,0.08) 50%, transparent 70%);
  border-radius: 50%;
  animation: orb-drift-1 18s ease-in-out infinite;
  pointer-events: none;
  z-index: 0;
}
.stApp::after {
  content: '';
  position: fixed;
  bottom: -100px; right: -60px;
  width: 420px; height: 420px;
  background: radial-gradient(circle, rgba(122,158,126,0.18) 0%, rgba(168,197,171,0.06) 50%, transparent 70%);
  border-radius: 50%;
  animation: orb-drift-2 22s ease-in-out infinite;
  pointer-events: none;
  z-index: 0;
}

@keyframes orb-drift-1 {
  0%,100% { transform: translate(0,0) scale(1); }
  33%      { transform: translate(60px,40px) scale(1.08); }
  66%      { transform: translate(-30px,70px) scale(0.95); }
}
@keyframes orb-drift-2 {
  0%,100% { transform: translate(0,0) scale(1); }
  40%      { transform: translate(-50px,-35px) scale(1.1); }
  70%      { transform: translate(30px,-60px) scale(0.92); }
}

/* ══════════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════════ */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg,
    rgba(242,235,224,0.96) 0%,
    rgba(235,224,208,0.98) 100%) !important;
  border-right: 1px solid rgba(200,149,108,0.25) !important;
  backdrop-filter: blur(20px);
  box-shadow: 4px 0 32px rgba(61,35,20,0.08);
}
[data-testid="stSidebar"] * { color: var(--espresso) !important; }
[data-testid="stSidebar"] h2 {
  font-family: 'Playfair Display', serif !important;
  font-size: 1.1rem !important;
  font-weight: 700 !important;
  color: var(--espresso) !important;
  border-bottom: 2px solid var(--caramel) !important;
  padding-bottom: 10px !important;
  margin-bottom: 18px !important;
  letter-spacing: -0.01em;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stCheckbox label {
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.09em !important;
  color: var(--mocha) !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(200,149,108,0.3) !important; }
[data-testid="stSidebar"] strong { color: var(--espresso) !important; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }

/* Sidebar selectbox/input bg */
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stTextInput > div > div > input,
[data-testid="stSidebar"] .stNumberInput > div > div > input {
  background: rgba(255,253,249,0.8) !important;
  border: 1px solid rgba(200,149,108,0.3) !important;
  border-radius: 8px !important;
}

/* ══════════════════════════════════════════════
   PAGE HERO
══════════════════════════════════════════════ */
.hero {
  position: relative;
  padding: 52px 48px 44px;
  margin-bottom: 36px;
  border-radius: var(--radius-xl);
  background: linear-gradient(135deg,
    rgba(255,253,249,0.85) 0%,
    rgba(242,235,224,0.75) 100%);
  border: 1px solid rgba(200,149,108,0.2);
  box-shadow:
    var(--shadow-lg),
    inset 0 1px 0 rgba(255,255,255,0.9),
    inset 0 -1px 0 rgba(200,149,108,0.1);
  backdrop-filter: blur(16px);
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute;
  top: -2px; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--caramel), var(--gold), var(--caramel-lt), var(--sage));
  border-radius: var(--radius-xl) var(--radius-xl) 0 0;
}
.hero::after {
  content: '✦';
  position: absolute;
  right: 48px; top: 50%;
  transform: translateY(-50%);
  font-size: 7rem;
  color: rgba(200,149,108,0.07);
  line-height: 1;
  font-family: serif;
  animation: spin-slow 30s linear infinite;
}
@keyframes spin-slow { to { transform: translateY(-50%) rotate(360deg); } }

.hero-eyebrow {
  font-family: 'DM Mono', monospace;
  font-size: 0.7rem;
  font-weight: 500;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--caramel);
  margin-bottom: 14px;
  display: flex; align-items: center; gap: 10px;
}
.hero-eyebrow::before {
  content: '';
  display: inline-block;
  width: 28px; height: 2px;
  background: var(--caramel);
  border-radius: 2px;
}
.hero-title {
  font-family: 'Playfair Display', serif;
  font-size: 2.8rem;
  font-weight: 900;
  color: var(--espresso);
  line-height: 1.15;
  margin: 0 0 12px;
  letter-spacing: -0.03em;
}
.hero-title em {
  font-style: italic;
  color: var(--caramel);
}
.hero-desc {
  font-size: 0.94rem;
  color: var(--slate);
  line-height: 1.7;
  max-width: 480px;
  font-weight: 400;
  margin: 0;
}

/* ══════════════════════════════════════════════
   GLASS CARDS
══════════════════════════════════════════════ */
.glass-card {
  background: linear-gradient(145deg,
    rgba(255,253,249,0.88) 0%,
    rgba(245,237,226,0.72) 100%);
  border: 1px solid rgba(200,149,108,0.2);
  border-radius: var(--radius-lg);
  box-shadow:
    var(--shadow-md),
    inset 0 1px 0 rgba(255,255,255,0.95),
    inset 0 -1px 0 rgba(200,149,108,0.08);
  backdrop-filter: blur(12px);
  padding: 24px 26px 20px;
  margin-bottom: 14px;
  transition: box-shadow 0.3s ease, transform 0.3s ease;
}
.glass-card:hover {
  box-shadow: var(--shadow-lg), inset 0 1px 0 rgba(255,255,255,0.95);
  transform: translateY(-1px);
}

/* ══════════════════════════════════════════════
   SECTION LABELS
══════════════════════════════════════════════ */
.sec-label {
  display: flex;
  align-items: center;
  gap: 10px;
  font-family: 'DM Mono', monospace;
  font-size: 0.65rem;
  font-weight: 500;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--caramel-dk);
  margin-bottom: 16px;
}
.sec-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, rgba(200,149,108,0.3), transparent);
}

/* ══════════════════════════════════════════════
   PREDICT CARD
══════════════════════════════════════════════ */
.predict-icon-wrap {
  width: 40px; height: 40px;
  background: linear-gradient(135deg, var(--caramel), var(--caramel-dk));
  border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.1rem;
  box-shadow: 0 4px 12px rgba(200,149,108,0.4);
  flex-shrink: 0;
}
.predict-row { display: flex; align-items: flex-start; gap: 14px; }
.predict-heading {
  font-family: 'Playfair Display', serif;
  font-size: 1.15rem;
  font-weight: 700;
  color: var(--espresso);
  margin: 0 0 3px;
  line-height: 1.2;
}
.predict-sub {
  font-size: 0.82rem;
  color: var(--slate);
  margin: 0;
  font-weight: 400;
}

/* ══════════════════════════════════════════════
   RESULT BANNERS
══════════════════════════════════════════════ */
.result-pos {
  background: linear-gradient(135deg,
    rgba(122,158,126,0.12) 0%,
    rgba(168,197,171,0.06) 100%);
  border: 1px solid rgba(122,158,126,0.35);
  border-left: 3px solid var(--sage);
  border-radius: var(--radius-sm);
  padding: 14px 18px;
  margin-top: 14px;
  display: flex; align-items: flex-start; gap: 12px;
  animation: slide-up 0.35s cubic-bezier(0.34,1.56,0.64,1);
}
.result-neg {
  background: linear-gradient(135deg,
    rgba(201,120,120,0.12) 0%,
    rgba(232,168,168,0.06) 100%);
  border: 1px solid rgba(201,120,120,0.35);
  border-left: 3px solid var(--rose);
  border-radius: var(--radius-sm);
  padding: 14px 18px;
  margin-top: 14px;
  display: flex; align-items: flex-start; gap: 12px;
  animation: slide-up 0.35s cubic-bezier(0.34,1.56,0.64,1);
}
@keyframes slide-up {
  from { opacity:0; transform: translateY(10px) scale(0.97); }
  to   { opacity:1; transform: translateY(0) scale(1); }
}
.result-main { font-family: 'Playfair Display', serif; font-size: 1rem; font-weight: 700; }
.result-sub  { font-size: 0.78rem; font-weight: 400; margin-top: 2px; opacity: 0.85; }
.res-pos-txt { color: #4a7c52; }
.res-neg-txt { color: #8c4040; }

/* ══════════════════════════════════════════════
   METRIC CARDS
══════════════════════════════════════════════ */
.metric-card {
  background: linear-gradient(160deg,
    rgba(255,253,249,0.95) 0%,
    rgba(242,235,224,0.8) 100%);
  border: 1px solid rgba(200,149,108,0.18);
  border-radius: var(--radius-md);
  padding: 22px 18px 18px;
  text-align: center;
  position: relative;
  overflow: hidden;
  box-shadow:
    var(--shadow-sm),
    inset 0 1px 0 rgba(255,255,255,1);
  transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.metric-card:hover {
  transform: translateY(-3px) scale(1.01);
  box-shadow: var(--shadow-md), inset 0 1px 0 rgba(255,255,255,1);
}
.metric-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; height: 3px;
}
.mc-acc::before  { background: linear-gradient(90deg, var(--caramel), var(--gold)); }
.mc-f1::before   { background: linear-gradient(90deg, #a07ba8, #c4a0cc); }
.mc-prec::before { background: linear-gradient(90deg, var(--sage), var(--sage-lt)); }
.mc-rec::before  { background: linear-gradient(90deg, var(--rose), var(--rose-lt)); }
.metric-card::after {
  content: '';
  position: absolute;
  bottom: -30px; right: -20px;
  width: 80px; height: 80px;
  border-radius: 50%;
  opacity: 0.06;
}
.mc-acc::after  { background: var(--caramel); }
.mc-f1::after   { background: #a07ba8; }
.mc-prec::after { background: var(--sage); }
.mc-rec::after  { background: var(--rose); }

.m-label {
  font-family: 'DM Mono', monospace;
  font-size: 0.62rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--latte);
  margin-bottom: 10px;
  font-weight: 500;
}
.m-value {
  font-family: 'Playfair Display', serif;
  font-size: 2rem;
  font-weight: 700;
  color: var(--espresso);
  line-height: 1;
  letter-spacing: -0.03em;
}
.m-sub {
  font-size: 0.67rem;
  color: var(--slate-lt);
  margin-top: 6px;
  font-weight: 400;
}

/* ══════════════════════════════════════════════
   VC INFO BAR
══════════════════════════════════════════════ */
.vc-bar {
  background: linear-gradient(135deg,
    rgba(255,253,249,0.9) 0%,
    rgba(242,235,224,0.75) 100%);
  border: 1px solid rgba(200,149,108,0.2);
  border-radius: var(--radius-md);
  padding: 16px 24px;
  display: flex;
  align-items: center;
  gap: 0;
  margin-top: 18px;
  box-shadow: var(--shadow-sm), inset 0 1px 0 rgba(255,255,255,0.95);
  flex-wrap: wrap;
  overflow: hidden;
  position: relative;
}
.vc-bar::before {
  content: '';
  position: absolute;
  left: 0; top: 0; bottom: 0; width: 3px;
  background: linear-gradient(180deg, var(--caramel), var(--gold));
}
.vc-stat { flex: 1; min-width: 130px; text-align: center; padding: 6px 12px; }
.vc-stat + .vc-stat { border-left: 1px solid rgba(200,149,108,0.18); }
.vc-stat-label {
  font-family: 'DM Mono', monospace;
  font-size: 0.6rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--latte);
  margin-bottom: 5px;
}
.vc-stat-value {
  font-family: 'Playfair Display', serif;
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--espresso);
  letter-spacing: -0.02em;
}

/* ══════════════════════════════════════════════
   DIVIDER
══════════════════════════════════════════════ */
.warm-divider {
  display: flex;
  align-items: center;
  gap: 16px;
  margin: 30px 0;
  color: rgba(200,149,108,0.4);
  font-family: serif;
  font-size: 1rem;
}
.warm-divider::before, .warm-divider::after {
  content: '';
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(200,149,108,0.3), transparent);
}

/* ══════════════════════════════════════════════
   STREAMLIT WIDGET OVERRIDES
══════════════════════════════════════════════ */
.stTextArea textarea {
  background: rgba(255,253,249,0.9) !important;
  border: 1.5px solid rgba(200,149,108,0.3) !important;
  border-radius: 12px !important;
  color: var(--espresso) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.94rem !important;
  padding: 14px 16px !important;
  box-shadow: inset 0 2px 8px rgba(61,35,20,0.04) !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextArea textarea:focus {
  border-color: var(--caramel) !important;
  box-shadow: 0 0 0 3px rgba(200,149,108,0.15), inset 0 2px 8px rgba(61,35,20,0.04) !important;
}
.stTextArea label, .stTextArea > div:first-child { display: none !important; }

/* Main analyze button */
.stButton > button {
  background: linear-gradient(135deg, var(--espresso) 0%, var(--mocha) 100%) !important;
  border: none !important;
  border-radius: 10px !important;
  color: var(--cream) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  padding: 11px 26px !important;
  letter-spacing: 0.02em !important;
  box-shadow: 0 4px 16px rgba(61,35,20,0.25) !important;
  transition: all 0.2s ease !important;
  position: relative; overflow: hidden;
}
.stButton > button::before {
  content: '';
  position: absolute;
  top: 0; left: -100%; width: 100%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
  transition: left 0.4s ease;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 24px rgba(61,35,20,0.3) !important;
}
.stButton > button:hover::before { left: 100%; }

/* Sidebar button */
[data-testid="stSidebar"] .stButton > button {
  background: linear-gradient(135deg, rgba(200,149,108,0.15), rgba(200,149,108,0.08)) !important;
  border: 1px solid rgba(200,149,108,0.35) !important;
  color: var(--espresso) !important;
  box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: linear-gradient(135deg, rgba(200,149,108,0.25), rgba(200,149,108,0.15)) !important;
  transform: none !important;
}

/* Selectbox / inputs inside main area */
.stSelectbox > div > div,
.stTextInput > div > div > input {
  background: rgba(255,253,249,0.9) !important;
  border: 1.5px solid rgba(200,149,108,0.25) !important;
  border-radius: 10px !important;
  color: var(--espresso) !important;
}

/* Streamlit alerts */
.stAlert {
  border-radius: 12px !important;
  border: none !important;
}

/* Footer */
.footer-note {
  text-align: center;
  color: var(--slate-lt);
  font-size: 0.75rem;
  margin-top: 44px;
  padding-top: 18px;
  border-top: 1px solid rgba(200,149,108,0.18);
  font-family: 'DM Mono', monospace;
  letter-spacing: 0.04em;
}
.footer-note strong { color: var(--caramel-dk); }

/* Entrance animation for main content */
.main-content {
  animation: fade-in-up 0.6s cubic-bezier(0.34,1.2,0.64,1) both;
}
@keyframes fade-in-up {
  from { opacity:0; transform: translateY(18px); }
  to   { opacity:1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ─── Page Hero ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero main-content">
  <div class="hero-eyebrow">SentiFlow · VC Analysis</div>
  <h1 class="hero-title">Sentiment Analysis<br>with <em>VC-Dimension</em> Control</h1>
  <p class="hero-desc">Explore how VC-dimension and regularization craft the boundary between learning and guessing. Powered by Logistic Regression & SVM.</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.markdown("## ⚙️ Model Settings")
model_type   = st.sidebar.selectbox("Model Type", ["Logistic Regression", "SVM (Linear)"])
max_features = st.sidebar.slider("Max Features (VC-dimension)", min_value=100, max_value=20000, step=100, value=12000)
C            = st.sidebar.slider("Regularization C", min_value=0.01, max_value=10.0, step=0.01, value=1.0)
st.sidebar.markdown("---")
compare_hf   = st.sidebar.checkbox("🤗 Compare with HuggingFace Dataset")
st.sidebar.markdown("---")
st.sidebar.markdown("**VC Experiment**")
vc_feat_text = st.sidebar.text_input("Feature sizes (comma-separated)", value="10,100,500,1000,2000,5000,10000")
vc_sample    = st.sidebar.number_input("Max samples (0 = all)", min_value=0, value=5000, step=100)
run_vc       = st.sidebar.button("▶ Run VC Experiment")

# ─── Load Data ───────────────────────────────────────────────────────────────

DATA_PATH = "data/sentiment_dataset.csv"
if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found at: {DATA_PATH}")
    st.stop()
try:
    data, removed = load_and_prepare_data(DATA_PATH, label_mapping={"negative": 0, "positive": 1})
except (FileNotFoundError, ValueError) as e:
    st.error(str(e)); st.stop()

# ─── Train / Load ─────────────────────────────────────────────────────────────

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)
get_model_filenames(model_type, max_features, C, MODEL_DIR)
model, vectorizer, model_name, vectorizer_name, was_loaded = load_or_train_model(
    data, model_type, max_features, C, MODEL_DIR, dataset_name="local"
)

# ─── Evaluation ──────────────────────────────────────────────────────────────

X = vectorizer.transform(data["text"])
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
y_pred    = model.predict(X_test)
acc       = accuracy_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)

# ─── Inference Section ───────────────────────────────────────────────────────

st.markdown('<div class="sec-label">✦ Inference</div>', unsafe_allow_html=True)

st.markdown("""
<div class="glass-card">
  <div class="predict-row">
    <div class="predict-icon-wrap">🔍</div>
    <div>
      <p class="predict-heading">Custom Sentiment Prediction</p>
      <p class="predict-sub">Type any sentence — the model will reveal its sentiment.</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

user_input = st.text_area("_", "The product is amazing and works perfectly!", height=108)

col_btn, _ = st.columns([1.6, 5])
with col_btn:
    analyze_clicked = st.button("✦ Analyze Sentiment", use_container_width=True)

if analyze_clicked:
    if user_input.strip() == "":
        st.warning("Please enter some text before analyzing.")
    else:
        pred = predict_texts(model, vectorizer, user_input)
        if pred == 1:
            st.markdown(
                '<div class="result-pos">'
                '<span style="font-size:1.5rem;flex-shrink:0;margin-top:1px">🌿</span>'
                '<div class="res-pos-txt">'
                '<div class="result-main">Positive Sentiment Detected</div>'
                '<div class="result-sub">The model is confident this expresses a warm, positive tone.</div>'
                '</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="result-neg">'
                '<span style="font-size:1.5rem;flex-shrink:0;margin-top:1px">🍂</span>'
                '<div class="res-neg-txt">'
                '<div class="result-main">Negative Sentiment Detected</div>'
                '<div class="result-sub">The model detected a critical or negative undertone.</div>'
                '</div></div>', unsafe_allow_html=True)

st.markdown('<div class="warm-divider">✦</div>', unsafe_allow_html=True)

# ─── Model Performance ────────────────────────────────────────────────────────

st.markdown('<div class="sec-label">✦ Model Performance</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
perf = [
    ("mc-acc",  "Accuracy",  f"{acc*100:.2f}%",  "Overall correct"),
    ("mc-f1",   "F1-Score",  f"{f1:.4f}",         "Harmonic mean"),
    ("mc-prec", "Precision", f"{precision:.4f}",  "TP / predicted +"),
    ("mc-rec",  "Recall",    f"{recall:.4f}",      "TP / actual +"),
]
for col, (cls, label, value, sub) in zip([col1, col2, col3, col4], perf):
    col.markdown(f"""
    <div class="metric-card {cls}">
      <div class="m-label">{label}</div>
      <div class="m-value">{value}</div>
      <div class="m-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

# ─── VC Info ─────────────────────────────────────────────────────────────────

N           = X_train.shape[0]
h           = max_features + 1
emp_error   = 1 - acc
bound       = vc_generalization_bound(emp_error, h, N)
train_error = 1.0 - accuracy_score(y_train, model.predict(X_train))
test_error  = 1.0 - acc

st.markdown(f"""
<div class="vc-bar">
  <div class="vc-stat">
    <div class="vc-stat-label">VC-Dimension (h)</div>
    <div class="vc-stat-value">{h:,}</div>
  </div>
  <div class="vc-stat">
    <div class="vc-stat-label">Training Samples</div>
    <div class="vc-stat-value">{N:,}</div>
  </div>
  <div class="vc-stat">
    <div class="vc-stat-label">Train Error</div>
    <div class="vc-stat-value">{train_error:.4f}</div>
  </div>
  <div class="vc-stat">
    <div class="vc-stat-label">Test Error</div>
    <div class="vc-stat-value">{test_error:.4f}</div>
  </div>
  <div class="vc-stat">
    <div class="vc-stat-label">Generalization Bound</div>
    <div class="vc-stat-value">≤ {bound:.4f}</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="warm-divider">✦</div>', unsafe_allow_html=True)

# ─── HuggingFace Comparison ──────────────────────────────────────────────────

hf_metrics = None
hf_name    = None

if compare_hf:
    st.sidebar.info("Default: `tweet_eval` / `sentiment`.")
    hf_dataset_id = st.sidebar.text_input("HF dataset id", value="tweet_eval")
    hf_subset     = st.sidebar.text_input("HF subset (optional)", value="sentiment")
    try:
        hf_name = f"{hf_dataset_id}:{hf_subset}" if hf_subset else hf_dataset_id
        df_hf, removed_hf = load_hf_dataset(hf_dataset_id, hf_subset)
        if removed_hf > 0:
            st.warning(f"HF dataset: removed {removed_hf} neutral/unknown rows")
        model_hf, vectorizer_hf, _, _, _ = load_or_train_model(
            df_hf, model_type, max_features, C, MODEL_DIR, dataset_name=hf_dataset_id.replace('/', '_')
        )
        Xh = vectorizer_hf.transform(df_hf["text"])
        yh = df_hf["label"]
        Xh_train, Xh_test, yh_train, yh_test = train_test_split(Xh, yh, test_size=0.25, random_state=42)
        yh_pred = model_hf.predict(Xh_test)
        acc_h  = accuracy_score(yh_test, yh_pred)
        f1_h   = f1_score(yh_test, yh_pred)
        prec_h = precision_score(yh_test, yh_pred)
        rec_h  = recall_score(yh_test, yh_pred)
        hf_metrics = {"Accuracy": acc_h, "F1": f1_h, "Precision": prec_h, "Recall": rec_h}

        st.markdown(f'<div class="sec-label">✦ HuggingFace — {hf_name}</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, cls, label, value in zip([c1,c2,c3,c4],
            ["mc-acc","mc-f1","mc-prec","mc-rec"],
            ["Accuracy","F1-Score","Precision","Recall"],
            [f"{acc_h*100:.2f}%",f"{f1_h:.4f}",f"{prec_h:.4f}",f"{rec_h:.4f}"]):
            col.markdown(f"""
            <div class="metric-card {cls}">
              <div class="m-label">{label}</div>
              <div class="m-value">{value}</div>
            </div>""", unsafe_allow_html=True)

        from sentiment.plot_utils import plot_metrics_comparison
        st.markdown('<div class="warm-divider">✦</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">✦ Metrics Comparison — Local vs HF</div>', unsafe_allow_html=True)
        local_metrics = {"Accuracy": acc, "F1": f1, "Precision": precision, "Recall": recall}
        st.pyplot(plot_metrics_comparison({"local": local_metrics, hf_name: hf_metrics}))

        train_error_h = 1.0 - accuracy_score(yh_train, model_hf.predict(Xh_train))
        test_error_h  = 1.0 - acc_h
        st.markdown('<div class="sec-label">✦ Train / Test Error</div>', unsafe_allow_html=True)
        st.pyplot(plot_metrics_comparison({
            "local": {"Train Error": train_error, "Test Error": test_error},
            hf_name: {"Train Error": train_error_h, "Test Error": test_error_h},
        }))

        st.markdown(f'<div class="sec-label">✦ Confusion Matrix — {hf_name}</div>', unsafe_allow_html=True)
        st.pyplot(plot_confusion_matrix(confusion_matrix(yh_test, yh_pred)))

        if run_vc:
            try:
                from sentiment.plot_utils import plot_vc_accuracy_curves
                xs = [int(x.strip()) for x in vc_feat_text.split(",") if x.strip().isdigit()]
                if not xs:
                    st.warning("No valid feature sizes.")
                else:
                    results  = {}
                    local_df = data.sample(n=int(vc_sample), random_state=42) if vc_sample > 0 and len(data) > vc_sample else data
                    with st.spinner("Running VC experiment on local dataset..."):
                        t0=time.time(); tr,te=[],[]
                        for mf in xs:
                            v=TfidfVectorizer(max_features=mf,ngram_range=(1,2))
                            Xa=v.fit_transform(local_df["text"]); ya=local_df["label"]
                            Xtr,Xte,ytr,yte=train_test_split(Xa,ya,test_size=0.25,random_state=42)
                            m=LogisticRegression(C=C,solver="saga",max_iter=1000) if model_type=="Logistic Regression" else LinearSVC(C=C,max_iter=2000)
                            m.fit(Xtr,ytr); tr.append(accuracy_score(ytr,m.predict(Xtr))); te.append(accuracy_score(yte,m.predict(Xte)))
                        results["local"]={"xs":xs,"train":tr,"test":te}
                    st.success(f"Local VC done in {time.time()-t0:.1f}s")
                    hf_df=df_hf.sample(n=int(vc_sample),random_state=42) if vc_sample>0 and len(df_hf)>vc_sample else df_hf
                    with st.spinner("Running VC experiment on HF dataset..."):
                        t0=time.time(); tr,te=[],[]
                        for mf in xs:
                            v=TfidfVectorizer(max_features=mf,ngram_range=(1,2))
                            Xa=v.fit_transform(hf_df["text"]); ya=hf_df["label"]
                            Xtr,Xte,ytr,yte=train_test_split(Xa,ya,test_size=0.25,random_state=42)
                            m=LogisticRegression(C=C,solver="saga",max_iter=1000) if model_type=="Logistic Regression" else LinearSVC(C=C,max_iter=2000)
                            m.fit(Xtr,ytr); tr.append(accuracy_score(ytr,m.predict(Xtr))); te.append(accuracy_score(yte,m.predict(Xte)))
                        results[hf_name]={"xs":xs,"train":tr,"test":te}
                    st.success(f"HF VC done in {time.time()-t0:.1f}s")
                    st.markdown('<div class="sec-label">✦ VC-Dimension Effect</div>', unsafe_allow_html=True)
                    st.pyplot(plot_vc_accuracy_curves(results, title="VC-dimension vs Accuracy (Train/Test)"))
            except Exception as e:
                st.error(f"VC experiment failed: {e}")
    except Exception as e:
        st.error(f"Failed to load/train on HF dataset: {e}")

# ─── Footer ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="footer-note">
  Increasing <strong>Max Features</strong> raises VC-dimension
  — higher overfitting risk when training data is limited.
</div>
""", unsafe_allow_html=True)
