# Sentiment Analysis with VC-Dimension Control
# Streamlit Application (Supports Logistic Regression + SVM)
#new app.py
import streamlit as st
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

# import modularized helpers from the package
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


# Streamlit Setup

st.set_page_config(page_title="Sentiment Analysis (VC-Dimension Controlled)", layout="wide")
st.title("Sentiment Analysis with VC-Dimension Control")



# Load Dataset (from file)
DATA_PATH = "data/sentiment_dataset.csv"

if not os.path.exists(DATA_PATH):
    st.error(f"❌ Dataset not found at: {DATA_PATH}\nPlease add your CSV file.")
    st.stop()

try:
    data, removed = load_and_prepare_data(DATA_PATH, label_mapping={"negative": 0, "positive": 1})
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except ValueError as e:
    st.error(str(e))
    st.stop()





#  Model Settings
st.sidebar.header("Settings")

model_type = st.sidebar.selectbox("Select Model Type", ["Logistic Regression", "SVM (Linear)"])

max_features = st.sidebar.slider(
    "Max Features (controls VC-dimension)",
    min_value=100, max_value=20000, step=100, value=12000
)

C = st.sidebar.slider(
    "Regularization Strength (smaller = stronger)",
    min_value=0.01, max_value=10.0, step=0.01, value=1.0
)

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Model filenames include model type, VC, and C
model_path, vectorizer_path, model_name, vectorizer_name = get_model_filenames(
    model_type, max_features, C, MODEL_DIR
)


model, vectorizer, model_name, vectorizer_name, was_loaded = load_or_train_model(
    data, model_type, max_features, C, MODEL_DIR, dataset_name="local"
)

if was_loaded:
    st.success(f" Loaded saved {model_type} model (VC={max_features}, C={C})")
else:
    st.success(f" Model trained and saved: {model_name}")


# Evaluation

from sklearn.model_selection import train_test_split

X = vectorizer.transform(data["text"])
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# ─── Inject Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
.predict-card {
    background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
    border: 1px solid #4a4a6a;
    border-radius: 16px;
    padding: 24px 28px 10px 28px;
    margin-bottom: 0px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.predict-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e0e0ff;
    margin-bottom: 4px;
}
.predict-subtitle {
    font-size: 0.92rem;
    color: #8888aa;
    margin-bottom: 6px;
}
.result-positive {
    background: linear-gradient(90deg, #0f4c2a, #1a7a44);
    border-left: 4px solid #2ecc71;
    border-radius: 10px;
    padding: 14px 20px;
    color: #afffcf;
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 12px;
}
.result-negative {
    background: linear-gradient(90deg, #4c0f0f, #7a1a1a);
    border-left: 4px solid #e74c3c;
    border-radius: 10px;
    padding: 14px 20px;
    color: #ffcfcf;
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 12px;
}
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #252540 100%);
    border: 1px solid #3a3a5a;
    border-radius: 12px;
    padding: 18px 16px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.2);
}
.metric-label {
    font-size: 0.78rem;
    color: #8888aa;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #a78bfa;
}
.section-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #4a4a6a, transparent);
    margin: 30px 0;
    border: none;
}
.vc-info-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #1e2a3a 100%);
    border: 1px solid #2a4a6a;
    border-radius: 12px;
    padding: 14px 22px;
    color: #a0c4ff;
    font-size: 0.95rem;
    margin-top: 14px;
}
</style>
""", unsafe_allow_html=True)

# ─── Custom Sentiment Prediction (ABOVE Model Performance) ───────────────────

st.markdown('<div class="predict-card">', unsafe_allow_html=True)
st.markdown('<div class="predict-title">🔍 Try Custom Sentiment Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="predict-subtitle">Type any sentence below and the model will classify its sentiment instantly.</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

user_input = st.text_area(
    "Input text",
    "The product is amazing and works perfectly!",
    height=120,
    label_visibility="collapsed",
)

col_btn, _ = st.columns([1, 5])
with col_btn:
    analyze_clicked = st.button("⚡ Analyze Sentiment", use_container_width=True)

if analyze_clicked:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        pred = predict_texts(model, vectorizer, user_input)
        if pred == 1:
            st.markdown(
                '<div class="result-positive">😊 &nbsp;<strong>Positive Sentiment</strong> — The model is confident this is a positive statement.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-negative">😞 &nbsp;<strong>Negative Sentiment</strong> — The model detected a negative tone.</div>',
                unsafe_allow_html=True
            )

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ─── Model Performance ───────────────────────────────────────────────────────

st.markdown("### 📊 Model Performance")
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
for col, label, value in zip(
    [col1, col2, col3, col4],
    ["Accuracy", "F1-Score", "Precision", "Recall"],
    [f"{acc*100:.2f}%", f"{f1:.4f}", f"{precision:.4f}", f"{recall:.4f}"]
):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── VC-Dimension Info ───────────────────────────────────────────────────────

N = X_train.shape[0]
h = max_features + 1
emp_error = 1 - acc
bound = vc_generalization_bound(emp_error, h, N)
train_error = 1.0 - accuracy_score(y_train, model.predict(X_train))
test_error = 1.0 - acc

st.markdown(f"""
<div class="vc-info-box">
    🧮 <strong>Approximate VC-Dimension (h)</strong> = {h} &nbsp;|&nbsp;
    📐 <strong>Generalization Bound:</strong> True error ≤ {bound:.3f}
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)



# compare with a HuggingFace Twitter dataset
st.sidebar.markdown("---")
compare_hf = st.sidebar.checkbox("Compare with HuggingFace Twitter dataset")
hf_metrics = None
hf_name = None
# VC experiment controls
st.sidebar.markdown("---")
vc_feat_text = st.sidebar.text_input("VC feature sizes (comma-separated)", value="10,100,500,1000,2000,5000,10000")
vc_sample = st.sidebar.number_input("Max samples per dataset (0 = all)", min_value=0, value=5000, step=100)
run_vc = st.sidebar.button("Run VC-dimension experiment (may be slow)")
if compare_hf:
    st.sidebar.info("Default dataset: `tweet_eval` subset `sentiment`. Change ID below to another HF dataset if you like.")
    hf_dataset_id = st.sidebar.text_input("HF dataset id", value="tweet_eval")
    hf_subset = st.sidebar.text_input("HF subset (optional)", value="sentiment")

    try:
        hf_name = f"{hf_dataset_id}:{hf_subset}" if hf_subset else hf_dataset_id
        df_hf, removed_hf = load_hf_dataset(hf_dataset_id, hf_subset)
        if removed_hf > 0:
            st.warning(f"HF dataset: removed {removed_hf} rows (neutral/unknown labels)")

        # train/load model for HF dataset (separate cache)
        model_hf, vectorizer_hf, model_name_hf, vec_name_hf, was_loaded_hf = load_or_train_model(
            df_hf, model_type, max_features, C, MODEL_DIR, dataset_name=hf_dataset_id.replace('/', '_')
        )

        # eval HF model
        Xh = vectorizer_hf.transform(df_hf["text"])
        yh = df_hf["label"]
        Xh_train, Xh_test, yh_train, yh_test = train_test_split(Xh, yh, test_size=0.25, random_state=42)
        yh_pred = model_hf.predict(Xh_test)

        acc_h = accuracy_score(yh_test, yh_pred)
        f1_h = f1_score(yh_test, yh_pred)
        prec_h = precision_score(yh_test, yh_pred)
        rec_h = recall_score(yh_test, yh_pred)

        hf_metrics = {"Accuracy": acc_h, "F1": f1_h, "Precision": prec_h, "Recall": rec_h}
        st.subheader(f"HF dataset ({hf_name}) — Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc_h*100:.2f}%")
        c2.metric("F1-score", f"{f1_h:.2f}")
        c3.metric("Precision", f"{prec_h:.2f}")
        c4.metric("Recall", f"{rec_h:.2f}")

        # comparison chart
        local_metrics = {"Accuracy": acc, "F1": f1, "Precision": precision, "Recall": recall}
        comp_fig = None
        try:
            comp_fig = None
            comp_fig = make_vc_figure(0, 0, "")  # placeholder to ensure matplotlib backend ready
        except Exception:
            comp_fig = None

        metrics_compare = {"local": local_metrics, hf_name: hf_metrics}
        from sentiment.plot_utils import plot_metrics_comparison

        fig_comp = plot_metrics_comparison(metrics_compare)
        st.subheader("Metrics comparison (local vs HF)")
        st.pyplot(fig_comp)

        # Train/Test error comparison for local vs HF
        train_error_h = 1.0 - accuracy_score(yh_train, model_hf.predict(Xh_train))
        test_error_h = 1.0 - acc_h

        error_compare = {
            "local": {"Train Error": train_error, "Test Error": test_error},
            hf_name: {"Train Error": train_error_h, "Test Error": test_error_h},
        }
        fig_err = plot_metrics_comparison(error_compare)
        st.subheader("Train/Test error comparison (local vs HF)")
        st.pyplot(fig_err)

        # HF confusion matrix
        cm_hf = confusion_matrix(yh_test, yh_pred)
        st.subheader(f"Confusion Matrix ({hf_name})")
        fig_hf_cm = plot_confusion_matrix(cm_hf)
        st.pyplot(fig_hf_cm)

        # If user requested VC experiment, run for both datasets and plot curves
        if run_vc:
            try:
                from sentiment.plot_utils import plot_vc_accuracy_curves

                # parse feature sizes
                xs = [int(x.strip()) for x in vc_feat_text.split(",") if x.strip().isdigit()]
                if len(xs) == 0:
                    st.warning("No valid feature sizes provided for VC experiment.")
                else:
                    results = {}
                    # local dataset sample
                    local_df = data
                    if vc_sample > 0 and len(local_df) > vc_sample:
                        local_df = local_df.sample(n=int(vc_sample), random_state=42)

                    with st.spinner("Running VC experiment on local dataset..."):
                        t0 = time.time()
                        train_accs = []
                        test_accs = []
                        for mf in xs:
                            vec = TfidfVectorizer(max_features=mf, ngram_range=(1, 2))
                            X_all = vec.fit_transform(local_df["text"])
                            y_all = local_df["label"]
                            Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.25, random_state=42)
                            if model_type == "Logistic Regression":
                                m = LogisticRegression(C=C, solver="saga", max_iter=1000)
                            else:
                                m = LinearSVC(C=C, max_iter=2000)
                            m.fit(Xtr, ytr)
                            train_accs.append(accuracy_score(ytr, m.predict(Xtr)))
                            test_accs.append(accuracy_score(yte, m.predict(Xte)))
                        results["local"] = {"xs": xs, "train": train_accs, "test": test_accs}
                        t1 = time.time()
                    st.success(f"Local VC experiment done in {t1-t0:.1f}s")

                    # HF dataset experiment
                    with st.spinner("Running VC experiment on HF dataset..."):
                        t0 = time.time()
                        hf_df = df_hf
                        if vc_sample > 0 and len(hf_df) > vc_sample:
                            hf_df = hf_df.sample(n=int(vc_sample), random_state=42)
                        train_accs = []
                        test_accs = []
                        for mf in xs:
                            vec = TfidfVectorizer(max_features=mf, ngram_range=(1, 2))
                            X_all = vec.fit_transform(hf_df["text"])
                            y_all = hf_df["label"]
                            Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.25, random_state=42)
                            if model_type == "Logistic Regression":
                                m = LogisticRegression(C=C, solver="saga", max_iter=1000)
                            else:
                                m = LinearSVC(C=C, max_iter=2000)
                            m.fit(Xtr, ytr)
                            train_accs.append(accuracy_score(ytr, m.predict(Xtr)))
                            test_accs.append(accuracy_score(yte, m.predict(Xte)))
                        results[hf_name] = {"xs": xs, "train": train_accs, "test": test_accs}
                        t1 = time.time()
                    st.success(f"HF VC experiment done in {t1-t0:.1f}s")

                    fig_vc = plot_vc_accuracy_curves(results, title="VC-dimension vs Accuracy (Train/Test)")
                    st.subheader("VC-dimension effect (Train vs Test accuracy)")
                    st.pyplot(fig_vc)

            except Exception as e:
                st.error(f"VC experiment failed: {e}")

    except Exception as e:
        st.error(f"Failed to load/train on HF dataset: {e}")


st.caption("Note: Increasing 'Max Features' increases VC-dimension → higher overfitting risk when data is small.")