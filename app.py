import streamlit as st
import joblib
import re
import nltk
import os
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO

# ── Download NLTK data silently ──────────────────────────────────────────────
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except:
        nltk.download('stopwords', quiet=True)

    try:
        nltk.data.find('corpora/wordnet')
    except:
        nltk.download('wordnet', quiet=True)

    try:
        nltk.data.find('corpora/omw-1.4')
    except:
        nltk.download('omw-1.4', quiet=True)

download_nltk_data()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Détection de Cyberharcèlement",
    page_icon="🛡️",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stTextArea textarea { font-size: 16px; }
    .result-toxic {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white; padding: 20px; border-radius: 12px;
        text-align: center; font-size: 22px; font-weight: bold;
        margin-top: 20px;
    }
    .result-safe {
        background: linear-gradient(135deg, #55efc4, #00b894);
        color: white; padding: 20px; border-radius: 12px;
        text-align: center; font-size: 22px; font-weight: bold;
        margin-top: 20px;
    }
    .confidence-bar { margin-top: 10px; font-size: 15px; }
    .info-box {
        background: #e8f4fd; border-left: 4px solid #3498db;
        padding: 12px 16px; border-radius: 6px; margin-bottom: 16px;
        font-size: 14px; color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# ── Preprocessing (même logique que le notebook) ──────────────────────────────
stop_words  = set(stopwords.words('english'))
lemmatizer  = WordNetLemmatizer()

def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = text.strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w)
              for w in tokens
              if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

def generate_wordcloud(text: str, colormap: str, title: str):
    """Generate a WordCloud image from text and return as BytesIO."""
    if not text.strip():
        return None
    wc = WordCloud(
        width=600, height=300,
        background_color='white',
        colormap=colormap,
        max_words=80,
        collocations=False
    ).generate(text)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf

# ── Load model & vectorizer ───────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model_path = "model.pkl"
    tfidf_path = "tfidf.pkl"
    if not os.path.exists(model_path) or not os.path.exists(tfidf_path):
        return None, None
    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    return model, tfidf

model, tfidf = load_artifacts()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🛡️ Détection de Cyberharcèlement")
st.markdown("---")

# Upload section (if model not found)
if model is None or tfidf is None:
    st.warning("⚠️ Fichiers `model.pkl` et `tfidf.pkl` introuvables.")
    st.markdown("""
    <div class='info-box'>
    📂 Veuillez uploader vos fichiers exportés depuis Google Colab :
    <b>model.pkl</b> et <b>tfidf.pkl</b>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        model_file = st.file_uploader("📦 model.pkl", type=["pkl"])
    with col2:
        tfidf_file  = st.file_uploader("📦 tfidf.pkl",  type=["pkl"])

    if model_file and tfidf_file:
        with open("model.pkl", "wb") as f:
            f.write(model_file.read())
        with open("tfidf.pkl", "wb") as f:
            f.write(tfidf_file.read())
        st.success("✅ Fichiers chargés ! Rechargez la page.")
        st.stop()
    else:
        st.stop()

# ── Main prediction UI ────────────────────────────────────────────────────────
st.markdown("""
<div class='info-box'>
🔍 Ce système analyse des commentaires en anglais et détecte s'ils contiennent
du <b>cyberharcèlement</b> (insultes, menaces, contenu toxique…).
</div>
""", unsafe_allow_html=True)

# Input
user_input = st.text_area(
    "✍️ Entrez un commentaire à analyser :",
    placeholder="Ex: You are a wonderful person! / I hate you so much...",
    height=140
)

# Examples
st.markdown("**💡 Exemples rapides :**")
col_ex1, col_ex2, col_ex3 = st.columns(3)

examples = {
    "✅ Positif":   "You did a great job! Keep up the amazing work.",
    "⚠️ Toxique":   "You are so stupid and ugly, nobody likes you.",
    "🔴 Très toxique": "I will find you and destroy you, you worthless trash.",
}

selected_example = None
for (label, example), col in zip(examples.items(), [col_ex1, col_ex2, col_ex3]):
    if col.button(label, use_container_width=True):
        selected_example = example

if selected_example:
    user_input = selected_example
    st.text_area("✍️ Commentaire sélectionné :", value=user_input,
                 height=80, key="example_display")

st.markdown("")

# Predict
if st.button("🔍 Analyser", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("⚠️ Veuillez entrer un texte.")
    else:
        with st.spinner("Analyse en cours..."):
            cleaned    = preprocess(user_input)
            vectorized = tfidf.transform([cleaned])
            prediction = model.predict(vectorized)[0]

            # Confidence (decision_function for LinearSVC, predict_proba for others)
            try:
                proba       = model.predict_proba(vectorized)[0]
                conf_toxic  = round(proba[1] * 100, 1)
                conf_safe   = round(proba[0] * 100, 1)
                show_conf   = True
            except AttributeError:
                # LinearSVC → decision_function
                score      = model.decision_function(vectorized)[0]
                conf_toxic = round(float(1 / (1 + np.exp(-score))) * 100, 1)
                conf_safe  = round(100 - conf_toxic, 1)
                show_conf  = True

        # Result display
        if prediction == 1:
            st.markdown(f"""
            <div class='result-toxic'>
                🚨 CYBERHARCÈLEMENT DÉTECTÉ<br>
                <small style='font-size:15px'>Confiance : {conf_toxic}%</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-safe'>
                ✅ COMMENTAIRE SAIN<br>
                <small style='font-size:15px'>Confiance : {conf_safe}%</small>
            </div>
            """, unsafe_allow_html=True)

        # Details
        with st.expander("🔎 Détails de l'analyse"):
            st.markdown(f"**Texte original :** {user_input}")
            st.markdown(f"**Après prétraitement :** `{cleaned if cleaned else '(vide après nettoyage)'}`")
            if show_conf:
                st.markdown("**Probabilités :**")
                st.progress(conf_safe / 100,   text=f"✅ Non toxique : {conf_safe}%")
                st.progress(conf_toxic / 100,  text=f"🚨 Toxique     : {conf_toxic}%")

        # ── WordCloud real-time ───────────────────────────────────────────
        if cleaned:
            st.markdown("### ☁️ WordCloud du commentaire")
            colormap = 'Reds' if prediction == 1 else 'Greens'
            wc_title = (
                "🔴 Mots détectés — Commentaire Toxique"
                if prediction == 1 else
                "🟢 Mots détectés — Commentaire Sain"
            )
            wc_buf = generate_wordcloud(cleaned, colormap, wc_title)
            if wc_buf:
                st.image(wc_buf, use_container_width=True)
            else:
                st.info("Texte trop court pour générer un WordCloud.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:13px;'>"
    "🛡️ Projet de Fin de Module — Détection de Cyberharcèlement | "
    "Logistic Regression + TF-IDF</p>",
    unsafe_allow_html=True
)
