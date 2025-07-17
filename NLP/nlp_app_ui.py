import os, re, string, pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ---------------- Config ----------------
IMG_SIZE = (224, 224)
CONF_THRESHOLD = 0.75
PESTICIDE_PKL = "pesticide_lookup.pkl"
CSV_PATH = "cleaned_synthetic_data.csv"

basic_stopwords = {
    'the','and','is','in','it','of','on','for','with','as','to','are','that','this','a','an','at','by',
    'be','from','has','have','but','was','or','we','not','can','will','if','all','so','when','what','which'
}

def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [w for w in text.split() if w not in basic_stopwords and len(w) > 2]
    return ' '.join(tokens)

@st.cache_resource(show_spinner=False)
def load_cnn(path="model.keras"):
    return tf.keras.models.load_model(path)

@st.cache_resource(show_spinner=False)
def load_nlp(v_path="vectorizer.pkl", m_path="nlp_model.pkl", le_path="label_encoder.pkl"):
    with open(v_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(m_path, "rb") as f:
        nlp_model = pickle.load(f)
    with open(le_path, "rb") as f:
        label_encoder = pickle.load(f)
    return vectorizer, nlp_model, label_encoder

@st.cache_resource(show_spinner=False)
def load_lookup():
    # Try pickle
    if os.path.exists(PESTICIDE_PKL):
        try:
            with open(PESTICIDE_PKL, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    # Fallback CSV
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        dcol = next((c for c in ["disease","Disease","label","class"] if c in df.columns), None)
        pcol = next((c for c in ["recommended_pesticide","pesticide","recommendation"] if c in df.columns), None)
        ccol = next((c for c in ["crop","Crop","plant","host"] if c in df.columns), None)
        if dcol is None:
            return {}
        if pcol is None:
            df["__pest__"] = "No recommendation available"
            pcol = "__pest__"
        lookup = {}
        for _, r in df.iterrows():
            d = str(r[dcol]).strip()
            pest = str(r[pcol]).strip() if pd.notna(r[pcol]) else "No recommendation available"
            crop = str(r[ccol]).strip() if ccol and pd.notna(r[ccol]) else "Unknown crop"
            if d not in lookup:
                lookup[d] = {"pesticide": pest if pest else "No recommendation available", "crop": set()}
            lookup[d]["pesticide"] = pest if pest else "No recommendation available"
            lookup[d]["crop"].add(crop)
        for d in lookup:
            crops = sorted([c for c in lookup[d]["crop"] if c])
            lookup[d]["crop"] = ", ".join(crops) if crops else "Unknown crop"
        return lookup
    return {}

def get_crop_pest(disease, lookup):
    info = lookup.get(disease)
    if not info:
        return "Unknown crop", "No recommendation available"
    crop = info.get("crop", "Unknown crop")
    pest = info.get("pesticide", "No recommendation available")
    if not pest or pest.lower() == "none":
        pest = "No recommendation available"
    return crop, pest

def preprocess_image(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)

def predict_cnn(image, cnn_model, label_encoder):
    arr = preprocess_image(image)
    preds = cnn_model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))
    disease = label_encoder.inverse_transform([idx])[0]
    return disease, conf, preds

def predict_text(text, vectorizer, nlp_model, label_encoder):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    idx = int(nlp_model.predict(vec)[0])
    disease = label_encoder.inverse_transform([idx])[0]
    prob = None
    if hasattr(nlp_model, "predict_proba"):
        prob = float(np.max(nlp_model.predict_proba(vec)))
    return disease, prob

def make_wordcloud(text):
    cleaned = clean_text(text)
    if not cleaned:
        return None
    wc = WordCloud(background_color="white", width=800, height=400).generate(cleaned)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

# ---------------- UI ----------------
st.set_page_config(page_title="Smart Crop Disease Diagnosis", page_icon="🌾")
st.title("🌾 Smart Crop Disease Diagnosis")

cnn_model = load_cnn()
vectorizer, nlp_model, label_encoder = load_nlp()
lookup = load_lookup()

tab_img, tab_txt = st.tabs(["Image Diagnosis", "Symptom Text Diagnosis"])

with tab_img:
    up_img = st.file_uploader("Upload crop image", type=["jpg","jpeg","png"], key="img_up")
    if st.button("Predict from Image"):
        if up_img is None:
            st.warning("Image required.")
        else:
            img = Image.open(up_img)
            st.image(img, caption="Input image", use_column_width=True)
            disease, conf, preds = predict_cnn(img, cnn_model, label_encoder)
            crop, pest = get_crop_pest(disease, lookup)
            st.subheader("Result")
            st.success(f"Disease: {disease}")
            st.info(f"Crop: {crop}")
            st.write(f"Confidence: {conf:.2f}")
            if pest != "No recommendation available":
                st.write(f"Pesticide: {pest}")
            else:
                st.write("No pesticide recommendation available.")
            # Prob table
            classes = list(label_encoder.classes_)
            if len(preds) == len(classes):
                dfp = pd.DataFrame({"Disease": classes, "Probability": preds}).sort_values("Probability", ascending=False)
                st.dataframe(dfp, use_container_width=True)

with tab_txt:
    sx_text = st.text_area("Enter field symptoms / description", height=150)
    if st.button("Predict from Text"):
        if not sx_text.strip():
            st.warning("Symptom description required.")
        else:
            disease, prob = predict_text(sx_text, vectorizer, nlp_model, label_encoder)
            crop, pest = get_crop_pest(disease, lookup)
            st.subheader("Result")
            st.success(f"Disease: {disease}")
            st.info(f"Crop: {crop}")
            if prob is not None:
                st.write(f"Model confidence: {prob:.2f}")
            if pest != "No recommendation available":
                st.write(f"Pesticide: {pest}")
            else:
                st.write("No pesticide recommendation available.")
            st.subheader("Symptom Wordcloud")
            fig = make_wordcloud(sx_text)
            if fig:
                st.pyplot(fig)
            else:
                st.write("Insufficient text for wordcloud.")

st.caption("Prototype educational tool. Agronomist confirmation advised.")
