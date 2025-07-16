import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nlp_model import predict_disease  


# Page config 

st.set_page_config(page_title="🌿 Crop Disease Recommender", layout="centered")


# Load data for word clouds

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_synthetic_data.csv")

df = load_data()


# Word Cloud Explorer

st.sidebar.header("📊 Explore Word Clouds")
diseases = df['disease'].dropna().unique()
selected_disease = st.sidebar.selectbox("Choose a Disease", sorted(diseases))

if st.sidebar.button("Generate Word Cloud"):
    text_blob = ' '.join(df[df['disease'] == selected_disease]['clean_description'].dropna().astype(str))
    if text_blob.strip():
        wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text_blob)
        st.sidebar.image(wc.to_array(), caption=f"Word Cloud: {selected_disease}", use_column_width=True)
    else:
        st.sidebar.warning("No text available for that disease.")

st.sidebar.markdown("---")


# NLP Prediction

st.title("🌾 NLP-Based Crop Disease Recommender")
st.markdown("Describe crop symptoms below to receive a predicted disease and pesticide recommendation.")

user_input = st.text_area("✏️ Enter crop symptom description", height=150)

if st.button("🔍 Diagnose"):
    if user_input.strip() == "":
        st.warning("Enter a symptom description.")
    else:
        disease, pesticide = predict_disease(user_input)
        st.success(f"🦠 **Predicted Disease:** {disease}")
        st.info(f"🧪 **Recommended Pesticide:** {pesticide}")


# Footer

st.markdown("---")
st.markdown("Source code available on [GitHub](https://github.com/DianaMayalo/Crop-Disease-Detection)")
