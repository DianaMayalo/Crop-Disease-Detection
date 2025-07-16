# nlp_model.py

import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# cleaner
basic_stopwords = set([
    'the', 'and', 'is', 'in', 'it', 'of', 'on', 'for', 'with', 'as', 'to', 'are',
    'that', 'this', 'a', 'an', 'at', 'by', 'be', 'from', 'has', 'have', 'but', 'was',
    'or', 'we', 'not', 'can', 'will', 'if', 'all', 'so', 'when', 'what', 'which'
])

def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in basic_stopwords and len(word) > 2]
    return ' '.join(tokens)

# Load data
df = pd.read_csv("cleaned_synthetic_data.csv")
df['recommended_pesticide'] = df['recommended_pesticide'].fillna("None")
df['clean_description'] = df['clean_description'].fillna("").apply(clean_text)

# Train Model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_description'])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['disease'])

model = LogisticRegression(max_iter=500)
model.fit(X, y)

# Pesticide Mapping
pesticide_lookup = df[['disease', 'recommended_pesticide']].drop_duplicates().set_index('disease')['recommended_pesticide'].to_dict()

#  Predict Function 
def predict_disease(description):
    cleaned_desc = clean_text(description)
    desc_vectorized = vectorizer.transform([cleaned_desc])
    pred_index = model.predict(desc_vectorized)[0]
    disease = label_encoder.inverse_transform([pred_index])[0]
    pesticide = pesticide_lookup.get(disease, 'No recommendation available')
    return disease, pesticide
