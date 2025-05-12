import streamlit as st
import joblib
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import string
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Initialize tools
sia = SentimentIntensityAnalyzer()

@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
        "Random Forest": joblib.load("models/random_forest.pkl"),
        "KNN": joblib.load("models/knn.pkl")
    }
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    return models, tfidf

@st.cache_resource
def load_accuracies():
    try:
        with open("models/model_accuracies.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "logistic_regression": "N/A",
            "random_forest": "N/A",
            "knn": "N/A"
        }

models_dict, tfidf = load_models()
model_accuracy = load_accuracies()

# Mapping for JSON keys
selected_key_map = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "KNN": "knn"
}

# Clean text & hashtags
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def process_hashtags(hashtag_str):
    if not isinstance(hashtag_str, str):
        return ""
    return hashtag_str.replace('#', '').replace(',', ' ').strip().lower()

# UI
st.title("🧠 Depression Prediction from Social Media Posts")
st.markdown("Select a model and enter a social media post to predict emotional state.")

# Model selector
selected_model_name = st.selectbox(
    "Choose a Machine Learning Model:",
    ("Logistic Regression", "Random Forest", "KNN")
)

# Show accuracy
selected_key = selected_key_map[selected_model_name]
acc = model_accuracy.get(selected_key, "N/A")
if acc != "N/A":
    try:
        acc_percent = float(acc) * 100
        st.write(f"📊 **{selected_model_name} Accuracy:** {acc_percent:.2f}%")
    except Exception:
        st.write(f"📊 **{selected_model_name} Accuracy:** {acc}")
else:
    st.write(f"📊 **{selected_model_name} Accuracy:** N/A")

# Input form
post_text = st.text_area("📝 Enter the post text:")
hashtags = st.text_input("🏷️ Enter hashtags:")

submitted = st.button("🔍 Predict")

if submitted and post_text.strip() != "":
    model = models_dict[selected_model_name]

    # Track steps
    steps = [
        "Step 1/5: Loading selected model...",
        "Step 2/5: Cleaning text...",
        "Step 3/5: Processing hashtags...",
        "Step 4/5: Calculating sentiment features...",
        "Step 5/5: Making prediction..."
    ]
    step_placeholders = [st.empty() for _ in steps]

    for i, step in enumerate(steps):
        step_placeholders[i].info(step)

        if i == 0:
            pass  # Already loaded

        elif i == 1:
            cleaned_text = clean_text(post_text)

        elif i == 2:
            cleaned_hashtags = process_hashtags(hashtags)
            final_input_text = cleaned_text + ' ' + cleaned_hashtags

        elif i == 3:
            vader_score = sia.polarity_scores(final_input_text)['compound']
            tfidf_vec = tfidf.transform([final_input_text])
            input_data = pd.DataFrame({'vader_sentiment': [vader_score]})
            final_input = pd.concat([
                pd.DataFrame(tfidf_vec.toarray(), columns=tfidf.get_feature_names_out()),
                input_data.reset_index(drop=True)
            ], axis=1)

        elif i == 4:
            prediction = model.predict(final_input)[0]
            probabilities = model.predict_proba(final_input)[0] if hasattr(model, "predict_proba") else []

    # Final result
    step_placeholders[-1].success(f"🧠 **Predicted Mental State:** {prediction}")

    # Confidence scores
    if len(probabilities) > 0:
        confidence_df = pd.DataFrame({
            "Class": model.classes_,
            "Confidence (%)": probabilities * 100
        })
        st.bar_chart(confidence_df.set_index("Class").sort_values(by="Confidence (%)", ascending=False))


    # Summary of input
    st.markdown("### 🧾 Summary of Input:")
    st.write(f"**Text:** {cleaned_text}")
    st.write(f"**Hashtags:** {cleaned_hashtags}")