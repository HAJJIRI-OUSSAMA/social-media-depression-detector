import logging
logging.basicConfig(level=logging.ERROR)  
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
import tweepy
from tweepy import TweepyException
import time
import os
from datetime import datetime
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Access the token
TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')

# Add this at the start of your app
if not os.getenv('TWITTER_BEARER_TOKEN'):
    st.error("Twitter Bearer Token not found. Please check your .env file.")
    logger.error("Missing Twitter Bearer Token")

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Cache model loading to improve app performance
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
        "Random Forest": joblib.load("models/random_forest.pkl"),
        "KNN": joblib.load("models/knn.pkl")
    }
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    return models, tfidf

# Cache accuracy metrics loading
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

# Load models and accuracies
models_dict, tfidf = load_models()
model_accuracy = load_accuracies()

# Map UI model names to JSON keys
selected_key_map = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "KNN": "knn"
}

# Text preprocessing functions
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def process_hashtags(hashtag_str):
    if not isinstance(hashtag_str, str):
        return ""
    return hashtag_str.replace('#', '').replace(',', ' ').strip().lower()

def extract_tweet(url):
    try:
        # Extract tweet ID from URL
        tweet_id = url.split('/')[-1].split('?')[0]

        client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN, wait_on_rate_limit=True)

        tweet = client.get_tweet(id=tweet_id, tweet_fields=['text'])
        if tweet and tweet.data:
            return tweet.data.text
        return None

    except TweepyException as e:
        st.error(f"Twitter API Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# UI Components
st.title("🧠 Depression Prediction from Social Media Posts")
st.markdown("Select a model and enter a social media post to predict emotional state.")

# Model selection dropdown
selected_model_name = st.selectbox(
    "Choose a Machine Learning Model:",
    ("Logistic Regression", "Random Forest", "KNN")
)

# Display model accuracy
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

# User Input Options
input_option = st.radio("Choose input method:", ("Enter Twitter Post URL", "Manually Enter Post"))

if input_option == "Enter Twitter Post URL":
    tweet_url = st.text_input("🔗 Enter Twitter/X post URL:")

    if tweet_url and ("twitter.com" in tweet_url or "x.com" in tweet_url):
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(100):
            progress_bar.progress(i + 1)
            status_text.text(f"Fetching tweet... ({i+1}%)")
            time.sleep(0.01)

        progress_bar.empty()
        status_text.empty()

        tweet_text = extract_tweet(tweet_url)
        if tweet_text:
            st.success("Tweet fetched successfully!")
            st.write("**Tweet content:**")
            st.write(tweet_text)

            # Run prediction automatically
            post_text = tweet_text
            hashtags = ""

            # Proceed to prediction
            model = models_dict[selected_model_name]
            cleaned_text = clean_text(post_text)
            cleaned_hashtags = process_hashtags(hashtags)
            final_input_text = cleaned_text + ' ' + cleaned_hashtags

            vader_score = sia.polarity_scores(final_input_text)['compound']
            tfidf_vec = tfidf.transform([final_input_text])
            input_data = pd.DataFrame({'vader_sentiment': [vader_score]})
            final_input = pd.concat([
                pd.DataFrame(tfidf_vec.toarray(), columns=tfidf.get_feature_names_out()),
                input_data.reset_index(drop=True)
            ], axis=1)

            prediction = model.predict(final_input)[0]
            probabilities = model.predict_proba(final_input)[0] if hasattr(model, "predict_proba") else []

            st.success(f"🧠 **Predicted Mental State:** {prediction}")

            if len(probabilities) > 0:
                confidence_df = pd.DataFrame({
                    "Class": model.classes_,
                    "Confidence (%)": probabilities * 100
                })
                st.bar_chart(confidence_df.set_index("Class").sort_values(by="Confidence (%)", ascending=False))

            st.markdown("### 🧾 Summary of Input:")
            st.write(f"**Text:** {cleaned_text}")
            st.write(f"**Hashtags:** {cleaned_hashtags}")
        else:
            st.warning("Could not fetch tweet. Try again later or switch to manual entry.")

elif input_option == "Manually Enter Post":
    post_text = st.text_area("📝 Post text:")
    hashtags = st.text_input("🏷️ Hashtags (optional):")

    submitted = st.button("🔍 Predict")

    if submitted and post_text.strip() != "":
        model = models_dict[selected_model_name]

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
                pass  # Model already loaded

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

        step_placeholders[-1].success(f"🧠 **Predicted Mental State:** {prediction}")

        if len(probabilities) > 0:
            confidence_df = pd.DataFrame({
                "Class": model.classes_,
                "Confidence (%)": probabilities * 100
            })
            st.bar_chart(confidence_df.set_index("Class").sort_values(by="Confidence (%)", ascending=False))

        st.markdown("### 🧾 Summary of Input:")
        st.write(f"**Text:** {cleaned_text}")
        st.write(f"**Hashtags:** {cleaned_hashtags}")

# Debug option
if st.checkbox("Debug Twitter API Connection"):
    try:
        client = tweepy.Client(bearer_token=os.getenv('TWITTER_BEARER_TOKEN'))
        st.success("✅ Twitter API connection successful")
    except Exception as e:
        st.error(f"❌ Twitter API connection failed: {str(e)}")