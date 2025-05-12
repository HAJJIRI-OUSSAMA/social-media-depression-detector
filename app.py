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
import snscrape.modules.twitter as sntwitter
import re
from urllib.parse import urlparse
import tweepy
import os
from datetime import datetime

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Cache model loading to improve app performance
@st.cache_resource
def load_models():
    """
    Load trained models and TF-IDF vectorizer
    Returns: Dictionary of models and vectorizer
    """
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
    """
    Load model accuracy scores from JSON file
    Returns: Dictionary of model accuracies
    """
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
    """
    Clean and normalize input text
    Returns: Cleaned text string
    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def process_hashtags(hashtag_str):
    """
    Process hashtags to match training data format
    Returns: Cleaned hashtag string
    """
    if not isinstance(hashtag_str, str):
        return ""
    return hashtag_str.replace('#', '').replace(',', ' ').strip().lower()

def setup_twitter_api():
    """
    Setup Twitter API client with credentials
    Returns: Tweepy Client object
    """
    client = tweepy.Client(
        bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
        wait_on_rate_limit=True
    )
    return client

def extract_tweet_from_url(url):
    """
    Extract tweet content using Twitter API v2
    Returns: tuple (tweet_text, hashtags)
    """
    try:
        # Extract tweet ID from URL
        tweet_id = url.split('/')[-1].split('?')[0]
        
        # Initialize Twitter client
        client = setup_twitter_api()
        
        # Fetch tweet
        tweet = client.get_tweet(
            tweet_id,
            tweet_fields=['entities', 'text']
        )
        
        if not tweet.data:
            raise ValueError("Tweet not found")
            
        # Extract text and hashtags
        text = tweet.data.text
        hashtags = ''
        if 'entities' in tweet.data and 'hashtags' in tweet.data.entities:
            hashtags = ' '.join([f"#{tag['text']}" for tag in tweet.data.entities['hashtags']])
            
        return text, hashtags
        
    except Exception as e:
        st.warning(f"""
        Tweet fetching failed: {str(e)}
        
        Please copy-paste the tweet content manually in the fields below.
        
        To use the Twitter API feature:
        1. Get your Twitter API bearer token from https://developer.twitter.com
        2. Set it as an environment variable: TWITTER_BEARER_TOKEN
        """)
        return None, None

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

# User input form
tweet_url = st.text_input("🔗 Enter Twitter/X post URL (optional):")

if tweet_url:
    if "twitter.com" in tweet_url or "x.com" in tweet_url:
        with st.spinner("Fetching tweet..."):
            tweet_text, tweet_hashtags = extract_tweet_from_url(tweet_url)
            if tweet_text:
                post_text = tweet_text
                hashtags = tweet_hashtags
                st.success("Tweet fetched successfully!")
            else:
                st.info("You can manually enter the tweet content below.")
    else:
        st.error("Please enter a valid Twitter/X post URL")

# Add a note about manual input
st.markdown("""
> **Note**: If tweet fetching fails, you can manually copy-paste the content from Twitter/X.
""")

# Keep existing text inputs as fallback
post_text = st.text_area("📝 Enter the post text:", value=post_text if 'post_text' in locals() else "")
hashtags = st.text_input("🏷️ Enter hashtags:", value=hashtags if 'hashtags' in locals() else "")

submitted = st.button("🔍 Predict")

# Prediction pipeline
if submitted and post_text.strip() != "":
    model = models_dict[selected_model_name]

    # Progress tracking
    steps = [
        "Step 1/5: Loading selected model...",
        "Step 2/5: Cleaning text...",
        "Step 3/5: Processing hashtags...",
        "Step 4/5: Calculating sentiment features...",
        "Step 5/5: Making prediction..."
    ]
    step_placeholders = [st.empty() for _ in steps]

    # Execute prediction steps
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
            # Calculate features
            vader_score = sia.polarity_scores(final_input_text)['compound']
            tfidf_vec = tfidf.transform([final_input_text])
            input_data = pd.DataFrame({'vader_sentiment': [vader_score]})
            final_input = pd.concat([
                pd.DataFrame(tfidf_vec.toarray(), columns=tfidf.get_feature_names_out()),
                input_data.reset_index(drop=True)
            ], axis=1)

        elif i == 4:
            # Make prediction
            prediction = model.predict(final_input)[0]
            probabilities = model.predict_proba(final_input)[0] if hasattr(model, "predict_proba") else []

    # Display results
    step_placeholders[-1].success(f"🧠 **Predicted Mental State:** {prediction}")

    # Show confidence scores if available
    if len(probabilities) > 0:
        confidence_df = pd.DataFrame({
            "Class": model.classes_,
            "Confidence (%)": probabilities * 100
        })
        st.bar_chart(confidence_df.set_index("Class").sort_values(by="Confidence (%)", ascending=False))

    # Display input summary
    st.markdown("### 🧾 Summary of Input:")
    st.write(f"**Text:** {cleaned_text}")
    st.write(f"**Hashtags:** {cleaned_hashtags}")