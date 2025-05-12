import os
import warnings

# Suppress joblib warning about core count
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

# Set CPU count manually to avoid triggering the check
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # You can change 4 to your actual core count

# Import required libraries for model loading, text processing, and sentiment analysis
import joblib
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import string

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Let user select which trained model to use for prediction
model_choice = input("Choose model (logistic_regression, random_forest, knn): ")
# Load the selected model and the TF-IDF vectorizer from saved files
model = joblib.load(f"models/{model_choice}.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# Get input from user for prediction
text = input("Enter post text: ")
hashtags = input("Enter hashtags (e.g., #sad #life): ")

# Text cleaning function - matches the preprocessing used in training
def clean_text(text):
    """
    Clean and normalize input text:
    1. Convert to lowercase
    2. Remove numbers
    3. Remove punctuation
    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

cleaned_text = clean_text(text)

# Hashtag processing function - matches the preprocessing used in training
def process_hashtags(hashtag_str):
    """
    Process hashtags:
    1. Remove # symbols
    2. Replace commas with spaces
    3. Convert to lowercase
    4. Normalize spacing
    """
    hashtag_str = hashtag_str.replace('#', '').replace(',', ' ').strip().lower()
    return ' '.join(hashtag_str.split())

cleaned_hashtags = process_hashtags(hashtags)

# Combine cleaned text and hashtags into single text input
final_input_text = cleaned_text + ' ' + cleaned_hashtags

# Calculate VADER sentiment score for the input
vader_score = sia.polarity_scores(final_input_text)['compound']

# Transform text to TF-IDF features using the same vectorizer used in training
tfidf_vec = tfidf.transform([final_input_text])
tfidf_df = pd.DataFrame(tfidf_vec.toarray(), columns=tfidf.get_feature_names_out())

# Create DataFrame with sentiment score
input_data = pd.DataFrame({
    'vader_sentiment': [vader_score]
})

# Combine TF-IDF features with sentiment score
final_input = pd.concat([tfidf_df, input_data], axis=1)

# Make prediction using the selected model
prediction = model.predict(final_input)[0]
print(f"\n🧠 Predicted Mental State: {prediction}")