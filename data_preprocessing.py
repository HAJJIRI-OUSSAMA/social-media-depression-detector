# Import necessary libraries and configure environment
import os
import warnings
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Download required NLTK resources for text processing and sentiment analysis
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Path to the dataset
DATASET_URL = "data/cleaned_sentiment_dataset.csv"

def clean_text(text):
    """
    Clean and normalize text data:
    1. Convert to lowercase
    2. Remove numbers
    3. Remove punctuation
    """
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def load_and_preprocess():
    """
    Main function to load and preprocess the social media data for depression prediction
    Returns: Features matrix (X), target variable (y), and fitted TF-IDF vectorizer
    """
    # Load the dataset
    df = pd.read_csv(DATASET_URL)

    # Clean the main text content
    df['cleaned_text'] = df['Text'].apply(clean_text)

    def process_hashtags(hashtag_str):
        """
        Process hashtags:
        1. Handle missing values
        2. Remove # symbols
        3. Convert to lowercase
        4. Normalize spacing
        """
        if pd.isna(hashtag_str):
            return ""
        hashtag_str = hashtag_str.replace('#', '').replace(',', ' ').strip().lower()
        return ' '.join(hashtag_str.split())

    # Process hashtags and combine with cleaned text
    df['cleaned_hashtags'] = df['Hashtags'].apply(process_hashtags)
    df['final_text'] = df['cleaned_text'] + ' ' + df['cleaned_hashtags']

    # Calculate sentiment scores using VADER
    # VADER (Valence Aware Dictionary and sEntiment Reasoner) is specifically attuned to social media
    sia = SentimentIntensityAnalyzer()
    df['vader_sentiment'] = df['final_text'].apply(lambda x: sia.polarity_scores(x)['compound'])

    # Convert text to numerical features using TF-IDF
    # TF-IDF measures the importance of words in the document collection
    tfidf = TfidfVectorizer(max_features=500)  # Limit to top 500 most important features
    tfidf_matrix = tfidf.fit_transform(df['final_text'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

    # Combine TF-IDF features with VADER sentiment scores
    X = pd.concat([tfidf_df, df[['vader_sentiment']].reset_index(drop=True)], axis=1)
    y = df['Sentiment']

    return X, y, tfidf