# Import pandas for data manipulation
import pandas as pd

# Load the original dataset containing social media posts and their sentiments
df = pd.read_csv("data/sentimentdataset.csv")

# Define sentiment categories for classification
# Happy category: includes positive emotions and feelings of wellbeing
happy_list = [
    'Positive', 'Happiness', 'Gratitude', 'Excitement', 'Contentment',
    'Serenity', 'Grateful', 'Accomplished', 'Joy', 'Elation', 'Cheerful'
]

# Neutral category: includes balanced or middle-ground emotional states
neutral_list = [
    'Neutral', 'Ambivalence', 'Indifference', 'Boredom', 'Reflective',
    'Calmness', 'Contemplative', 'Stoic', 'Unmoved', 'Detached'
]

# Depressed category: includes negative emotions and potential signs of depression
depressed_list = [
    'Negative', 'Sadness', 'Devastated', 'Frustrated', 'Melancholy',
    'Loneliness', 'Despair', 'Hopelessness', 'Grief', 'Helplessness',
    'Resentment', 'Regret', 'Disappointment', 'Envy', 'Jealousy',
    'Anxiety', 'Fear', 'Disgust', 'Anger', 'Irritation', 'Overwhelmed'
]

# Function to map detailed sentiments to three main categories
def map_sentiment(sentiment):
    """
    Maps original sentiment labels to simplified 3-class system:
    - Happy: positive emotions
    - Neutral: balanced emotions
    - Depressed: negative emotions
    Returns None for unknown sentiments
    """
    sentiment = sentiment.strip().capitalize()
    if sentiment in happy_list:
        return 'Happy'
    elif sentiment in neutral_list:
        return 'Neutral'
    elif sentiment in depressed_list:
        return 'Depressed'
    else:
        return None  # For unknown sentiment values

# Apply the mapping function to convert detailed sentiments to three classes
df['Sentiment'] = df['Sentiment'].apply(map_sentiment)

# Remove any rows where sentiment mapping failed (unknown categories)
df_cleaned = df[df['Sentiment'].notnull()]

# Save the processed dataset with simplified sentiment categories
df_cleaned.to_csv("data/cleaned_sentiment_dataset.csv", index=False)

# Print summary of the dataset composition
print("✅ Dataset cleaned and saved with 3 classes: Happy, Neutral, Depressed")
print(df_cleaned['Sentiment'].value_counts())