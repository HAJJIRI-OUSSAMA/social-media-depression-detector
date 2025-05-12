# Configure environment and suppress warnings
import os
import warnings
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Import required modules
# - Preprocessing module for data preparation
# - Scikit-learn models and evaluation metrics
# - Joblib for model persistence
from data_preprocessing import load_and_preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json

# Load and preprocess the dataset
# Returns: feature matrix (X), target labels (y), and fitted TF-IDF vectorizer
X, y, tfidf = load_and_preprocess()

# Split data into training and testing sets
# 80% training, 20% testing with fixed random seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models to train
# - Logistic Regression: Linear classifier
# - Random Forest: Ensemble of decision trees
# - KNN: K-Nearest Neighbors classifier
models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "knn": KNeighborsClassifier(n_neighbors=5)
}

# Dictionary to store model performance results
results = {}

# Train and evaluate each model
print("\n📊 Model Training & Evaluation:")
for name, model in models.items():
    print(f"\nTraining {name}...")
    # Train the model
    model.fit(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_test)
    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"✅ Accuracy ({name}): {acc * 100:.2f}%")

# Identify the best performing model
best_model_name = max(results, key=results.get)
best_model_acc = results[best_model_name]
print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_model_acc * 100:.2f}%)")

# Save trained models and preprocessors for later use
# - Save each trained model
# - Save accuracy results
# - Save TF-IDF vectorizer for text preprocessing
for name, model in models.items():
    joblib.dump(model, f"models/{name}.pkl")

with open("models/model_accuracies.json", "w") as f:
    json.dump(results, f, indent=4)

joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")