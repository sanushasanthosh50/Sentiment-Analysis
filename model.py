
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

print("Loading IMDb dataset...")

# Load dataset
data = pd.read_csv("dataset/IMDB Dataset.csv")

print("Dataset loaded successfully!")
print("Total Reviews:", len(data))

# Features and Labels
X = data["review"]
y = data["sentiment"]

# Convert labels to match app.py
y = y.str.capitalize()

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training model...")

# Convert text to numerical features
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

print("Testing model...")

# Predict
predictions = model.predict(X_test_vectorized)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Save trained files
pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel trained and saved successfully!")
print("Generated:")
print("- sentiment_model.pkl")
print("- vectorizer.pkl")

