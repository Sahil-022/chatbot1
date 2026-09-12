"""
train_model.py
---------------
Trains an intent-classification model for the chatbot using:
 - TF-IDF vectorization (feature extraction from text)
 - Logistic Regression / SVM (Machine Learning classifier)

Run this script once (or whenever intents.json is updated) to (re)generate
the model artifacts inside the `model/` directory:
    model/vectorizer.pkl
    model/classifier.pkl
    model/labels.pkl
"""

import json
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from nlp_utils import preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)


def load_training_data():
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts, labels = [], []
    for intent in data["intents"]:
        tag = intent["tag"]
        for pattern in intent.get("patterns", []):
            texts.append(preprocess(pattern))
            labels.append(tag)

    return texts, labels


def train():
    print("Loading training data from intents.json ...")
    texts, labels = load_training_data()
    print(f"Loaded {len(texts)} training examples across {len(set(labels))} intents.")

    # Feature extraction
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    y = labels

    # Train/test split for a quick sanity check on accuracy
    if len(set(labels)) > 1 and len(texts) > 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, y_train = X, y
        X_test, y_test = X, y

    print("Training Logistic Regression classifier ...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nValidation Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, preds, zero_division=0))

    # Refit on ALL data for the final deployed model (small dataset -> use everything)
    clf_final = LogisticRegression(max_iter=1000)
    clf_final.fit(X, y)

    # Persist artifacts
    with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODEL_DIR, "classifier.pkl"), "wb") as f:
        pickle.dump(clf_final, f)
    with open(os.path.join(MODEL_DIR, "labels.pkl"), "wb") as f:
        pickle.dump(sorted(set(labels)), f)

    print(f"Model artifacts saved to: {MODEL_DIR}")
    print("Training complete! You can now run the app with: python app.py")


if __name__ == "__main__":
    train()
