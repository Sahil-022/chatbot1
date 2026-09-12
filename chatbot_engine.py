"""
chatbot_engine.py
------------------
Core conversational engine for the AI Chatbot Assistant.

Responsibilities:
 - Load the trained ML model (TF-IDF + Logistic Regression) and intents data
 - Preprocess incoming user queries using NLP techniques
 - Recognize user intent with a confidence score
 - Manage simple conversational flow / context
 - Generate a relevant, real-time response
"""

import json
import os
import pickle
import random
from datetime import datetime

from nlp_utils import preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")
MODEL_DIR = os.path.join(BASE_DIR, "model")

CONFIDENCE_THRESHOLD = 0.30  # Below this, fall back to a clarifying response


class ChatbotEngine:
    def __init__(self):
        self.intents = self._load_intents()
        self.responses_map = {
            intent["tag"]: intent["responses"] for intent in self.intents["intents"]
        }
        self.vectorizer, self.classifier, self.labels = self._load_model()

        # Very small in-memory context per session (for conversational flow)
        self.sessions = {}

    # ---------- Loading ----------
    def _load_intents(self):
        with open(INTENTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_model(self):
        vec_path = os.path.join(MODEL_DIR, "vectorizer.pkl")
        clf_path = os.path.join(MODEL_DIR, "classifier.pkl")
        labels_path = os.path.join(MODEL_DIR, "labels.pkl")

        if not (os.path.exists(vec_path) and os.path.exists(clf_path)):
            raise FileNotFoundError(
                "Model artifacts not found. Please run `python train_model.py` first."
            )

        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
        with open(clf_path, "rb") as f:
            classifier = pickle.load(f)
        with open(labels_path, "rb") as f:
            labels = pickle.load(f)

        return vectorizer, classifier, labels

    # ---------- Intent Recognition ----------
    def predict_intent(self, message: str):
        """Returns (tag, confidence) for the given raw user message."""
        cleaned = preprocess(message)
        X = self.vectorizer.transform([cleaned])

        probs = self.classifier.predict_proba(X)[0]
        classes = self.classifier.classes_

        best_idx = probs.argmax()
        tag = classes[best_idx]
        confidence = float(probs[best_idx])

        return tag, confidence

    # ---------- Conversational Flow ----------
    def _get_session(self, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": [],
                "last_intent": None,
            }
        return self.sessions[session_id]

    def _rule_based_overrides(self, message: str):
        """
        Handles small-talk / real-time queries that don't need ML
        classification (e.g. current time/date) for extra responsiveness.
        """
        lowered = message.lower()
        if "time" in lowered and "what" in lowered:
            return f"The current time is {datetime.now().strftime('%I:%M %p')}."
        if "date" in lowered and ("what" in lowered or "today" in lowered):
            return f"Today's date is {datetime.now().strftime('%B %d, %Y')}."
        return None

    def get_response(self, message: str, session_id: str = "default"):
        """
        Main entry point: takes a raw user message and returns a dict with
        the bot's reply, detected intent, and confidence score.
        """
        message = (message or "").strip()
        session = self._get_session(session_id)

        if not message:
            return {
                "response": "Please type a message so I can help you!",
                "intent": "empty_input",
                "confidence": 0.0,
            }

        # Quick rule-based real-time answers (date/time)
        override = self._rule_based_overrides(message)
        if override:
            session["history"].append((message, override))
            return {"response": override, "intent": "realtime_info", "confidence": 1.0}

        tag, confidence = self.predict_intent(message)

        if confidence < CONFIDENCE_THRESHOLD:
            tag = "fallback"

        responses = self.responses_map.get(tag, self.responses_map.get("fallback"))
        reply = random.choice(responses)

        session["last_intent"] = tag
        session["history"].append((message, reply))

        return {
            "response": reply,
            "intent": tag,
            "confidence": round(confidence, 3),
        }


# Singleton instance used by the Flask app
_engine_instance = None


def get_engine():
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ChatbotEngine()
    return _engine_instance
