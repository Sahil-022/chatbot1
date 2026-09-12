"""
nlp_utils.py
------------
Text preprocessing utilities used across training and inference.
Uses simple, dependency-light NLP techniques:
 - lowercasing
 - punctuation removal
 - tokenization
 - stopword removal
 - lemmatization (via NLTK WordNet)
"""

import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Ensure required NLTK data is available (downloaded once, cached locally).
def _ensure_nltk_data():
    # Only wordnet/omw (lemmatization) and stopwords are actually required;
    # tokenization is handled by our own lightweight whitespace tokenizer below,
    # so we deliberately don't depend on NLTK's punkt tokenizer models.
    resources = {
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
        "corpora/stopwords": "stopwords",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except Exception:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                pass


_ensure_nltk_data()

try:
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    # Fallback minimal stopword list if NLTK data isn't available offline.
    STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "am", "i", "you", "he",
        "she", "it", "we", "they", "to", "of", "in", "on", "at", "for", "with",
        "do", "does", "did", "and", "or", "but", "my", "your", "me", "this",
        "that", "what", "how", "can", "could", "please"
    }

_lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase and strip punctuation/extra whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    """Simple whitespace tokenizer (robust even without NLTK punkt)."""
    return clean_text(text).split()


def preprocess(text: str) -> str:
    """
    Full preprocessing pipeline:
    clean -> tokenize -> remove stopwords -> lemmatize -> rejoin
    Returns a single normalized string, ideal for TF-IDF vectorization.
    """
    tokens = tokenize(text)
    processed_tokens = []
    for tok in tokens:
        if tok in STOPWORDS:
            continue
        try:
            lemma = _lemmatizer.lemmatize(tok)
        except Exception:
            lemma = tok
        processed_tokens.append(lemma)

    # Avoid returning an empty string (keeps at least original tokens)
    if not processed_tokens:
        processed_tokens = tokens

    return " ".join(processed_tokens)
