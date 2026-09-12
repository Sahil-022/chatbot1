# AI Chatbot Assistant

An AI-powered chatbot capable of understanding user queries and generating relevant, real-time responses using NLP and Machine Learning, served through a Flask web application.

## Highlights

- **Intent recognition** — user messages are cleaned, tokenized, and lemmatized, then classified into intents (greeting, order status, refund, payment, support, etc.) using a TF‑IDF + Logistic Regression model.
- **Conversational flow** — a lightweight session-based engine tracks conversation history and falls back gracefully to a clarifying response when confidence is low.
- **Real-time responses** — a Flask REST endpoint (`/chat`) returns responses instantly to the browser-based chat UI, with a couple of live rule-based answers (current date/time) layered on top of the ML classifier.
- **Clean, responsive chat UI** — HTML/CSS/JS front end with a typing indicator and live intent/confidence readout.

## Tech Stack

- **Python 3.9+**
- **NLP:** NLTK (tokenization, stopword removal, lemmatization)
- **Machine Learning:** scikit-learn (TF-IDF vectorizer + Logistic Regression classifier)
- **Backend:** Flask
- **Frontend:** HTML, CSS, vanilla JavaScript

## Project Structure

```
AI_Chatbot_Assistant/
├── app.py                 # Flask application (routes + server)
├── chatbot_engine.py       # Core engine: intent prediction + conversational flow
├── train_model.py          # Script to train the intent classification model
├── nlp_utils.py             # Text preprocessing (clean, tokenize, lemmatize)
├── intents.json             # Training data: intents, patterns, and responses
├── requirements.txt
├── README.md
├── model/                   # Generated after training (vectorizer, classifier, labels)
│   ├── vectorizer.pkl
│   ├── classifier.pkl
│   └── labels.pkl
├── templates/
│   └── index.html            # Chat UI page
└── static/
    ├── style.css
    └── script.js
```

## Setup & Run

1. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the intent classification model:**
   ```bash
   python train_model.py
   ```
   This reads `intents.json`, extracts TF-IDF features, trains a Logistic Regression classifier, prints a validation accuracy report, and saves the model artifacts into `model/`.

4. **Run the Flask app:**
   ```bash
   python app.py
   ```

5. Open your browser at **http://127.0.0.1:5000** and start chatting.

> Note: the `model/` folder already ships with pre-trained artifacts, so you can skip straight to step 4. Re-run `train_model.py` any time you edit `intents.json`.

## How It Works

1. **Preprocessing (`nlp_utils.py`):** incoming text is lowercased, stripped of punctuation, tokenized, stopwords removed, and lemmatized.
2. **Feature extraction:** the cleaned text is converted into a TF-IDF vector.
3. **Intent classification:** a Logistic Regression model predicts the most likely intent tag and a confidence score.
4. **Response generation:** if confidence is above the threshold (30%), a random response tied to that intent is returned; otherwise, the bot asks the user to rephrase (fallback intent).
5. **Conversational flow:** each browser session gets a UUID; the chatbot engine stores a short history per session, enabling future extensions like multi-turn context (e.g. "What about my last order?").

## Extending the Chatbot

- **Add new intents:** open `intents.json`, add a new object with a `tag`, a list of example `patterns`, and possible `responses`, then re-run `python train_model.py`.
- **Improve accuracy:** add more varied training patterns per intent (10–20 examples per intent works well for small bots).
- **Swap the classifier:** `train_model.py` uses Logistic Regression for speed and interpretability; you can swap in `sklearn.svm.SVC(probability=True)` or a small neural network without changing the rest of the pipeline.
- **Persistent storage:** swap the in-memory `sessions` dict in `chatbot_engine.py` for a database (e.g., Redis/SQLite) for production use across multiple server workers.

## Resume Bullet Alignment

This project directly demonstrates:
- Designing and developing an AI-powered chatbot that understands user queries and generates relevant responses using NLP techniques.
- Implementing intent recognition and conversational flow to handle common user queries with automated, real-time responses.
- Technologies used: **Python, NLP, Machine Learning, Flask**.
