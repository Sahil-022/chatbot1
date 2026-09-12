"""
app.py
------
Flask web server for the AI Chatbot Assistant.

Routes:
    GET  /              -> Renders the chat UI
    POST /chat          -> Accepts a user message (JSON) and returns the bot's reply
    GET  /health         -> Simple health-check endpoint

Run with:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import uuid
from flask import Flask, render_template, request, jsonify, session

from chatbot_engine import get_engine

app = Flask(__name__)
app.secret_key = "replace-this-with-a-secure-random-secret-key"

engine = get_engine()


@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "")

    session_id = session.get("session_id", "default")

    result = engine.get_response(user_message, session_id=session_id)

    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
