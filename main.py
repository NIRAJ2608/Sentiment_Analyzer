from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys

# ----------------------
# Load Model & Tokenizer
# ----------------------
MODEL_PATH = "best_bidirectional_lstm.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN = 100  # same maxlen used during training

# Patch for pickle compatibility (old keras -> new tf.keras)
sys.modules['keras.preprocessing.text'] = tf.keras.preprocessing.text
sys.modules['keras.preprocessing.sequence'] = tf.keras.preprocessing.sequence

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load tokenizer
with open(TOKENIZER_PATH, "rb") as handle:
    tokenizer = pickle.load(handle)

# ----------------------
# Flask App
# ----------------------
app = Flask(__name__)

def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    pred = model.predict(padded)[0][0]  # binary classification
    sentiment = "POSITIVE" if pred > 0.5 else "NEGATIVE"
    confidence = round(pred if pred > 0.5 else (1 - pred), 2)  # return as 0–1 float
    return sentiment, confidence

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_text = data.get("text", "")
    if not user_text:
        return jsonify({"error": "No text provided"}), 400
    
    sentiment, confidence = predict_sentiment(user_text)
    return jsonify({
        "sentiment": sentiment,
        "confidence": float(confidence)
    })

if __name__ == "__main__":
    app.run(debug=True)
