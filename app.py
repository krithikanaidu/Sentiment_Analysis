import os
import re
from flask import Flask, request, jsonify, render_template 
import joblib

app = Flask(__name__)

# Load the updated model
model = joblib.load("sa2_model.pkl")

# Text Preprocessing Function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    return text.strip()

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review_text = data.get("review")

    if not review_text:
        return jsonify({'error': 'Review Text not available'}), 400

    # Preprocess input text
    cleaned_text = preprocess_text(review_text)

    # Make prediction
    prediction = model.predict([cleaned_text])[0]

    # Map predictions to sentiment labels
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment = sentiment_map.get(prediction, "Unknown")

    return jsonify({'review': review_text, 'sentiment': sentiment})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
