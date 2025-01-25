import os
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load your model
model = joblib.load("sa2_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review_text = data.get("review")

    if not review_text:
        return jsonify({'error': 'Review Text not available'}), 400

    prediction = model.predict([review_text])[0]
    sentiment = 'Positive' if prediction == 2 else "Negative"

    return jsonify({'review': review_text, 'sentiment': sentiment})

if __name__ == '__main__':
    # Use the PORT environment variable or default to 8000
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
