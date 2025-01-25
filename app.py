from flask import Flask, jsonify, request, render_template
import joblib

app = Flask(__name__)

model = joblib.load('sa2_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.json
    review_text = data.get('review')

    if not review_text:
        return jsonify({'error': 'No review text detected'}),400

    prediction = model.predict([review_text])[0]
    sentiment = 'positive' if prediction == 2 else 'negative'

    return jsonify({'review' : review_text, 'sentiment' : sentiment })

if __name__ == '__main__':
    app.run(debug = True)  
    

 