from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("models/movie_rating_predictor.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    prediction = model.predict(np.array([data]))
    return jsonify({'predicted_rating': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
