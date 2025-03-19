import joblib
import numpy as np

model = joblib.load("models/movie_rating_predictor.pkl")

sample_data = np.array([[5, 10, 1, 7]])  # Example input
predicted_rating = model.predict(sample_data)
print(f"Predicted Movie Rating: {predicted_rating[0]}")
