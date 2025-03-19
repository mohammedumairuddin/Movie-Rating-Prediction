import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.fillna(df.mean(), inplace=True)
    df = pd.get_dummies(df, columns=['Genre', 'Director', 'Actor'], drop_first=True)
    X = df.drop(columns=['Rating'])
    y = df['Rating']
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = preprocess_data("data/IMDb_Movies_India.csv")

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

joblib.dump(model, "models/movie_rating_predictor.pkl")
