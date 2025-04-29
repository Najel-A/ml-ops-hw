from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Load model
model = load('model.pkl')

# Input schema
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(features: IrisFeatures):
    features_array = np.array([
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]).reshape(1, -1)
    
    prediction = model.predict(features_array)
    return {"prediction": int(prediction[0])}

@app.get("/")
def read_root():
    return {"message": "Iris Prediction API"}