from fastapi import FastAPI
from app.schema import PredictionRequest, PredictionResponse
from app.model import GestureClassifier
import numpy as np

app = FastAPI()
classifier = GestureClassifier()

@app.get("/")
def root():
    return {"message": "Welcome to the Hand Gesture Recognition API"}

@app.post("/predict", response_model=PredictionResponse)
def predict_gesture(request: PredictionRequest):
    
    input_array = np.array(request.landmarks).reshape(1, -1)

    gesture = classifier.predict(input_array)
    
    return PredictionResponse(gesture=gesture)

