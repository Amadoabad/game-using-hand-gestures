from fastapi import FastAPI
from app.schema import PredictionRequest, PredictionResponse
from app.model import GestureClassifier

app = FastAPI()
classifier = GestureClassifier()

@app.get("/")
def root():
    return {"message": "Welcome to the Hand Gesture Recognition API"}

@app.post("/predict", response_model=PredictionResponse)
def predict_gesture(request: PredictionRequest):
    """
    Predicts the hand gesture from the landmarks of the hand
    Arguments:
        request -- request containing the landmarks of the hand
    Returns:
        predicted gesture
    """
    landmarks = request.landmarks
    hand_landmarks = [landmark.model_dump() for landmark in landmarks]
    
    gesture = classifier.predict(hand_landmarks)
    
    return PredictionResponse(gesture=gesture)

