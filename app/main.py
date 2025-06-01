import time
import numpy as np
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.schema import PredictionRequest, PredictionResponse
from app.model import GestureClassifier



app = FastAPI()

class LatencyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        print(f"[Server Latency] {request.method} {request.url.path} took {process_time:.2f} ms")
        return response
    
app.add_middleware(LatencyMiddleware)
classifier = GestureClassifier()


@app.get("/")
def root():
    return {"message": "Welcome to the Hand Gesture Recognition API"}

@app.post("/predict", response_model=PredictionResponse)
def predict_gesture(request: PredictionRequest):
    
    input_array = np.array(request.landmarks).reshape(1, -1)
    gesture, confidence = classifier.predict(input_array)
    return PredictionResponse(gesture=gesture, confidence=confidence)

