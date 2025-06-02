import time
import numpy as np
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.schema import PredictionRequest, PredictionResponse
from app.model import GestureClassifier
from app.metrics import metrics_router
from app.metrics import REQUEST_TIME, PREDICTION_COUNTER, LOW_CONFIDENCE_COUNTER



app = FastAPI()
app.include_router(metrics_router)

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
    start_time = time.time()
    
    input_array = np.array(request.landmarks).reshape(1, -1)
    gesture, confidence = classifier.predict(input_array)

    PREDICTION_COUNTER.inc()
    if confidence < 0.5:
        LOW_CONFIDENCE_COUNTER.inc()
    REQUEST_TIME.observe(time.time() - start_time)
    
    return PredictionResponse(gesture=gesture, confidence=confidence)
