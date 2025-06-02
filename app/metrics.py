from prometheus_client import Counter, Summary, generate_latest, CONTENT_TYPE_LATEST
from fastapi import APIRouter, Response
import time

metrics_router = APIRouter()

REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
PREDICTION_COUNTER = Counter('prediction_count', 'Number of predictions made')
LOW_CONFIDENCE_COUNTER = Counter('low_confidence_predictions', 'Predictions with confidence < 0.5')

@metrics_router.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)