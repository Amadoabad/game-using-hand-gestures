from pydantic import BaseModel, Field
from typing import List


class Landmark(BaseModel):
    x: float
    y: float
    z: float

class PredictionRequest(BaseModel):
    landmarks: List[Landmark] 
    

class PredictionResponse(BaseModel):
    gesture: str
    confidence: float