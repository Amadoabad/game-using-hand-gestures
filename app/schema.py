from pydantic import BaseModel
from typing import List

class LandMark(BaseModel):
    x: float
    y: float
    z: float
    
class PredictionRequest(BaseModel):
    landmarks: List[LandMark]


class PredictionResponse(BaseModel):
    gesture: str