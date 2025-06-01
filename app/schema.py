from pydantic import BaseModel, Field
from typing import List

# A list of exactly 63 floats
class PredictionRequest(BaseModel):
    landmarks: List[float] = Field(..., min_items=63, max_items=63)
    
class PredictionResponse(BaseModel):
    gesture: str
