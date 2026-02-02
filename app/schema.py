# app/schema.py

from pydantic import BaseModel
from typing import List

class ReviewRequest(BaseModel):
    review: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "review": "The app crashes when I try to upload photos! Please fix this bug."
            }
        }

class ReviewResponse(BaseModel):
    review: str
    predicted_labels: List[str]
    probabilities: List[float]
    label_names: List[str]
