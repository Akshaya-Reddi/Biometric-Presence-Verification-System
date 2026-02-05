from pydantic import BaseModel
from typing import List

class AttendanceRequest(BaseModel):
    user_id: str
    session_id: str
    device_id: str
    timestamp: int
    signature: str

    embeddings: List[List[float]]
    liveness_scores: List[float]
