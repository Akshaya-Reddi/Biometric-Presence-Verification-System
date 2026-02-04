from pydantic import BaseModel
from typing import List

class VerifyRequest(BaseModel):
    embedding_vector: List[float]
