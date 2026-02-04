from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List

from app.database.deps import get_db
from app.repositories.face_embedding_repo import store_embedding

router = APIRouter()


# Request schema (JSON, not multipart)
class EnrollmentRequest(BaseModel):
    user_id: str
    embedding_vector: List[float]
    stability_score: float


@router.post("/enroll")
def enroll_user(
    payload: EnrollmentRequest,
    db: Session = Depends(get_db)
):
    store_embedding(
        db=db,
        user_id=payload.user_id,
        embedding_vector=payload.embedding_vector,
        quality_score=payload.stability_score
    )

    return {
        "status": "success",
        "message": "Biometric enrollment stored securely"
    }
