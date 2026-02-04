from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import numpy as np

from app.database.deps import get_db
from app.services.verification_service import verify_identity
from app.schemas.verification import VerifyRequest

router = APIRouter()

@router.post("/verify")
def verify_user(
    payload: VerifyRequest,
    db: Session = Depends(get_db)
):
    embedding = np.array(payload.embedding_vector, dtype="float32")

    result = verify_identity(db, embedding)

    return result
