from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database.deps import get_db
from app.services.attendance_service import decide_attendance
import uuid

router = APIRouter()

@router.post("/debug/attendance")
def debug_attendance(
    match_confidence: float,
    liveness_score: float,
    stability_score: float,
    db: Session = Depends(get_db)
):
    status, reason = decide_attendance(
        db=db,
        user_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        match_confidence=match_confidence,
        liveness_score=liveness_score,
        stability_score=stability_score,
        attendance_id=uuid.uuid4()
    )

    return {"status": status.value, "reason": reason}
