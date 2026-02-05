from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.deps import get_db
from app.services.attendance_service import decide_attendance
from app.repositories.attendance_repo import create_attendance
from app.models.attendance import AttendanceStatus

router = APIRouter()

@router.post("/attendance/mark")
def mark_attendance(
    user_id: str,
    session_id: str,
    match_confidence: float,
    liveness_score: float,
    stability_score: float,
    db: Session = Depends(get_db)
):
    status, reason = decide_attendance(
        db=db,
        user_id=user_id,
        session_id=session_id,
        match_confidence=match_confidence,
        liveness_score=liveness_score,
        stability_score=stability_score
    )

    record = create_attendance(
        db=db,
        user_id=user_id,
        session_id=session_id,
        status=status,
        confidence=match_confidence,
        stability=stability_score,
        liveness=liveness_score
    )

    return {
        "status": status.value,
        "reason": reason,
        "attendance_id": str(record.id)
    }
