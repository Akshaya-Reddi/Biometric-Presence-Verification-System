from sqlalchemy.orm import Session
from app.models.attendance import Attendance

def create_attendance(
    db: Session,
    *,
    user_id,
    session_id,
    status,
    confidence,
    stability,
    liveness
):
    record = Attendance(
        user_id=user_id,
        session_id=session_id,
        status=status,
        confidence=confidence,
        stability=stability,
        liveness_score=liveness
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return record
