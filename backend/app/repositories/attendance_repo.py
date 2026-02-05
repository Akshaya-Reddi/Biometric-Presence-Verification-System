from sqlalchemy.orm import Session
from app.models.attendance import Attendance
from sqlalchemy.exc import IntegrityError

def create_attendance(
    db: Session,
    *,
    id,
    user_id,
    session_id,
    status,
    confidence,
    stability,
    liveness
):
    try:
        record = Attendance(
            id=id,
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
        db.flush() #important

        return record

    except IntegrityError:
        db.rollback()
        raise ValueError("Duplicate attendance")