from sqlalchemy.orm import Session
from app.models.attendance import Attendance

def already_marked(db: Session, user_id, session_id) -> bool:
    record = (
        db.query(Attendance)
        .filter(
            Attendance.user_id == user_id,
            Attendance.session_id == session_id,
            Attendance.status == "present"
        )
        .first()
    )

    return record is not None
