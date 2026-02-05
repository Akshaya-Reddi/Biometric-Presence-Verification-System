from sqlalchemy.orm import Session
from app.models.attendance import Attendance, AttendanceStatus
from sqlalchemy import exists

def already_marked(db, user_id, session_id):
    return db.query(
        exists().where(
            Attendance.user_id == user_id,
            Attendance.session_id == session_id,
            Attendance.status == AttendanceStatus.PRESENT
        )
    ).scalar()