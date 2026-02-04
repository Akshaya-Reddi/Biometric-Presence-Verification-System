import uuid
from sqlalchemy.exc import IntegrityError
from backend.app.database.session import SessionLocal
from backend.app.models.attendance import Attendance
from backend.app.models.audit_log import AuditLog

from sqlalchemy.orm import Session
from app.models.attendance import Attendance, AttendanceStatus
from app.services.duplicate_guard import already_marked
from app.services.thresholds import (
    FACE_MATCH_THRESHOLD,
    LIVENESS_THRESHOLD,
    STABILITY_THRESHOLD
)

def mark_attendance(user_id: str, session_id: str, method="biometric"):
    db = SessionLocal()

    try:
        attendance = Attendance(
            id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id,
            method=method
        )

        db.add(attendance)

        audit = AuditLog(
            id=str(uuid.uuid4()),
            actor_id=user_id,
            action="ATTENDANCE_MARKED",
            details=f"Session={session_id}, method={method}"
        )
        db.add(audit)

        db.commit()

        return {
            "status": "marked",
            "user_id": user_id,
            "session_id": session_id
        }

    except IntegrityError:
        db.rollback()
        return {
            "status": "already_marked",
            "user_id": user_id,
            "session_id": session_id
        }

    finally:
        db.close()

def decide_attendance(
    db: Session,
    *,
    user_id,
    session_id,
    match_confidence: float,
    liveness_score: float,
    stability_score: float
):
    # 1️Duplicate check
    if already_marked(db, user_id, session_id):
        return AttendanceStatus.DUPLICATE, "Already marked present"

    # 2️Liveness gate
    if liveness_score < LIVENESS_THRESHOLD:
        return AttendanceStatus.REJECTED, "Liveness failed"

    # 3️Face match gate
    if match_confidence < FACE_MATCH_THRESHOLD:
        return AttendanceStatus.REJECTED, "Face mismatch"

    # 4️Stability gate
    if stability_score < STABILITY_THRESHOLD:
        return AttendanceStatus.REJECTED, "Low stability"

    # Passed all checks
    return AttendanceStatus.PRESENT, "Attendance confirmed"