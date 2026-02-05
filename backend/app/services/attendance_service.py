from sqlalchemy.orm import Session
from app.models.attendance import Attendance, AttendanceStatus


def already_marked(db: Session, user_id, session_id) -> bool:
    record = (
        db.query(Attendance)
        .filter(
            Attendance.user_id == user_id,
            Attendance.session_id == session_id,
            Attendance.status == AttendanceStatus.PRESENT
        )
        .first()
    )
    return record is not None


def decide_attendance(
    db: Session,
    user_id: str,
    session_id: str,
    match_confidence: float,
    liveness_score: float,
    stability_score: float
):
    # Thresholds (can later move to config)
    FACE_THRESHOLD = 0.80
    LIVENESS_THRESHOLD = 0.70
    STABILITY_THRESHOLD = 0.60

    # Duplicate check
    if already_marked(db, user_id, session_id):
        return AttendanceStatus.REJECTED, "Duplicate attendance"

    # Face match check
    if match_confidence < FACE_THRESHOLD:
        return AttendanceStatus.REJECTED, "Face match too low"

    # Liveness check
    if liveness_score < LIVENESS_THRESHOLD:
        return AttendanceStatus.REJECTED, "Liveness check failed"

    # Stability check
    if stability_score < STABILITY_THRESHOLD:
        return AttendanceStatus.REJECTED, "Device not stable"

    return AttendanceStatus.PRESENT, "Attendance marked successfully"
