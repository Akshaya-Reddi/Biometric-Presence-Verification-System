from sqlalchemy.orm import Session
from app.models.attendance import Attendance, AttendanceStatus
from app.services.audit_service import log_event
from app.services.thresholds import LIVENESS_THRESHOLD, FACE_MATCH_THRESHOLD, STABILITY_THRESHOLD
from app.services.replay_guard import replay_protection

from app.services.calibration import (
    adaptive_face_threshold,
    adaptive_liveness_threshold,
    combined_confidence
)

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
    *,
    user_id,
    session_id,
    match_confidence: float,
    liveness_score: float,
    stability_score: float,
    attendance_id=None
):

    # Duplicate check
    if already_marked(db, user_id, session_id):
        log_event(db, attendance_id, "DUPLICATE_CHECK", "Duplicate attendance attempt")
        return AttendanceStatus.DUPLICATE, "Already marked present"

    face_threshold = adaptive_face_threshold(stability_score)
    live_threshold = adaptive_liveness_threshold(stability_score)
    # Liveness
    if liveness_score < LIVENESS_THRESHOLD:
        log_event(db, attendance_id, "LIVENESS", f"Failed: {liveness_score}")
        return AttendanceStatus.REJECTED, "Liveness failed"

    # Face match
    if match_confidence < FACE_MATCH_THRESHOLD:
        log_event(db, attendance_id, "FACE_MATCH", f"Failed: {match_confidence}")
        return AttendanceStatus.REJECTED, "Face mismatch"

    # Stability
    if stability_score < STABILITY_THRESHOLD:
        log_event(db, attendance_id, "STABILITY", f"Failed: {stability_score}")
        return AttendanceStatus.REJECTED, "Low stability"
    
    # 0️ Replay guard
    if not replay_protection(user_id, session_id):
        return AttendanceStatus.REJECTED, "Replay detected"

    confidence_score = combined_confidence(
        match_confidence,
        liveness_score,
        stability_score
    )
    
    log_event(db, attendance_id, "SUCCESS", "Attendance confirmed")

    return AttendanceStatus.PRESENT, f"Attendance confirmed (confidence={confidence_score:.2f})"
