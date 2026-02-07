from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.models.confidence_log import ConfidenceLog

DRIFT_THRESHOLD = 0.15
WINDOW_SIZE = 5


def detect_confidence_drift(db: Session, user_id):

    logs = (
        db.query(ConfidenceLog)
        .filter(ConfidenceLog.user_id == user_id)
        .order_by(desc(ConfidenceLog.created_at))
        .limit(WINDOW_SIZE)
        .all()
    )

    if len(logs) < WINDOW_SIZE:
        return False, 0.0

    confidences = [log.confidence for log in logs]

    drift = confidences[0] - confidences[-1]

    if drift < -DRIFT_THRESHOLD:
        return True, drift

    return False, drift
