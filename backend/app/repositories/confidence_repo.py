from sqlalchemy.orm import Session
from app.models.confidence_log import ConfidenceLog

def log_confidence(db: Session, user_id, confidence):
    entry = ConfidenceLog(
        user_id=user_id,
        confidence=confidence
    )
    db.add(entry)
