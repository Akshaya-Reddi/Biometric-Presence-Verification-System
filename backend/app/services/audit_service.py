import uuid
from sqlalchemy.orm import Session
from app.models.audit_log import AuditLog


def log_event(
    db: Session,
    attendance_id,
    stage: str,
    message: str,
    extra_data: str | None = None
):
    log = AuditLog(
        id=uuid.uuid4(),
        attendance_id=attendance_id,
        stage=stage,
        message=message,
        extra_data=extra_data
    )

    db.add(log)
