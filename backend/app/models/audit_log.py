import uuid
from datetime import datetime
from sqlalchemy import Column, ForeignKey, String, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database.base import Base

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    attendance_id = Column(UUID(as_uuid=True), ForeignKey("attendance.id"))
    stage = Column(String, nullable=False)
    message = Column(String, nullable=False)
    extra_data = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    attendance = relationship("Attendance", backref="audit_logs")
