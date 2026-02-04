import uuid
import enum
from datetime import datetime
from sqlalchemy import Column, ForeignKey, Enum, Float, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database.base import Base

class AttendanceStatus(enum.Enum):
    PRESENT = "present"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"

class Attendance(Base):
    __tablename__ = "attendance"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)

    status = Column(Enum(AttendanceStatus), nullable=False)
    confidence = Column(Float)
    stability = Column(Float)
    liveness_score = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")
    session = relationship("Session")
