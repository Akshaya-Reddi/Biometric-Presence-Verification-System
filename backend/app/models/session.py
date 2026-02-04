import uuid
from datetime import datetime
from sqlalchemy import Column, ForeignKey, Boolean, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database.base import Base

class Session(Base):
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    class_id = Column(UUID(as_uuid=True), ForeignKey("classes.id"), nullable=False)

    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    is_open = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    class_ = relationship("Class", backref="sessions")
