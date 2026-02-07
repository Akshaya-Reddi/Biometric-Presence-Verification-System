import uuid
from datetime import datetime
from sqlalchemy import Column, Float, DateTime
from sqlalchemy.dialects.postgresql import UUID
from app.database.base_class import Base

class ConfidenceLog(Base):
    __tablename__ = "confidence_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
