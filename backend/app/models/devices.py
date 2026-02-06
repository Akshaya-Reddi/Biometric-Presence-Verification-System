import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database.base_class import Base

class Device(Base):
    __tablename__ = "devices"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    device_id = Column(String, unique=True, nullable=False)
    device_name = Column(String)
    registered_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")
