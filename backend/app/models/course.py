import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime
from sqlalchemy.dialects.postgresql import UUID
from app.database.base_class import Base

class Class(Base):
    __tablename__ = "classes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    class_code = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
