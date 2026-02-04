import uuid
from datetime import datetime
from sqlalchemy import Column, ForeignKey, LargeBinary, Float, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database.base import Base

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    embedding = Column(LargeBinary, nullable=False)
    quality_score = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", backref="embeddings")
