from sqlalchemy.orm import Session
from app.models.face_embedding import FaceEmbedding
from app.utils.embedding_codec import vector_to_bytes, bytes_to_vector
import uuid


def store_embedding(
    db: Session,
    user_id: str,
    embedding_vector,
    quality_score: float
):
    encrypted_blob = vector_to_bytes(embedding_vector)

    record = FaceEmbedding(
        user_id=uuid.UUID(user_id),
        embedding=encrypted_blob,
        quality_score=quality_score
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return record


def fetch_user_embeddings(db: Session, user_id: str):
    records = (
        db.query(FaceEmbedding)
        .filter(FaceEmbedding.user_id == uuid.UUID(user_id))
        .all()
    )

    return [
        bytes_to_vector(record.embedding)
        for record in records
    ]


def fetch_all_embeddings(db: Session):
    records = db.query(FaceEmbedding).all()

    return [
        {
            "user_id": str(record.user_id),
            "vector": bytes_to_vector(record.embedding)
        }
        for record in records
    ]
