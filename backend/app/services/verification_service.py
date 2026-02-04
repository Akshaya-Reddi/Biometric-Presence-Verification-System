import numpy as np
from sqlalchemy.orm import Session
from app.repositories.face_embedding_repo import fetch_all_embeddings
from app.services.vector_sync import VectorIndex

MATCH_THRESHOLD = 0.75   # tune later

def verify_identity(db: Session, live_embedding: np.ndarray):
    """
    Returns:
    {
        status: "match" | "no_match",
        user_id: str | None,
        confidence: float
    }
    """

    # 1. Normalize live embedding
    live_embedding = live_embedding.astype("float32")
    live_embedding = live_embedding / np.linalg.norm(live_embedding)

    # 2. Search FAISS
    index = VectorIndex()
    results = index.search(live_embedding, top_k=1)

    if not results:
        return {
            "status": "no_match",
            "user_id": None,
            "confidence": 0.0
        }

    top = results[0]

    if top["confidence"] < MATCH_THRESHOLD:
        return {
            "status": "no_match",
            "user_id": None,
            "confidence": float(top["confidence"])
        }

    return {
        "status": "match",
        "user_id": top["user_id"],
        "confidence": float(top["confidence"])
    }
