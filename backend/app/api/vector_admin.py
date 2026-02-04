from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database.deps import get_db
from app.repositories.face_embedding_repo import fetch_all_embeddings
from app.services.vector_sync import VectorIndex

router = APIRouter()
vector_index = VectorIndex()

@router.post("/admin/vector/sync")
def sync_vectors(db: Session = Depends(get_db)):
    embeddings = fetch_all_embeddings(db)
    vector_index.load(embeddings)

    return {
        "status": "success",
        "loaded_vectors": len(embeddings)
    }
