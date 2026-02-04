from app.repositories.face_embedding_repo import store_embedding

def enroll_face(db, user_id, identity_vector, stability_score):
    store_embedding(
        db=db,
        user_id=user_id,
        embedding_vector=identity_vector,
        quality_score=stability_score
    )
